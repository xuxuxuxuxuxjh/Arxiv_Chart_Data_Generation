#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pipeline_common import (
    ARXIV_ROOT,
    PAPER_ID_RE,
    WORK_ROOT,
    clean_text,
    infer_month_year_from_path,
    list_dirs,
    normalize_output_image,
    read_json,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan 2020-2025 arXiv extracted figure JSON files into a manifest."
    )
    parser.add_argument("--src", type=Path, default=ARXIV_ROOT)
    parser.add_argument(
        "--charxiv", type=Path, default=WORK_ROOT / "charxiv_paper_ids.json"
    )
    parser.add_argument(
        "--out", type=Path, default=WORK_ROOT / "candidates_2020_2025.jsonl"
    )
    parser.add_argument(
        "--stats-out",
        type=Path,
        default=WORK_ROOT / "reports" / "candidate_scan_stats.json",
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--limit-buckets", type=int, default=0)
    return parser.parse_args()


def candidate_id(paper_id: str, figure_index: Any, image_kind: str, record: dict, json_path: Path) -> str:
    if image_kind == "merged":
        return f"{paper_id}_fig{int(figure_index):04d}_merged"
    idx = record.get("image_index_in_figure") or record.get("image_index") or 1
    return f"{paper_id}_fig{int(figure_index):04d}_img{int(idx):02d}"


def record_from_json(json_path: Path, charxiv_papers: set[str]) -> dict[str, Any] | None:
    try:
        record = read_json(json_path)
    except Exception as exc:
        return {"_scan_error": f"json_decode_error:{exc}", "json_path": str(json_path)}
    if not isinstance(record, dict):
        return {"_scan_error": "json_not_object", "json_path": str(json_path)}

    paper_id = str(record.get("paper_id") or json_path.parent.parent.name).strip()
    if not PAPER_ID_RE.match(paper_id):
        return None
    figure_index = record.get("figure_index")
    if figure_index is None:
        return {"_scan_error": "missing_figure_index", "json_path": str(json_path)}
    month, year, source_bucket = infer_month_year_from_path(json_path)
    if year is None or year < 2020 or year > 2025:
        return None
    is_merged = bool(record.get("is_merged_figure")) or record.get("record_type") == "merged_figure" or "_merged_" in json_path.name
    image_kind = "merged" if is_merged else "single"
    image_path = normalize_output_image(record, json_path)
    labels = record.get("labels") or []
    if not isinstance(labels, list):
        labels = [labels]

    return {
        "candidate_id": candidate_id(paper_id, figure_index, image_kind, record, json_path),
        "paper_id": paper_id,
        "year": year,
        "month": month,
        "source_bucket": source_bucket,
        "figure_index": int(figure_index),
        "image_kind": image_kind,
        "image_path": image_path,
        "json_path": str(json_path),
        "caption_latex": clean_text(record.get("caption_latex")),
        "reference_paragraphs_latex": record.get("reference_paragraphs_latex") or [],
        "labels": labels,
        "figure_tex": clean_text(record.get("figure_tex")),
        "status": record.get("status", "success" if is_merged else record.get("status")),
        "source_effective_ext": record.get("source_effective_ext", ""),
        "was_converted_to_png": record.get("was_converted_to_png", None),
        "merged_panel_count": record.get("merged_panel_count"),
        "merged_source_image_count": record.get("merged_source_image_count"),
        "merge_layout": record.get("merge_layout"),
        "is_charxiv_paper": paper_id in charxiv_papers,
    }


def iter_candidate_jsons(paper_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for child in sorted(paper_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        if name.endswith("_extracted_figs") or name.endswith("_extracted_figs_merged"):
            paths.extend(sorted(child.glob("*.json")))
    return paths


def scan_bucket(bucket: Path, charxiv_papers: set[str]) -> tuple[list[dict], dict]:
    stats = {
        "bucket": bucket.name,
        "records": 0,
        "errors": Counter(),
        "papers": 0,
        "single": 0,
        "merged": 0,
    }
    records: list[dict] = []
    for tar_dir in list_dirs(bucket):
        if not tar_dir.name.startswith("arXiv_src_"):
            continue
        for paper_dir in list_dirs(tar_dir):
            if not PAPER_ID_RE.match(paper_dir.name):
                continue
            stats["papers"] += 1
            for json_path in iter_candidate_jsons(paper_dir):
                item = record_from_json(json_path, charxiv_papers)
                if item is None:
                    continue
                if "_scan_error" in item:
                    stats["errors"][item["_scan_error"]] += 1
                    continue
                stats["records"] += 1
                stats[item["image_kind"]] += 1
                records.append(item)
    stats["errors"] = dict(stats["errors"].most_common())
    return records, stats


def load_charxiv_papers(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(str(item) for item in data.get("paper_ids", []))


def main() -> int:
    args = parse_args()
    charxiv_papers = load_charxiv_papers(args.charxiv)
    buckets = [
        path
        for path in list_dirs(args.src)
        if path.name.startswith("arxiv_")
        and len(path.name) == len("arxiv_2001")
        and 2020 <= 2000 + int(path.name[-4:-2]) <= 2025
    ]
    if args.limit_buckets:
        buckets = buckets[: args.limit_buckets]
    print(f"scanning {len(buckets)} buckets with {args.workers} workers", flush=True)

    all_records: list[dict] = []
    bucket_stats: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(scan_bucket, bucket, charxiv_papers): bucket for bucket in buckets
        }
        for future in as_completed(futures):
            bucket = futures[future]
            records, stats = future.result()
            print(
                f"{bucket.name}: {len(records)} records ({stats.get('single', 0)} single, {stats.get('merged', 0)} merged)",
                flush=True,
            )
            all_records.extend(records)
            bucket_stats.append(stats)

    all_records.sort(key=lambda r: (r["year"], r["month"], r["paper_id"], r["figure_index"], r["image_kind"], r["candidate_id"]))
    count = write_jsonl(args.out, all_records)

    stats: dict[str, Any] = {
        "total_candidates": count,
        "bucket_stats": sorted(bucket_stats, key=lambda x: x["bucket"]),
        "by_year": Counter(str(r["year"]) for r in all_records),
        "by_month": Counter(str(r["month"]) for r in all_records),
        "by_kind": Counter(r["image_kind"] for r in all_records),
        "by_suffix": Counter(Path(r["image_path"]).suffix.lower() for r in all_records),
        "caption_length_buckets": Counter(),
        "unique_papers": len({r["paper_id"] for r in all_records}),
        "charxiv_candidates": sum(1 for r in all_records if r["is_charxiv_paper"]),
    }
    for record in all_records:
        length = len(record.get("caption_latex") or "")
        bucket = f"{min(length // 100, 20) * 100}-{min(length // 100, 20) * 100 + 99}"
        if length >= 2000:
            bucket = "2000+"
        stats["caption_length_buckets"][bucket] += 1
    serializable = {}
    for key, value in stats.items():
        if isinstance(value, Counter):
            serializable[key] = dict(value.most_common())
        else:
            serializable[key] = value
    write_json(args.stats_out, serializable)
    print(f"wrote {args.out}: {count} candidates", flush=True)
    print(f"wrote {args.stats_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
