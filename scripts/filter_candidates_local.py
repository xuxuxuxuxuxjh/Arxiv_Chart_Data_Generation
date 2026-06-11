#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image

from pipeline_common import WORK_ROOT, image_info, iter_jsonl, metadata_keyword_score, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hard-filter arXiv figure candidates.")
    parser.add_argument(
        "--input", type=Path, default=WORK_ROOT / "candidates_2020_2025.jsonl"
    )
    parser.add_argument(
        "--out", type=Path, default=WORK_ROOT / "candidates_2020_2025.filtered.jsonl"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=WORK_ROOT / "reports" / "local_filter_report.json",
    )
    parser.add_argument("--min-short-side", type=int, default=256)
    parser.add_argument("--min-area", type=int, default=80000)
    parser.add_argument("--min-caption-length", type=int, default=10)
    parser.add_argument("--min-aspect", type=float, default=0.15)
    parser.add_argument("--max-aspect", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--chunk-size", type=int, default=4096)
    return parser.parse_args()


def hard_filter(record: dict[str, Any], args: argparse.Namespace) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    extra: dict[str, Any] = {}
    if record.get("image_kind") != "merged" and record.get("status") != "success":
        reasons.append("status_not_success")
    image_path = Path(record.get("image_path") or "")
    if not image_path.exists():
        reasons.append("image_missing")
        return False, reasons, extra
    if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}:
        reasons.append("unsupported_image_suffix")
    caption = record.get("caption_latex") or ""
    if len(caption.strip()) < args.min_caption_length:
        reasons.append("caption_too_short")
    try:
        width, height, fmt = image_info(image_path)
        extra.update({"width": width, "height": height, "image_format": fmt})
        short_side = min(width, height)
        area = width * height
        aspect = width / height if height else 0
        extra.update({"short_side": short_side, "area": area, "aspect_ratio": aspect})
        if short_side < args.min_short_side:
            reasons.append("short_side_too_small")
        if area < args.min_area:
            reasons.append("area_too_small")
        if aspect < args.min_aspect or aspect > args.max_aspect:
            reasons.append("aspect_ratio_out_of_range")
    except Exception as exc:
        reasons.append(f"image_decode_failed:{type(exc).__name__}")
    return not reasons, reasons, extra


def first_pass_merge_index(path: Path) -> tuple[set[tuple[str, int]], Counter, Counter]:
    merged_keys: set[tuple[str, int]] = set()
    group_single_counts: Counter[tuple[str, int]] = Counter()
    by_year_before = Counter()
    total = 0
    for record in iter_jsonl(path):
        total += 1
        if total % 100000 == 0:
            print(f"first pass read {total:,}", flush=True)
        key = (record["paper_id"], int(record["figure_index"]))
        by_year_before[str(record.get("year"))] += 1
        if (
            record.get("image_kind") == "merged"
            and 2 <= int(record.get("merged_panel_count") or 0) <= 8
        ):
            merged_keys.add(key)
        elif record.get("image_kind") == "single":
            group_single_counts[key] += 1
    by_year_before["__total__"] = total
    return merged_keys, group_single_counts, by_year_before


def precheck_record(
    record: dict[str, Any],
    args: argparse.Namespace,
    merged_keys: set[tuple[str, int]],
    group_single_counts: Counter,
) -> tuple[str, dict[str, Any] | None, list[str]]:
    key = (record["paper_id"], int(record["figure_index"]))
    if record.get("image_kind") == "single" and key in merged_keys:
        return "skip", None, ["single_dropped_due_to_merged"]
    if record.get("image_kind") == "merged" and key not in merged_keys:
        return "skip", None, ["merged_panel_count_out_of_range"]
    if record.get("image_kind") == "single" and group_single_counts[key] > 1:
        record = dict(record)
        record["image_kind"] = "single_panel"
    return "check", record, []


def process_record(record: dict[str, Any], args: argparse.Namespace) -> tuple[bool, dict[str, Any], list[str]]:
    ok, reasons, extra = hard_filter(record, args)
    record.update(extra)
    if ok:
        score, hits = metadata_keyword_score(record)
        record["metadata_chart_score"] = score
        record["metadata_keyword_hits"] = hits
    return ok, record, reasons


def process_chunk(
    chunk: list[dict[str, Any]],
    args: argparse.Namespace,
    merged_keys: set[tuple[str, int]],
    group_single_counts: Counter,
) -> tuple[list[dict[str, Any]], Counter, Counter, Counter, Counter, Counter, int]:
    kept: list[dict[str, Any]] = []
    discard_reasons = Counter()
    merge_counters = Counter()
    by_year_after = Counter()
    by_kind_after = Counter()
    metadata_score_counter = Counter()
    hard_filter_kept = 0

    to_check: list[dict[str, Any]] = []
    for record in chunk:
        action, prepared, reasons = precheck_record(record, args, merged_keys, group_single_counts)
        if reasons:
            for reason in reasons:
                discard_reasons[reason] += 1
                if reason == "single_dropped_due_to_merged":
                    merge_counters["dropped_single_due_to_merged"] += 1
        if action == "check" and prepared is not None:
            to_check.append(prepared)
            if prepared.get("image_kind") == "merged":
                merge_counters["used_merged"] += 1

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_record, record, args) for record in to_check]
        for future in as_completed(futures):
            ok, record, reasons = future.result()
            if ok:
                hard_filter_kept += 1
                score = record.get("metadata_chart_score", 0)
                if score > 0:
                    metadata_score_counter["positive"] += 1
                elif score == 0:
                    metadata_score_counter["zero"] += 1
                else:
                    metadata_score_counter["negative"] += 1
                by_year_after[str(record.get("year"))] += 1
                by_kind_after[record.get("image_kind")] += 1
                kept.append(record)
            else:
                for reason in reasons:
                    discard_reasons[reason] += 1
    kept.sort(key=lambda r: (r["year"], r["month"], r["paper_id"], r["figure_index"], r["candidate_id"]))
    return (
        kept,
        discard_reasons,
        merge_counters,
        by_year_after,
        by_kind_after,
        metadata_score_counter,
        hard_filter_kept,
    )


def main() -> int:
    args = parse_args()
    merged_keys, group_single_counts, by_year_before = first_pass_merge_index(args.input)
    discard_reasons = Counter()
    merge_counters = Counter()
    by_year_after = Counter()
    by_kind_after = Counter()
    metadata_score_counter = Counter()
    raw_count = int(by_year_before.pop("__total__", 0))
    hard_filter_kept = 0
    final_count = 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    chunk: list[dict[str, Any]] = []
    with args.out.open("w", encoding="utf-8") as out_f:
        for record in iter_jsonl(args.input):
            chunk.append(record)
            if len(chunk) < args.chunk_size:
                continue
            (
                kept,
                dr,
                mc,
                bya,
                bka,
                msc,
                hfk,
            ) = process_chunk(chunk, args, merged_keys, group_single_counts)
            processed += len(chunk)
            chunk = []
            discard_reasons.update(dr)
            merge_counters.update(mc)
            by_year_after.update(bya)
            by_kind_after.update(bka)
            metadata_score_counter.update(msc)
            hard_filter_kept += hfk
            for item in kept:
                out_f.write(json.dumps(item, ensure_ascii=False, sort_keys=True))
                out_f.write("\n")
            final_count += len(kept)
            if processed % 100000 < args.chunk_size:
                print(f"second pass processed {processed:,}; wrote {final_count:,}", flush=True)
        if chunk:
            (
                kept,
                dr,
                mc,
                bya,
                bka,
                msc,
                hfk,
            ) = process_chunk(chunk, args, merged_keys, group_single_counts)
            processed += len(chunk)
            discard_reasons.update(dr)
            merge_counters.update(mc)
            by_year_after.update(bya)
            by_kind_after.update(bka)
            metadata_score_counter.update(msc)
            hard_filter_kept += hfk
            for item in kept:
                out_f.write(json.dumps(item, ensure_ascii=False, sort_keys=True))
                out_f.write("\n")
            final_count += len(kept)

    report = {
        "input_count": raw_count,
        "hard_filter_kept": hard_filter_kept,
        "final_count_after_merged_first": final_count,
        "retention_rate": final_count / raw_count if raw_count else 0,
        "discard_reasons": dict(discard_reasons.most_common()),
        "merged_first": dict(merge_counters.most_common()),
        "by_year_before": dict(by_year_before.most_common()),
        "by_year_after": dict(by_year_after.most_common()),
        "by_kind_after": dict(by_kind_after.most_common()),
        "metadata_score": dict(metadata_score_counter.most_common()),
    }
    write_json(args.report, report)
    print(f"wrote {args.out}: {final_count} records", flush=True)
    print(f"wrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
