#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from pipeline_common import WORK_ROOT, iter_jsonl, simple_phash, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deduplicate accepted chart candidates.")
    parser.add_argument(
        "--input",
        type=Path,
        default=WORK_ROOT / "candidates_2020_2025.chart_classified.jsonl",
    )
    parser.add_argument(
        "--out", type=Path, default=WORK_ROOT / "candidates_2020_2025.deduped.jsonl"
    )
    parser.add_argument(
        "--duplicates",
        type=Path,
        default=WORK_ROOT / "logs" / "duplicate_chart_candidates.jsonl",
    )
    parser.add_argument(
        "--report", type=Path, default=WORK_ROOT / "reports" / "dedup_report.json"
    )
    parser.add_argument("--include-weak", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = []
    for record in iter_jsonl(args.input):
        classifier = record.get("classifier", {})
        if classifier.get("accepted") or (args.include_weak and classifier.get("weak_accept")):
            records.append(record)

    records.sort(key=lambda r: (-float(r.get("classifier", {}).get("chart_confidence") or 0), r["candidate_id"]))
    kept = []
    duplicates = []
    seen_keys = set()
    phash_groups: dict[str, list[dict]] = defaultdict(list)
    counters = Counter()
    for record in records:
        key = (record["paper_id"], int(record["figure_index"]), record["image_kind"])
        if key in seen_keys:
            record["duplicate_reason"] = "paper_figure_kind_duplicate"
            duplicates.append(record)
            counters["paper_figure_kind_duplicate"] += 1
            continue
        seen_keys.add(key)
        try:
            phash = simple_phash(Path(record["image_path"]))
        except Exception as exc:
            record["phash_error"] = repr(exc)
            phash = None
        record["phash"] = phash
        if phash and phash_groups.get(phash):
            record["duplicate_reason"] = "exact_phash_duplicate"
            record["duplicate_of"] = phash_groups[phash][0]["candidate_id"]
            duplicates.append(record)
            counters["exact_phash_duplicate"] += 1
            phash_groups[phash].append(record)
            continue
        if phash:
            phash_groups[phash].append(record)
        kept.append(record)

    kept.sort(key=lambda r: (r["year"], r["month"], r["paper_id"], r["figure_index"], r["candidate_id"]))
    write_jsonl(args.out, kept)
    write_jsonl(args.duplicates, duplicates)
    report = {
        "input_accepted": len(records),
        "kept": len(kept),
        "duplicates": len(duplicates),
        "duplicate_reasons": dict(counters.most_common()),
        "by_year": dict(Counter(str(r["year"]) for r in kept).most_common()),
        "by_kind": dict(Counter(r["image_kind"] for r in kept).most_common()),
        "charxiv_kept": sum(1 for r in kept if r.get("is_charxiv_paper")),
    }
    write_json(args.report, report)
    print(f"wrote {args.out}: {len(kept)} kept, {len(duplicates)} duplicates", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
