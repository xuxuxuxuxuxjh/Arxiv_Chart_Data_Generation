#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from pipeline_common import WORK_ROOT, iter_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select a balanced classifier pilot set.")
    parser.add_argument(
        "--input", type=Path, default=WORK_ROOT / "candidates_2020_2025.filtered.jsonl"
    )
    parser.add_argument(
        "--out", type=Path, default=WORK_ROOT / "classifier_pilot_input.jsonl"
    )
    parser.add_argument("--target", type=int, default=2000)
    parser.add_argument("--min-score", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    per_year_target = {year: args.target // 6 for year in range(2020, 2026)}
    per_year_target[2025] += args.target - sum(per_year_target.values())
    selected: list[dict] = []
    by_year = Counter()
    by_month = Counter()
    by_kind = Counter()
    seen_papers = Counter()
    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            year = int(record.get("year"))
            month = str(record.get("month"))
            if by_year[year] >= per_year_target[year]:
                continue
            if int(record.get("metadata_chart_score", 0)) < args.min_score:
                continue
            if seen_papers[record["paper_id"]] >= 2:
                continue
            if by_month[month] >= max(1, per_year_target[year] // 12 + 10):
                continue
            selected.append(record)
            by_year[year] += 1
            by_month[month] += 1
            by_kind[record.get("image_kind")] += 1
            seen_papers[record["paper_id"]] += 1
            if len(selected) >= args.target:
                break
    if len(selected) < args.target:
        with args.input.open("r", encoding="utf-8") as f:
            selected_ids = {r["candidate_id"] for r in selected}
            for line in f:
                record = json.loads(line)
                if record["candidate_id"] in selected_ids:
                    continue
                if int(record.get("metadata_chart_score", 0)) < 1:
                    continue
                selected.append(record)
                if len(selected) >= args.target:
                    break
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for record in selected:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")
    print(f"wrote {args.out}: {len(selected)} records", flush=True)
    print("by_year", dict(by_year), flush=True)
    print("by_kind", dict(by_kind), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
