#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common_v2 import EDIT2_ROOT, WORK_ROOT, iter_jsonl, simple_phash, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare all edition2 chart inputs from edition1 filtered/classified manifests."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=WORK_ROOT / "candidates_2020_2025.chart_classified.jsonl",
        help="Classified candidate manifest.",
    )
    parser.add_argument("--out", type=Path, default=EDIT2_ROOT / "filtered_charts_2020_2025.jsonl")
    parser.add_argument("--duplicates", type=Path, default=EDIT2_ROOT / "logs" / "prepare_duplicates.jsonl")
    parser.add_argument("--report", type=Path, default=EDIT2_ROOT / "reports" / "prepare_charts.json")
    parser.add_argument("--include-weak", action="store_true", help="Also include weak_accept charts.")
    parser.add_argument("--skip-phash", action="store_true")
    return parser.parse_args()


def accepted(record: dict[str, Any], include_weak: bool) -> bool:
    classifier = record.get("classifier") or {}
    return bool(classifier.get("accepted")) or (include_weak and bool(classifier.get("weak_accept")))


def main() -> int:
    args = parse_args()
    records = [record for record in iter_jsonl(args.input) if accepted(record, args.include_weak)]
    records = [record for record in records if 2020 <= int(record.get("year") or 0) <= 2025]
    records.sort(
        key=lambda r: (
            -float((r.get("classifier") or {}).get("chart_confidence") or 0),
            r.get("candidate_id", ""),
        )
    )

    kept: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    seen_figure_keys: set[tuple[str, int, str]] = set()
    phash_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counters: Counter[str] = Counter()

    for record in records:
        key = (str(record.get("paper_id")), int(record.get("figure_index") or -1), str(record.get("image_kind")))
        if key in seen_figure_keys:
            duplicate = dict(record)
            duplicate["duplicate_reason"] = "paper_figure_kind_duplicate"
            duplicates.append(duplicate)
            counters["paper_figure_kind_duplicate"] += 1
            continue
        seen_figure_keys.add(key)

        phash = None
        if not args.skip_phash:
            try:
                phash = simple_phash(Path(record["image_path"]))
            except Exception as exc:
                record = dict(record)
                record["phash_error"] = repr(exc)
        record = dict(record)
        record["phash"] = phash
        if phash and phash_groups.get(phash):
            duplicate = dict(record)
            duplicate["duplicate_reason"] = "exact_phash_duplicate"
            duplicate["duplicate_of"] = phash_groups[phash][0].get("candidate_id")
            duplicates.append(duplicate)
            phash_groups[phash].append(record)
            counters["exact_phash_duplicate"] += 1
            continue
        if phash:
            phash_groups[phash].append(record)
        record["edition2_input_id"] = record.get("candidate_id")
        kept.append(record)

    kept.sort(key=lambda r: (int(r.get("year") or 0), str(r.get("month")), str(r.get("paper_id")), int(r.get("figure_index") or 0), str(r.get("candidate_id"))))
    write_jsonl(args.out, kept)
    write_jsonl(args.duplicates, duplicates)
    report = {
        "input": str(args.input),
        "output": str(args.out),
        "input_accepted": len(records),
        "kept": len(kept),
        "duplicates": len(duplicates),
        "include_weak": args.include_weak,
        "skip_phash": args.skip_phash,
        "duplicate_reasons": dict(counters.most_common()),
        "by_year": dict(Counter(str(r.get("year")) for r in kept).most_common()),
        "by_kind": dict(Counter(str(r.get("image_kind")) for r in kept).most_common()),
        "charxiv_paper": dict(Counter(str(bool(r.get("is_charxiv_paper"))) for r in kept).most_common()),
    }
    write_json(args.report, report)
    print(f"wrote {args.out}: {len(kept)} records", flush=True)
    print(f"wrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
