#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from pipeline_common import CHARXIV_ROOT, WORK_ROOT, read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract unique paper IDs from CharXiv metadata."
    )
    parser.add_argument("--src", type=Path, default=CHARXIV_ROOT)
    parser.add_argument(
        "--out", type=Path, default=WORK_ROOT / "charxiv_paper_ids.json"
    )
    return parser.parse_args()


def load_chart_types(src: Path, split: str) -> dict[int, list[str]]:
    path = src / f"chart_types_{split}.json"
    if not path.exists():
        return {}
    data = read_json(path)
    mapping: dict[int, list[str]] = {}
    records = data.values() if isinstance(data, dict) else data
    for record in records:
        if not isinstance(record, dict):
            continue
        figure_id = record.get("figure_id")
        if figure_id is None:
            continue
        chart_type = (
            record.get("chart_type")
            or record.get("chart_types")
            or record.get("answer")
            or record.get("label")
        )
        if isinstance(chart_type, list):
            values = [str(item) for item in chart_type]
        elif chart_type:
            values = [str(chart_type)]
        else:
            values = []
        mapping[int(figure_id)] = values
    return mapping


def main() -> int:
    args = parse_args()
    paper_ids: set[str] = set()
    records: list[dict] = []
    split_counts = Counter()
    category_counts = Counter()

    for split in ("val", "test"):
        metadata_path = args.src / f"image_metadata_{split}.json"
        chart_types = load_chart_types(args.src, split)
        data = read_json(metadata_path)
        items = data.values() if isinstance(data, dict) else data
        for item in items:
            if not isinstance(item, dict):
                continue
            paper_id = str(item.get("paper_id", "")).strip()
            if not paper_id:
                continue
            figure_id = item.get("figure_id")
            paper_ids.add(paper_id)
            split_counts[split] += 1
            category_counts[str(item.get("category", ""))] += 1
            records.append(
                {
                    "split": split,
                    "figure_id": figure_id,
                    "paper_id": paper_id,
                    "chart_types": chart_types.get(int(figure_id), [])
                    if figure_id is not None
                    else [],
                    "title": item.get("title", ""),
                    "caption": item.get("caption", ""),
                    "category": item.get("category", ""),
                    "year": item.get("year", ""),
                    "figure_path": item.get("figure_path", ""),
                }
            )

    output = {
        "paper_ids": sorted(paper_ids),
        "records": records,
        "stats": {
            "unique_paper_count": len(paper_ids),
            "record_count": len(records),
            "split_counts": dict(split_counts),
            "category_counts": dict(category_counts.most_common()),
        },
    }
    write_json(args.out, output)
    print(
        f"wrote {args.out}: {len(paper_ids)} unique papers, {len(records)} records",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
