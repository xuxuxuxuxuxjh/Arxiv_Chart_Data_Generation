#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pipeline_common import WORK_ROOT, iter_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample inclusive/exclusive chart groups.")
    parser.add_argument(
        "--input", type=Path, default=WORK_ROOT / "candidates_2020_2025.deduped.jsonl"
    )
    parser.add_argument(
        "--charxiv", type=Path, default=WORK_ROOT / "charxiv_paper_ids.json"
    )
    parser.add_argument(
        "--inclusive-out",
        type=Path,
        default=WORK_ROOT / "sample_charxiv_inclusive_50k.jsonl",
    )
    parser.add_argument(
        "--exclusive-out",
        type=Path,
        default=WORK_ROOT / "sample_charxiv_exclusive_50k.jsonl",
    )
    parser.add_argument(
        "--report", type=Path, default=WORK_ROOT / "reports" / "sampling_report.md"
    )
    parser.add_argument("--target", type=int, default=50000)
    parser.add_argument("--max-images-per-paper", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260611)
    return parser.parse_args()


def load_charxiv(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return set(data.get("paper_ids", []))


def chart_bucket(record: dict[str, Any]) -> str:
    types = [str(t).lower() for t in record.get("classifier", {}).get("chart_types") or []]
    joined = " ".join(types)
    if "line" in joined or "curve" in joined:
        return "line_chart"
    if "bar" in joined:
        return "bar_chart"
    if "scatter" in joined:
        return "scatter_plot"
    if "heatmap" in joined or "confusion" in joined or "matrix" in joined:
        return "heatmap_confusion_matrix"
    if "histogram" in joined or "density" in joined or "distribution" in joined:
        return "histogram_density_distribution"
    if "box" in joined or "violin" in joined or "area" in joined:
        return "box_violin_area"
    if record.get("classifier", {}).get("is_multi_panel"):
        return "multi_panel_chart"
    return "other"


def annotate(records: list[dict[str, Any]]) -> None:
    for idx, record in enumerate(records):
        record["sample_meta"] = {
            "chart_type_bucket": chart_bucket(record),
            "sample_index": idx,
        }


def select_records(
    pool: list[dict[str, Any]],
    target: int,
    *,
    rng: random.Random,
    max_images_per_paper: int,
    must_include_charxiv_first: bool,
) -> list[dict[str, Any]]:
    by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in pool:
        by_year[int(record["year"])].append(record)
    for values in by_year.values():
        rng.shuffle(values)

    selected: list[dict[str, Any]] = []
    per_paper = Counter()
    per_month = Counter()
    year_targets = {year: target // 6 for year in range(2020, 2026)}
    year_targets[2025] += target - sum(year_targets.values())
    max_per_month = {
        year: math.ceil(year_targets[year] / 12 * 1.3) for year in year_targets
    }

    def try_add(record: dict[str, Any]) -> bool:
        if len(selected) >= target:
            return False
        paper_id = record["paper_id"]
        month = str(record["month"])
        year = int(record["year"])
        if per_paper[paper_id] >= max_images_per_paper:
            return False
        if per_month[month] >= max_per_month.get(year, target):
            return False
        selected.append(record)
        per_paper[paper_id] += 1
        per_month[month] += 1
        return True

    if must_include_charxiv_first:
        charxiv_records = [r for r in pool if r.get("is_charxiv_paper")]
        charxiv_records.sort(key=lambda r: (-float(r.get("classifier", {}).get("chart_confidence") or 0), r["candidate_id"]))
        for record in charxiv_records:
            try_add(record)

    for year, desired in year_targets.items():
        year_records = by_year.get(year, [])
        year_selected_before = sum(1 for r in selected if int(r["year"]) == year)
        for record in sorted(
            year_records,
            key=lambda r: (
                len([s for s in selected if chart_bucket(s) == chart_bucket(r)]),
                -float(r.get("classifier", {}).get("chart_confidence") or 0),
            ),
        ):
            if len(selected) >= target:
                break
            if sum(1 for r in selected if int(r["year"]) == year) >= desired:
                break
            try_add(record)

    if len(selected) < target:
        leftovers = [
            r for r in pool if r["candidate_id"] not in {s["candidate_id"] for s in selected}
        ]
        leftovers.sort(key=lambda r: -float(r.get("classifier", {}).get("chart_confidence") or 0))
        for record in leftovers:
            if len(selected) >= target:
                break
            try_add(record)
    if len(selected) < target:
        selected_ids = {s["candidate_id"] for s in selected}
        leftovers = [r for r in pool if r["candidate_id"] not in selected_ids]
        leftovers.sort(key=lambda r: -float(r.get("classifier", {}).get("chart_confidence") or 0))
        for record in leftovers:
            if len(selected) >= target:
                break
            paper_id = record["paper_id"]
            if per_paper[paper_id] >= max_images_per_paper:
                continue
            selected.append(record)
            per_paper[paper_id] += 1

    annotate(selected)
    return selected


def distribution(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "year": dict(Counter(str(r["year"]) for r in records).most_common()),
        "month": dict(Counter(str(r["month"]) for r in records).most_common()),
        "chart_type_bucket": dict(Counter(chart_bucket(r) for r in records).most_common()),
        "image_kind": dict(Counter(r["image_kind"] for r in records).most_common()),
        "multi_panel": dict(Counter(str(bool(r.get("classifier", {}).get("is_multi_panel"))) for r in records).most_common()),
        "is_charxiv_paper": dict(Counter(str(bool(r.get("is_charxiv_paper"))) for r in records).most_common()),
        "unique_papers": len({r["paper_id"] for r in records}),
    }


def write_report(path: Path, inclusive: list[dict], exclusive: list[dict], charxiv_papers: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    inclusive_charxiv_papers = {r["paper_id"] for r in inclusive if r.get("is_charxiv_paper")}
    exclusive_bad = [r for r in exclusive if r["paper_id"] in charxiv_papers]
    lines = [
        "# Sampling Report",
        "",
        "## Summary",
        "",
        f"- Inclusive count: {len(inclusive)}",
        f"- Exclusive count: {len(exclusive)}",
        f"- Inclusive CharXiv paper count: {len(inclusive_charxiv_papers)}",
        f"- Exclusive records with CharXiv paper_id: {len(exclusive_bad)}",
        "",
        "## Inclusive Distribution",
        "",
        "```json",
        json.dumps(distribution(inclusive), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Exclusive Distribution",
        "",
        "```json",
        json.dumps(distribution(exclusive), ensure_ascii=False, indent=2, sort_keys=True),
        "```",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    records = list(iter_jsonl(args.input))
    charxiv_papers = load_charxiv(args.charxiv)

    inclusive = select_records(
        list(records),
        args.target,
        rng=rng,
        max_images_per_paper=args.max_images_per_paper,
        must_include_charxiv_first=True,
    )
    exclusive_pool = [r for r in records if r["paper_id"] not in charxiv_papers]
    exclusive = select_records(
        exclusive_pool,
        args.target,
        rng=rng,
        max_images_per_paper=args.max_images_per_paper,
        must_include_charxiv_first=False,
    )
    write_jsonl(args.inclusive_out, inclusive)
    write_jsonl(args.exclusive_out, exclusive)
    write_report(args.report, inclusive, exclusive, charxiv_papers)
    print(f"wrote inclusive {len(inclusive)} -> {args.inclusive_out}", flush=True)
    print(f"wrote exclusive {len(exclusive)} -> {args.exclusive_out}", flush=True)
    print(f"wrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
