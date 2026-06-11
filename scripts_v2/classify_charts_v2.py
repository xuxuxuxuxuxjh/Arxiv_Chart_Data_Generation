#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from common_v2 import (
    CHART_TYPES,
    EDIT2_ROOT,
    GEMINI_MODEL,
    WORK_ROOT,
    append_jsonl,
    extract_json_object,
    gemini_generate,
    image_part_gemini,
    iter_jsonl,
    write_json,
)


CLASSIFIER_PROMPT = """You are filtering arXiv figures for a chart-reasoning dataset.

Use the image first. Caption text is auxiliary only.

Keep real charts/plots suitable for visual question answering and dense captioning.
Reject model architecture diagrams, flowcharts, pipelines, algorithm diagrams, formula screenshots, table screenshots, code/UI screenshots, natural/medical/remote-sensing/microscopy images, and qualitative examples unless the visual itself is a chart, heatmap, matrix plot, or confusion matrix.

You must choose chart_types only from this closed set:
{chart_types}

Layout is not a chart type. Use is_multi_panel, panel_count, and panel_layout for layout.

Return strict JSON only:
{{
  "is_real_chart": true,
  "chart_confidence": 0.93,
  "chart_types": ["line_chart"],
  "primary_chart_type": "line_chart",
  "non_chart_reason": null,
  "is_diagram": false,
  "is_table_screenshot": false,
  "is_photo_or_qualitative_image": false,
  "is_multi_panel": true,
  "panel_count": 4,
  "panel_layout": "2x2",
  "text_readability": "good",
  "has_axes_or_scale": true,
  "has_legend_or_series_labels": true,
  "has_numeric_values": true,
  "suitable_for_vqa": true,
  "suitable_for_dense_caption": true,
  "risk_notes": []
}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify chart candidates with v2 closed chart type enum.")
    parser.add_argument("--input", type=Path, default=WORK_ROOT / "candidates_2020_2025.filtered.jsonl")
    parser.add_argument("--out", type=Path, default=EDIT2_ROOT / "candidates_2020_2025.chart_classified_v2.jsonl")
    parser.add_argument("--failures", type=Path, default=EDIT2_ROOT / "logs" / "classifier_failed.jsonl")
    parser.add_argument("--report", type=Path, default=EDIT2_ROOT / "reports" / "classifier_report.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--image-max-pixels", type=int, default=350000)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {record["candidate_id"] for record in iter_jsonl(path)}


def normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    chart_types = result.get("chart_types") or []
    if isinstance(chart_types, str):
        chart_types = [chart_types]
    chart_types = [str(item) for item in chart_types if str(item) in CHART_TYPES]
    if not chart_types and result.get("is_real_chart"):
        chart_types = ["other_chart"]
    primary = str(result.get("primary_chart_type") or (chart_types[0] if chart_types else "other_chart"))
    if primary not in CHART_TYPES:
        primary = chart_types[0] if chart_types else "other_chart"
    result["chart_types"] = chart_types
    result["primary_chart_type"] = primary
    return result


def classify_one(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run:
        score = float(record.get("metadata_chart_score") or 0)
        result = {
            "is_real_chart": score > 0,
            "chart_confidence": 0.8 if score > 0 else 0.4,
            "chart_types": ["other_chart"] if score > 0 else [],
            "primary_chart_type": "other_chart" if score > 0 else None,
            "non_chart_reason": None if score > 0 else "dry_run_metadata_score_not_positive",
            "is_diagram": False,
            "is_table_screenshot": False,
            "is_photo_or_qualitative_image": False,
            "is_multi_panel": record.get("image_kind") == "merged",
            "panel_count": record.get("merged_panel_count"),
            "panel_layout": str(record.get("merge_layout") or ""),
            "text_readability": "unknown",
            "has_axes_or_scale": score > 0,
            "has_legend_or_series_labels": False,
            "has_numeric_values": False,
            "suitable_for_vqa": score > 0,
            "suitable_for_dense_caption": score > 0,
            "risk_notes": ["dry_run_result"],
        }
    else:
        prompt = CLASSIFIER_PROMPT.format(chart_types=", ".join(CHART_TYPES))
        raw = gemini_generate(
            [
                image_part_gemini(
                    Path(record["image_path"]),
                    cache_dir=EDIT2_ROOT / "tmp" / "gemini_classifier_images",
                    max_pixels=args.image_max_pixels,
                ),
                {"text": prompt + "\n\nCaption LaTeX:\n" + (record.get("caption_latex") or "")[:5000]},
            ],
            max_output_tokens=4096,
            temperature=0,
            top_p=1,
            reasoning_effort="low",
            timeout=180,
            retries=2,
        )
        result = normalize_result(extract_json_object(raw))

    accepted = (
        bool(result.get("is_real_chart"))
        and float(result.get("chart_confidence") or 0) >= 0.75
        and bool(result.get("suitable_for_vqa"))
        and bool(result.get("suitable_for_dense_caption"))
        and not bool(result.get("is_diagram"))
        and not bool(result.get("is_table_screenshot"))
        and not bool(result.get("is_photo_or_qualitative_image"))
    )
    weak_accept = bool(result.get("is_real_chart")) and 0.60 <= float(result.get("chart_confidence") or 0) < 0.75
    out = dict(record)
    out["classifier"] = {
        "model": GEMINI_MODEL,
        "protocol": "gemini_native_generateContent",
        "model_config": {
            "maxOutputTokens": 4096,
            "temperature": 0,
            "topP": 1,
            "extra_kwargs": {"reasoning_effort": "low"},
            "image_max_pixels": args.image_max_pixels,
        },
        **result,
        "accepted": accepted,
        "weak_accept": weak_accept,
        "chart_type_enum": list(CHART_TYPES),
    }
    return out


def main() -> int:
    args = parse_args()
    done = already_done(args.out)
    records = [record for record in iter_jsonl(args.input) if record["candidate_id"] not in done]
    if args.shuffle:
        random.seed(20260611)
        random.shuffle(records)
    if args.limit:
        records = records[: args.limit]
    print(f"classifying {len(records)} records, already done {len(done)}", flush=True)

    success = 0
    failures = 0
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        batch_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(classify_one, record, args): record for record in batch}
            for future in as_completed(futures):
                record = futures[future]
                try:
                    out = future.result()
                except Exception as exc:
                    failures += 1
                    append_jsonl(args.failures, [{"candidate_id": record.get("candidate_id"), "error": repr(exc)}])
                    continue
                append_jsonl(args.out, [out])
                success += 1
        print(
            f"classified success={success} failures={failures} batch_elapsed={time.perf_counter() - batch_start:.1f}s",
            flush=True,
        )

    all_records = list(iter_jsonl(args.out)) if args.out.exists() else []
    accepted_count = sum(1 for r in all_records if (r.get("classifier") or {}).get("accepted"))
    report = {
        "classified_count": len(all_records),
        "new_success": success,
        "new_failures": failures,
        "accepted": accepted_count,
        "weak_accept": sum(1 for r in all_records if (r.get("classifier") or {}).get("weak_accept")),
        "chart_types": dict(Counter(t for r in all_records for t in ((r.get("classifier") or {}).get("chart_types") or [])).most_common()),
        "model": GEMINI_MODEL,
        "dry_run": args.dry_run,
    }
    write_json(args.report, report)
    print(f"wrote {args.out}; report {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
