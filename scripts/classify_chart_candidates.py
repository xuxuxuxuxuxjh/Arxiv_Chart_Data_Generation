#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pipeline_common import (
    GEMINI_MODEL,
    WORK_ROOT,
    append_jsonl,
    extract_json_object,
    gemini_generate,
    image_part,
    iter_jsonl,
    write_json,
)


CLASSIFIER_PROMPT = """You are filtering figures from arXiv papers for a chart-understanding dataset.

Decide whether the image is a real chart/plot that is suitable for visual question answering and dense captioning.

Keep charts such as line charts, bar charts, scatter plots, histograms, heatmaps, confusion matrices, box/violin plots, area charts, ROC/PR/calibration curves, ablation curves, and multi-panel figures where most panels are charts.

Reject model architecture diagrams, flowcharts, pipelines, framework diagrams, algorithm diagrams, pure formula screenshots, table screenshots, code/UI screenshots, natural/medical/remote-sensing/microscopy images, and qualitative result examples unless the visual itself is a chart/heatmap/confusion matrix.

Use the image first. The caption is auxiliary context only.

Return strict JSON only with this schema:
{
  "is_real_chart": true,
  "chart_confidence": 0.93,
  "chart_types": ["line_chart", "bar_chart"],
  "non_chart_reason": null,
  "is_diagram": false,
  "is_table_screenshot": false,
  "is_photo_or_qualitative_image": false,
  "is_multi_panel": true,
  "text_readability": "good",
  "has_axes_or_scale": true,
  "has_legend_or_series_labels": true,
  "has_numeric_values": true,
  "suitable_for_vqa": true,
  "suitable_for_dense_caption": true,
  "risk_notes": []
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify filtered candidates with Gemini.")
    parser.add_argument(
        "--input", type=Path, default=WORK_ROOT / "candidates_2020_2025.filtered.jsonl"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=WORK_ROOT / "candidates_2020_2025.chart_classified.jsonl",
    )
    parser.add_argument(
        "--failures",
        type=Path,
        default=WORK_ROOT / "logs" / "classifier_failed.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=WORK_ROOT / "reports" / "classifier_report.json",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {record["candidate_id"] for record in iter_jsonl(path)}


def classify_one(record: dict[str, Any], dry_run: bool = False, sleep: float = 0.0) -> dict[str, Any]:
    if sleep:
        time.sleep(sleep)
    if dry_run:
        score = record.get("metadata_chart_score", 0)
        result = {
            "is_real_chart": score > 0,
            "chart_confidence": 0.80 if score > 0 else 0.40,
            "chart_types": ["unknown_chart"] if score > 0 else [],
            "non_chart_reason": None if score > 0 else "dry_run_metadata_score_not_positive",
            "is_diagram": False,
            "is_table_screenshot": False,
            "is_photo_or_qualitative_image": False,
            "is_multi_panel": record.get("image_kind") == "merged",
            "text_readability": "unknown",
            "has_axes_or_scale": score > 0,
            "has_legend_or_series_labels": False,
            "has_numeric_values": False,
            "suitable_for_vqa": score > 0,
            "suitable_for_dense_caption": score > 0,
            "risk_notes": ["dry_run_result"],
        }
    else:
        parts = [
            image_part(Path(record["image_path"])),
            {
                "text": CLASSIFIER_PROMPT
                + "\n\nCaption LaTeX:\n"
                + (record.get("caption_latex") or "")[:5000],
            },
        ]
        raw = gemini_generate(
            parts,
            max_output_tokens=4096,
            temperature=0,
            top_p=1,
            reasoning_effort="low",
            timeout=180,
            retries=2,
        )
        try:
            result = extract_json_object(raw)
        except Exception as exc:
            raise RuntimeError(f"json_parse_failed raw={raw[:2000]!r}") from exc
    accepted = (
        bool(result.get("is_real_chart"))
        and float(result.get("chart_confidence") or 0) >= 0.75
        and bool(result.get("suitable_for_dense_caption"))
        and bool(result.get("suitable_for_vqa"))
        and not bool(result.get("is_diagram"))
        and not bool(result.get("is_table_screenshot"))
        and not bool(result.get("is_photo_or_qualitative_image"))
    )
    weak_accept = (
        bool(result.get("is_real_chart"))
        and 0.60 <= float(result.get("chart_confidence") or 0) < 0.75
    )
    out = dict(record)
    out["classifier"] = {
        "model": GEMINI_MODEL,
        "protocol": "gemini_native_generateContent",
        "model_config": {
            "maxOutputTokens": 4096,
            "temperature": 0,
            "topP": 1,
            "extra_kwargs": {"reasoning_effort": "low"},
        },
        **result,
        "accepted": accepted,
        "weak_accept": weak_accept,
    }
    return out


def main() -> int:
    args = parse_args()
    done = already_done(args.out)
    records = [r for r in iter_jsonl(args.input) if r["candidate_id"] not in done]
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
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(classify_one, record, args.dry_run, args.sleep): record
                for record in batch
            }
            for future in as_completed(futures):
                record = futures[future]
                try:
                    out = future.result()
                except Exception as exc:
                    failures += 1
                    append_jsonl(
                        args.failures,
                        [
                            {
                                "candidate_id": record.get("candidate_id"),
                                "image_path": record.get("image_path"),
                                "json_path": record.get("json_path"),
                                "error": repr(exc),
                            }
                        ],
                    )
                    continue
                append_jsonl(args.out, [out])
                success += 1
        print(f"classified {success}, failures {failures}", flush=True)

    all_records = list(iter_jsonl(args.out)) if args.out.exists() else []
    report = {
        "classified_count": len(all_records),
        "new_success": success,
        "new_failures": failures,
        "accepted": sum(1 for r in all_records if r.get("classifier", {}).get("accepted")),
        "weak_accept": sum(1 for r in all_records if r.get("classifier", {}).get("weak_accept")),
        "model": GEMINI_MODEL,
        "dry_run": args.dry_run,
    }
    write_json(args.report, report)
    print(f"wrote {args.out}; report {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
