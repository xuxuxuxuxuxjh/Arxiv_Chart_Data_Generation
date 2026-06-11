#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import random
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


TASK_SEQUENCE = (
    ["descriptive_extraction"] * 20
    + ["text_in_chart"] * 15
    + ["number_in_chart"] * 15
    + ["visual_comparison"] * 20
    + ["trend_reasoning"] * 15
    + ["counting"] * 10
)

QUESTION_PROMPT = """Generate exactly one image-only question for this chart.

The question must be answerable from the visible chart image alone. Do not require paper background, the full caption, hidden data, or external knowledge.

Target task type: {task_type}

Allowed task types:
- descriptive_extraction: title, axis label, legend text, colorbar label, subplot label.
- text_in_chart: visible method/category/panel/series label.
- number_in_chart: visible or approximate number only when readable.
- visual_comparison: highest, lowest, larger, smaller, better, worse.
- trend_reasoning: increasing, decreasing, stable, crossing, convergence.
- counting: number of visible bars, curves, panels, clusters, or reliably countable elements.

First version rules:
- evidence_source must be "image_only".
- requires_caption_context must be false.
- question_risk should be "low" only if the answer is visually reliable.
- Do not generate unanswerable or uncertain questions.
- Do not include the answer.

Return strict JSON only:
{{
  "question": "...",
  "task_type": "{task_type}",
  "answer_type": "short_text",
  "evidence_source": "image_only",
  "difficulty": "medium",
  "requires_exact_reading": false,
  "requires_caption_context": false,
  "question_risk": "low"
}}

Caption LaTeX is auxiliary context for terminology only; the question itself must be answerable without reading this caption outside the image.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate chart questions with Gemini.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--failures",
        type=Path,
        default=WORK_ROOT / "logs" / "question_generation_failures.jsonl",
    )
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {r["source"]["candidate_id"] for r in iter_jsonl(path)}


def make_question(record: dict[str, Any], group: str, idx: int, task_type: str, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        question = {
            "question": "What is the main chart type visible in this figure?",
            "task_type": "descriptive_extraction",
            "answer_type": "short_text",
            "evidence_source": "image_only",
            "difficulty": "easy",
            "requires_exact_reading": False,
            "requires_caption_context": False,
            "question_risk": "low",
        }
    else:
        parts = [
            image_part(Path(record["image_path"])),
            {
                "text": QUESTION_PROMPT.format(task_type=task_type)
                + "\n\nChart classifier output:\n"
                + str(record.get("classifier", {}))
                + "\n\nCaption LaTeX:\n"
                + (record.get("caption_latex") or "")[:5000],
            },
        ]
        raw = gemini_generate(
            parts,
            max_output_tokens=64000,
            temperature=1,
            top_p=0.95,
            reasoning_effort="high",
            timeout=240,
            retries=2,
        )
        question = extract_json_object(raw)
    question["task_type"] = question.get("task_type") or task_type
    question["evidence_source"] = "image_only"
    question["requires_caption_context"] = False
    return {
        "id": f"arxiv_chart_question_{group}_{idx:06d}",
        "group": group,
        "image": record["image_path"],
        "source": {
            "candidate_id": record["candidate_id"],
            "paper_id": record["paper_id"],
            "year": record["year"],
            "month": record["month"],
            "figure_index": record["figure_index"],
            "image_kind": record["image_kind"],
            "is_charxiv_paper": record.get("is_charxiv_paper", False),
            "json_path": record["json_path"],
        },
        **question,
        "question_generation": {
            "model": GEMINI_MODEL,
            "protocol": "gemini_native_generateContent",
            "model_config": {
                "maxOutputTokens": 64000,
                "temperature": 1,
                "topP": 0.95,
                "extra_kwargs": {"reasoning_effort": "high"},
            },
            "dry_run": dry_run,
        },
    }


def main() -> int:
    args = parse_args()
    done = already_done(args.out)
    records = [r for r in iter_jsonl(args.input) if r["candidate_id"] not in done]
    records = records[: args.limit] if args.limit else records
    tasks = itertools.cycle(TASK_SEQUENCE)
    jobs = [(record, next(tasks), idx) for idx, record in enumerate(records, 1)]
    print(f"generating {len(jobs)} questions for {args.group}", flush=True)
    success = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(make_question, record, args.group, idx, task, args.dry_run): (record, task)
            for record, task, idx in jobs
        }
        for future in as_completed(futures):
            record, task = futures[future]
            try:
                out = future.result()
            except Exception as exc:
                failures += 1
                append_jsonl(
                    args.failures,
                    [
                        {
                            "candidate_id": record["candidate_id"],
                            "group": args.group,
                            "task_type": task,
                            "error": repr(exc),
                        }
                    ],
                )
                continue
            append_jsonl(args.out, [out])
            success += 1
            if success % 50 == 0:
                print(f"questions success={success} failures={failures}", flush=True)
    write_json(
        args.out.with_suffix(".report.json"),
        {"success": success, "failures": failures, "group": args.group, "dry_run": args.dry_run},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
