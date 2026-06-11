#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from common_v2 import (
    ANSWER_TYPES,
    EDIT2_ROOT,
    GEMINI_MODEL,
    TASK_SPECS,
    TASK_TYPES,
    append_jsonl,
    enum_or_default,
    extract_json_object,
    gemini_generate,
    image_part_gemini,
    iter_jsonl,
    question_is_low_value,
    stable_id,
    write_json,
)


QUESTION_PROMPT = """Generate one challenging image-only chart reasoning question.

The question must be answerable from the visible chart image alone. Do not require paper background, hidden data, or external knowledge. Caption text is provided only for terminology.

Target task type: {task_type}
Task definition: {task_description}
Allowed answer types for this task: {answer_types}

Global rules:
- Prefer medium or hard questions.
- The question must require reasoning over visible chart content.
- Do not ask title-only, axis-label-only, legend-only, colorbar-label-only, or subplot-label-only questions.
- Text reading is allowed only when it is one step in a multi-step reasoning question.
- Avoid generic "what is shown/plotted" questions.
- Include enough visual anchors so the answer can be verified from the image.
- Do not include the answer.
- The current image may be a single extracted panel from a larger paper figure. Do not ask about panels, subfigures, datasets, or categories that are mentioned in the caption but are not visible in the current image.
- Only use phrases such as "across the panels", "four panels", "top-left", "bottom-right", or "(a)/(b)" if the current visible image itself contains those panels.

Return strict JSON only:
{{
  "question": "...",
  "task_type": "{task_type}",
  "answer_type": "numeric_approx",
  "difficulty": "hard",
  "requires_exact_reading": false,
  "requires_caption_context": false,
  "reasoning_steps_required": 2,
  "visual_elements_required": ["legend", "x_axis", "curve", "panel"],
  "risk_notes": []
}}
"""


GROUPED_QUESTION_PROMPT = """Generate one challenging image-only chart reasoning question for each requested task type.

The questions must be answerable from the visible chart image alone. Do not require paper background, hidden data, or external knowledge. Caption text is provided only for terminology.

Requested task types:
{task_specs}

Global rules:
- Return exactly one question for each requested task type.
- Prefer medium or hard questions.
- Every question must require reasoning over visible chart content.
- Do not ask title-only, axis-label-only, legend-only, colorbar-label-only, or subplot-label-only questions.
- Text reading is allowed only when it is one step in a multi-step reasoning question.
- Avoid generic "what is shown/plotted" questions.
- Include enough visual anchors so each answer can be verified from the image.
- Do not include any answers.
- The current image may be a single extracted panel from a larger paper figure. Do not ask about panels, subfigures, datasets, or categories that are mentioned in the caption but are not visible in the current image.
- Only use phrases such as "across the panels", "four panels", "top-left", "bottom-right", or "(a)/(b)" if the current visible image itself contains those panels.

Return strict JSON only:
{{
  "questions": [
    {{
      "question": "...",
      "task_type": "cross_element_comparison",
      "answer_type": "choice",
      "difficulty": "hard",
      "requires_exact_reading": false,
      "requires_caption_context": false,
      "reasoning_steps_required": 2,
      "visual_elements_required": ["legend", "x_axis", "curve", "panel"],
      "risk_notes": []
    }}
  ]
}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate question candidates for every task type on each chart."
    )
    parser.add_argument("--input", type=Path, default=EDIT2_ROOT / "filtered_charts_2020_2025.jsonl")
    parser.add_argument("--out", type=Path, default=EDIT2_ROOT / "question_candidates.jsonl")
    parser.add_argument("--failures", type=Path, default=EDIT2_ROOT / "logs" / "question_generation_failures.jsonl")
    parser.add_argument("--report", type=Path, default=EDIT2_ROOT / "reports" / "question_candidates.json")
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--task-types", nargs="*", default=list(TASK_TYPES))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-max-pixels", type=int, default=350000)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument(
        "--group-tasks-per-image",
        action="store_true",
        help="Use one model call per image to generate all missing task types.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def done_keys(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    keys = set()
    for record in iter_jsonl(path):
        cid = (record.get("source") or {}).get("candidate_id")
        task_type = record.get("task_type")
        if cid and task_type:
            keys.add((cid, task_type))
    return keys


def validate_question(result: dict[str, Any], task_type: str) -> dict[str, Any]:
    question = str(result.get("question") or "").strip()
    if not question:
        raise ValueError("empty question")
    if question_is_low_value(question):
        raise ValueError(f"low_value_question:{question}")
    result["task_type"] = enum_or_default(result.get("task_type"), TASK_TYPES, task_type)
    allowed_answers = TASK_SPECS[result["task_type"]]["answer_types"]
    result["answer_type"] = enum_or_default(result.get("answer_type"), allowed_answers, allowed_answers[0])
    if result["answer_type"] not in ANSWER_TYPES:
        result["answer_type"] = allowed_answers[0]
    result["difficulty"] = enum_or_default(result.get("difficulty"), ("medium", "hard"), "medium")
    result["requires_caption_context"] = False
    steps = int(result.get("reasoning_steps_required") or 1)
    result["reasoning_steps_required"] = max(1, steps)
    visual = result.get("visual_elements_required") or []
    if isinstance(visual, str):
        visual = [visual]
    result["visual_elements_required"] = [str(item) for item in visual][:12]
    risk_notes = result.get("risk_notes") or []
    if isinstance(risk_notes, str):
        risk_notes = [risk_notes]
    result["risk_notes"] = [str(item) for item in risk_notes][:12]
    return result


def question_conflicts_with_visible_layout(result: dict[str, Any], record: dict[str, Any]) -> bool:
    question = str(result.get("question") or "")
    classifier = record.get("classifier") or {}
    is_multi_panel = bool(classifier.get("is_multi_panel"))
    panel_count = int(classifier.get("panel_count") or (2 if is_multi_panel else 1))
    multi_panel_terms = re.search(
        r"\b(four|three|two|multiple)\s+panels\b|\bacross\s+(all\s+)?(the\s+)?panels\b|"
        r"\btop[- ]left\b|\btop[- ]right\b|\bbottom[- ]left\b|\bbottom[- ]right\b|"
        r"\bpanel\s+\(?[a-z]\)?\b",
        question,
        re.I,
    )
    return bool(multi_panel_terms) and (not is_multi_panel or panel_count <= 1)


def task_specs_text(task_types: list[str]) -> str:
    lines = []
    for task_type in task_types:
        spec = TASK_SPECS[task_type]
        lines.append(
            f"- {task_type}: {spec['description']} Allowed answer types: {', '.join(spec['answer_types'])}."
        )
    return "\n".join(lines)


def build_question_record(record: dict[str, Any], result: dict[str, Any], task_type: str, args: argparse.Namespace) -> dict[str, Any]:
    cid = record["candidate_id"]
    result = validate_question(result, task_type)
    if question_conflicts_with_visible_layout(result, record):
        raise ValueError(f"layout_conflict_question:{result.get('question')}")
    return {
        "id": stable_id("qv2", f"{cid}:{task_type}:{result['question']}"),
        "candidate_id": cid,
        "image": record["image_path"],
        "source": {
            "candidate_id": cid,
            "paper_id": record.get("paper_id"),
            "year": record.get("year"),
            "month": record.get("month"),
            "figure_index": record.get("figure_index"),
            "image_kind": record.get("image_kind"),
            "is_charxiv_paper": record.get("is_charxiv_paper", False),
            "json_path": record.get("json_path"),
            "caption_latex": record.get("caption_latex", ""),
            "classifier": record.get("classifier") or {},
        },
        **result,
        "question_generation": {
            "model": GEMINI_MODEL,
            "protocol": "gemini_native_generateContent",
            "model_config": {
                "maxOutputTokens": 8192,
                "temperature": 0.9,
                "topP": 0.95,
                "extra_kwargs": {"reasoning_effort": "high"},
                "image_max_pixels": args.image_max_pixels,
                "group_tasks_per_image": args.group_tasks_per_image,
            },
            "dry_run": args.dry_run,
        },
    }


def generate_one(record: dict[str, Any], task_type: str, args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run:
        result = {
            "question": f"Which visible element in the chart best supports a {task_type} judgment?",
            "task_type": task_type,
            "answer_type": TASK_SPECS[task_type]["answer_types"][0],
            "difficulty": "medium",
            "requires_exact_reading": False,
            "requires_caption_context": False,
            "reasoning_steps_required": 2,
            "visual_elements_required": ["chart"],
            "risk_notes": ["dry_run"],
        }
    else:
        prompt = QUESTION_PROMPT.format(
            task_type=task_type,
            task_description=TASK_SPECS[task_type]["description"],
            answer_types=", ".join(TASK_SPECS[task_type]["answer_types"]),
        )
        last_exc: Exception | None = None
        for attempt in range(args.retries + 1):
            try:
                raw = gemini_generate(
                    [
                        image_part_gemini(
                            Path(record["image_path"]),
                            cache_dir=EDIT2_ROOT / "tmp" / "gemini_question_images",
                            max_pixels=args.image_max_pixels,
                        ),
                        {
                            "text": prompt
                            + "\n\nChart classifier output:\n"
                            + str(record.get("classifier") or {})
                            + "\n\nCaption LaTeX:\n"
                            + (record.get("caption_latex") or "")[:5000],
                        },
                    ],
                    max_output_tokens=8192,
                    temperature=0.9,
                    top_p=0.95,
                    reasoning_effort="high",
                    timeout=240,
                    retries=1,
                )
                result = validate_question(extract_json_object(raw), task_type)
                if question_conflicts_with_visible_layout(result, record):
                    raise ValueError(f"layout_conflict_question:{result.get('question')}")
                break
            except Exception as exc:
                last_exc = exc
                if attempt >= args.retries:
                    raise
                time.sleep(1.0 + attempt)
        else:
            assert last_exc is not None
            raise last_exc

    return build_question_record(record, result, task_type, args)


def extract_grouped_questions(raw: str, task_types: list[str]) -> list[dict[str, Any]]:
    result = extract_json_object(raw)
    questions = result.get("questions")
    if isinstance(questions, dict):
        questions = list(questions.values())
    if not isinstance(questions, list):
        questions = [value for value in result.values() if isinstance(value, dict)]
    if not isinstance(questions, list):
        raise ValueError("grouped question response missing questions list")
    requested = set(task_types)
    return [
        item
        for item in questions
        if isinstance(item, dict) and str(item.get("task_type") or "") in requested
    ]


def generate_group(record: dict[str, Any], task_types: list[str], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.dry_run:
        return [generate_one(record, task_type, args) for task_type in task_types]

    prompt = GROUPED_QUESTION_PROMPT.format(task_specs=task_specs_text(task_types))
    last_exc: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            raw = gemini_generate(
                [
                    image_part_gemini(
                        Path(record["image_path"]),
                        cache_dir=EDIT2_ROOT / "tmp" / "gemini_question_images",
                        max_pixels=args.image_max_pixels,
                    ),
                    {
                        "text": prompt
                        + "\n\nChart classifier output:\n"
                        + str(record.get("classifier") or {})
                        + "\n\nCaption LaTeX:\n"
                        + (record.get("caption_latex") or "")[:5000],
                    },
                ],
                max_output_tokens=16384,
                temperature=0.9,
                top_p=0.95,
                reasoning_effort="high",
                timeout=240,
                retries=1,
            )
            raw_items = extract_grouped_questions(raw, task_types)
            by_task: dict[str, dict[str, Any]] = {}
            for item in raw_items:
                task_type = str(item.get("task_type") or "")
                if task_type not in by_task:
                    by_task[task_type] = item
            records = []
            for task_type in task_types:
                if task_type not in by_task:
                    continue
                try:
                    records.append(build_question_record(record, by_task[task_type], task_type, args))
                except Exception:
                    continue
            if not records:
                raise ValueError("no valid grouped questions")
            return records
        except Exception as exc:
            last_exc = exc
            if attempt >= args.retries:
                raise
            time.sleep(1.0 + attempt)
    assert last_exc is not None
    raise last_exc


def main() -> int:
    args = parse_args()
    invalid_tasks = [task for task in args.task_types if task not in TASK_SPECS]
    if invalid_tasks:
        raise ValueError(f"invalid task types: {invalid_tasks}")
    done = done_keys(args.out)
    records = list(iter_jsonl(args.input))
    if args.limit_images:
        records = records[: args.limit_images]

    if args.group_tasks_per_image:
        jobs = [
            (record, [task_type for task_type in args.task_types if (record["candidate_id"], task_type) not in done])
            for record in records
        ]
        jobs = [(record, tasks) for record, tasks in jobs if tasks]
        expected = sum(len(tasks) for _, tasks in jobs)
    else:
        jobs = [
            (record, task_type)
            for record in records
            for task_type in args.task_types
            if (record["candidate_id"], task_type) not in done
        ]
        expected = len(jobs)
    print(
        f"generating question candidates jobs={len(jobs)} expected_questions={expected} "
        f"images={len(records)} tasks={len(args.task_types)} grouped={args.group_tasks_per_image}",
        flush=True,
    )
    success = 0
    failures = 0
    by_task: dict[str, int] = {task: 0 for task in args.task_types}
    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start : start + args.batch_size]
        batch_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            if args.group_tasks_per_image:
                futures = {
                    executor.submit(generate_group, record, tasks, args): (record, tasks)
                    for record, tasks in batch
                }
            else:
                futures = {
                    executor.submit(generate_one, record, task, args): (record, task)
                    for record, task in batch
                }
            for future in as_completed(futures):
                record, task_or_tasks = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    if args.group_tasks_per_image:
                        failed_tasks = list(task_or_tasks)
                    else:
                        failed_tasks = [str(task_or_tasks)]
                    failures += len(failed_tasks)
                    append_jsonl(
                        args.failures,
                        [
                            {"candidate_id": record.get("candidate_id"), "task_type": task, "error": repr(exc)}
                            for task in failed_tasks
                        ],
                    )
                    continue
                outs = result if args.group_tasks_per_image else [result]
                expected_tasks = list(task_or_tasks) if args.group_tasks_per_image else [str(task_or_tasks)]
                got_tasks = {str(out.get("task_type")) for out in outs}
                append_jsonl(args.out, outs)
                for out in outs:
                    task = str(out.get("task_type"))
                    by_task[task] = by_task.get(task, 0) + 1
                    success += 1
                missing_tasks = [task for task in expected_tasks if task not in got_tasks]
                if missing_tasks:
                    failures += len(missing_tasks)
                    append_jsonl(
                        args.failures,
                        [
                            {
                                "candidate_id": record.get("candidate_id"),
                                "task_type": task,
                                "error": "missing_from_grouped_response",
                            }
                            for task in missing_tasks
                        ],
                    )
        print(
            f"questions success={success} failures={failures} done_jobs={min(start + len(batch), len(jobs))}/{len(jobs)} "
            f"elapsed={time.perf_counter() - batch_start:.1f}s",
            flush=True,
        )

    write_json(
        args.report,
        {
            "input": str(args.input),
            "output": str(args.out),
            "images": len(records),
            "task_types": list(args.task_types),
            "new_success": success,
            "new_failures": failures,
            "by_task": by_task,
            "group_tasks_per_image": args.group_tasks_per_image,
            "dry_run": args.dry_run,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
