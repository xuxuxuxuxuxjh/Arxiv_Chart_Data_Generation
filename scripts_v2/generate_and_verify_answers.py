#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from common_v2 import (
    EDIT2_ROOT,
    GEMINI_MODEL,
    append_jsonl,
    extract_json_object,
    extract_thinking_and_final_answer,
    gemini_generate,
    image_part_gemini,
    iter_jsonl,
    normalize_answer,
    write_json,
)


ANSWER_PROMPT = """Answer this chart question using only visible chart evidence.

You must first reason, then give a concise final answer.

Rules:
- Do not use paper background, caption-only claims, or external knowledge.
- If approximate reading is needed, state a reasonable approximation.
- If the question is not answerable from the image alone, final answer must be: Not answerable from the image alone
- Keep reasoning grounded in visible chart elements.

Question: {question}
Task type: {task_type}
Answer type: {answer_type}

Return exactly this format:
<think>
Your concise visual reasoning.
</think>
Final answer: ...
"""


JUDGE_PROMPT = """You are a strict chart QA verifier.

Judge whether the candidate answer is correct using only the visible chart image.

Input:
Question: {question}
Task type: {task_type}
Answer type: {answer_type}
Candidate reasoning:
{thinking}
Candidate final answer:
{final_answer}

Rules:
- If the question is not answerable from the image alone, mark incorrect unless the candidate final answer says so.
- For approximate numeric answers, allow reasonable visual reading tolerance.
- For trend, comparison, ranking, and hypothetical questions, judge semantic correctness rather than exact wording.
- Do not reward hallucinated paper context.

Return strict JSON only:
{{
  "verdict": "correct",
  "is_answerable_from_image": true,
  "answer_matches_image": true,
  "normalized_answer": "...",
  "corrected_answer": null,
  "reason": "..."
}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Gemini answers and verify them with Gemini judger.")
    parser.add_argument("--input", type=Path, default=EDIT2_ROOT / "question_candidates.jsonl")
    parser.add_argument("--raw-out", type=Path, default=EDIT2_ROOT / "answers_raw.jsonl")
    parser.add_argument("--verified-out", type=Path, default=EDIT2_ROOT / "answers_verified.jsonl")
    parser.add_argument("--failures", type=Path, default=EDIT2_ROOT / "logs" / "answer_failures.jsonl")
    parser.add_argument(
        "--extraction-failures",
        type=Path,
        default=EDIT2_ROOT / "logs" / "answer_extraction_failures.jsonl",
    )
    parser.add_argument("--judge-failures", type=Path, default=EDIT2_ROOT / "logs" / "answer_judge_failures.jsonl")
    parser.add_argument("--report", type=Path, default=EDIT2_ROOT / "reports" / "answers_verified.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-max-pixels", type=int, default=350000)
    parser.add_argument("--answer-retries", type=int, default=1)
    parser.add_argument("--judge-retries", type=int, default=1)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry records that only exist in raw/judge-failure outputs; verified records are still skipped.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {record["id"] for record in iter_jsonl(path)}


def answer_once(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run:
        raw = "<think>The visible evidence supports the answer.</think>\nFinal answer: chart value"
    else:
        raw = gemini_generate(
            [
                image_part_gemini(
                    Path(record["image"]),
                    cache_dir=EDIT2_ROOT / "tmp" / "gemini_answer_images",
                    max_pixels=args.image_max_pixels,
                ),
                {
                    "text": ANSWER_PROMPT.format(
                        question=record["question"],
                        task_type=record.get("task_type"),
                        answer_type=record.get("answer_type"),
                    )
                },
            ],
            max_output_tokens=8192,
            temperature=0.7,
            top_p=0.95,
            reasoning_effort="high",
            timeout=240,
            retries=1,
        )
    extracted = extract_thinking_and_final_answer(raw)
    out = dict(record)
    out["answer_generation"] = {
        "model": GEMINI_MODEL,
        "protocol": "gemini_native_generateContent",
        "model_config": {
            "maxOutputTokens": 8192,
            "temperature": 0.7,
            "topP": 0.95,
            "extra_kwargs": {"reasoning_effort": "high"},
            "image_max_pixels": args.image_max_pixels,
        },
        "raw_response": raw,
        "thinking": extracted["thinking"],
        "final_answer": extracted["final_answer"],
        "extract_format": extracted["format"],
        "dry_run": args.dry_run,
    }
    out["answer"] = extracted["final_answer"]
    out["answer_normalized"] = normalize_answer(extracted["final_answer"], record.get("answer_type", "short_phrase"))
    return out


def judge_once(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    answer_generation = record.get("answer_generation") or {}
    if args.dry_run:
        result = {
            "verdict": "correct",
            "is_answerable_from_image": True,
            "answer_matches_image": True,
            "normalized_answer": record.get("answer_normalized") or record.get("answer"),
            "corrected_answer": None,
            "reason": "dry_run",
        }
    else:
        raw = gemini_generate(
            [
                image_part_gemini(
                    Path(record["image"]),
                    cache_dir=EDIT2_ROOT / "tmp" / "gemini_answer_judge_images",
                    max_pixels=args.image_max_pixels,
                ),
                {
                    "text": JUDGE_PROMPT.format(
                        question=record["question"],
                        task_type=record.get("task_type"),
                        answer_type=record.get("answer_type"),
                        thinking=answer_generation.get("thinking", ""),
                        final_answer=answer_generation.get("final_answer", record.get("answer", "")),
                    )
                },
            ],
            max_output_tokens=4096,
            temperature=0,
            top_p=1,
            reasoning_effort="medium",
            timeout=180,
            retries=1,
        )
        result = extract_json_object(raw)
        result["raw_response"] = raw
    verdict = str(result.get("verdict") or "").lower()
    passed = (
        verdict == "correct"
        and bool(result.get("is_answerable_from_image"))
        and bool(result.get("answer_matches_image"))
    )
    out = dict(record)
    out["answer_judge"] = {
        "model": GEMINI_MODEL,
        "protocol": "gemini_native_generateContent",
        "model_config": {
            "maxOutputTokens": 4096,
            "temperature": 0,
            "topP": 1,
            "extra_kwargs": {"reasoning_effort": "medium"},
            "image_max_pixels": args.image_max_pixels,
        },
        **result,
        "passed": passed,
        "dry_run": args.dry_run,
    }
    out["answer_verified"] = passed
    return out


def generate_and_judge(record: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any] | None]:
    last_answer_error: Exception | None = None
    for answer_attempt in range(args.answer_retries + 1):
        try:
            answered = answer_once(record, args)
        except Exception as exc:
            last_answer_error = exc
            if answer_attempt >= args.answer_retries:
                raise
            time.sleep(1 + answer_attempt)
            continue

        last_judge_error: Exception | None = None
        for judge_attempt in range(args.judge_retries + 1):
            try:
                judged = judge_once(answered, args)
            except Exception as exc:
                last_judge_error = exc
                if judge_attempt >= args.judge_retries:
                    break
                time.sleep(1 + judge_attempt)
                continue
            if judged.get("answer_verified"):
                return answered, judged
            last_judge_error = RuntimeError(f"answer_judge_failed:{judged.get('answer_judge')}")
            break
        if answer_attempt < args.answer_retries:
            time.sleep(1 + answer_attempt)
            continue
        if last_judge_error:
            answered["answer_judge_error"] = repr(last_judge_error)
            return answered, None
    assert last_answer_error is not None
    raise last_answer_error


def main() -> int:
    args = parse_args()
    done = done_ids(args.verified_out)
    if not args.retry_failed:
        done |= done_ids(args.raw_out)
    records = [record for record in iter_jsonl(args.input) if record["id"] not in done]
    if args.limit:
        records = records[: args.limit]
    print(f"answering candidates={len(records)}", flush=True)
    raw_count = verified_count = failures = 0
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        batch_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(generate_and_judge, record, args): record for record in batch}
            for future in as_completed(futures):
                record = futures[future]
                try:
                    raw_record, verified = future.result()
                except Exception as exc:
                    failures += 1
                    failure_record = {
                        "id": record.get("id"),
                        "candidate_id": record.get("candidate_id"),
                        "error": repr(exc),
                    }
                    append_jsonl(args.failures, [failure_record])
                    if "final answer" in repr(exc).lower() or "empty model response" in repr(exc).lower():
                        append_jsonl(args.extraction_failures, [failure_record])
                    continue
                append_jsonl(args.raw_out, [raw_record])
                raw_count += 1
                if verified:
                    append_jsonl(args.verified_out, [verified])
                    verified_count += 1
                else:
                    append_jsonl(args.judge_failures, [raw_record])
        print(
            f"answers raw={raw_count} verified={verified_count} failures={failures} "
            f"done={min(start + len(batch), len(records))}/{len(records)} elapsed={time.perf_counter() - batch_start:.1f}s",
            flush=True,
        )
    write_json(
        args.report,
        {
            "input": str(args.input),
            "raw_out": str(args.raw_out),
            "verified_out": str(args.verified_out),
            "new_raw": raw_count,
            "new_verified": verified_count,
            "new_failures": failures,
            "retry_failed": args.retry_failed,
            "dry_run": args.dry_run,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
