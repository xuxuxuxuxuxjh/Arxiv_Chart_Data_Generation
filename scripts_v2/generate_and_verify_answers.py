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


ANSWER_VERIFY_SCHEMA_VERSION = "triple_consistency_v1"


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


CONSISTENCY_JUDGE_PROMPT = """You are a strict answer consistency judge.

Compare the extracted final answers from three independent Gemini answer generations.
Do not use the chart image. Do not decide whether the answer is visually correct.
Only decide whether all extracted final answers are semantically the same answer to the same question.

Input:
Question: {question}
Task type: {task_type}
Answer type: {answer_type}
Extracted final answers:
{answers}

Rules:
- For approximate numeric answers, allow small rounding or visual-reading differences only when scale and unit match.
- For trend, comparison, ranking, boolean, and choice answers, require the same conclusion.
- If any answer says the question is not answerable from the image alone, all three must say that to be consistent.
- Do not invent a corrected answer. Choose the canonical answer only from the three extracted answers.

Return strict JSON only:
{{
  "verdict": "consistent",
  "all_answers_consistent": true,
  "canonical_answer": "...",
  "canonical_answer_index": 1,
  "normalized_answers": ["...", "...", "..."],
  "normalized_answer": "...",
  "reason": "..."
}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate three Gemini answers and verify answer consistency.")
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
    parser.add_argument("--answer-samples", type=int, default=3, help="Independent Gemini answer generations per question.")
    parser.add_argument("--answer-retries", type=int, default=1)
    parser.add_argument("--judge-retries", type=int, default=1)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry records that only exist in raw/judge-failure outputs; verified records are still skipped.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def done_ids(path: Path, *, require_judge: bool) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for record in iter_jsonl(path):
        generation = record.get("answer_generation") or {}
        judge = record.get("answer_judge") or {}
        has_generation = generation.get("schema_version") == ANSWER_VERIFY_SCHEMA_VERSION
        has_judge = judge.get("schema_version") == ANSWER_VERIFY_SCHEMA_VERSION
        if has_generation and (has_judge or not require_judge):
            ids.add(record["id"])
    return ids


def answer_once(record: dict[str, Any], args: argparse.Namespace, sample_index: int, attempt: int) -> dict[str, Any]:
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
    try:
        extracted = extract_thinking_and_final_answer(raw)
    except Exception as exc:
        raise ValueError(f"answer_extraction_failed sample_index={sample_index} attempt={attempt}: {exc}") from exc
    final_answer = extracted["final_answer"]
    return {
        "sample_index": sample_index,
        "attempt": attempt,
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
        "final_answer": final_answer,
        "normalized_answer": normalize_answer(final_answer, record.get("answer_type", "short_phrase")),
        "extract_format": extracted["format"],
        "dry_run": args.dry_run,
    }


def generate_answer_samples(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.answer_samples != 3:
        raise ValueError("answer consistency verification requires --answer-samples 3")
    samples: list[dict[str, Any]] = []
    for sample_index in range(1, args.answer_samples + 1):
        last_error: Exception | None = None
        for attempt in range(args.answer_retries + 1):
            try:
                sample = answer_once(record, args, sample_index, attempt + 1)
            except Exception as exc:
                last_error = exc
                if attempt >= args.answer_retries:
                    break
                time.sleep(1 + attempt)
                continue
            samples.append(sample)
            break
        else:
            last_error = RuntimeError("unreachable answer retry state")
        if len(samples) < sample_index:
            assert last_error is not None
            raise RuntimeError(f"answer_sample_failed sample_index={sample_index}: {last_error!r}") from last_error

    first = samples[0]
    out = dict(record)
    out["answer_generation"] = {
        "schema_version": ANSWER_VERIFY_SCHEMA_VERSION,
        "model": GEMINI_MODEL,
        "protocol": "gemini_native_generateContent",
        "sample_count": args.answer_samples,
        "samples": samples,
        "final_answers": [sample["final_answer"] for sample in samples],
        "normalized_answers": [sample["normalized_answer"] for sample in samples],
        "raw_response": first["raw_response"],
        "thinking": first["thinking"],
        "final_answer": first["final_answer"],
        "extract_format": first["extract_format"],
        "model_config": {
            "maxOutputTokens": 8192,
            "temperature": 0.7,
            "topP": 0.95,
            "extra_kwargs": {"reasoning_effort": "high"},
            "image_max_pixels": args.image_max_pixels,
        },
        "dry_run": args.dry_run,
    }
    out["answer"] = first["final_answer"]
    out["answer_normalized"] = first["normalized_answer"]
    return out


def format_answers_for_judge(samples: list[dict[str, Any]]) -> str:
    lines = []
    for sample in samples:
        lines.append(f"{sample['sample_index']}. {sample['final_answer']}")
    return "\n".join(lines)


def judge_once(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    answer_generation = record.get("answer_generation") or {}
    samples = answer_generation.get("samples") or []
    if len(samples) != 3:
        raise ValueError(f"expected 3 extracted answers, got {len(samples)}")
    if args.dry_run:
        result = {
            "verdict": "consistent",
            "all_answers_consistent": True,
            "canonical_answer": samples[0]["final_answer"],
            "canonical_answer_index": 1,
            "normalized_answers": [sample["normalized_answer"] for sample in samples],
            "normalized_answer": samples[0]["normalized_answer"],
            "reason": "dry_run",
        }
    else:
        raw = gemini_generate(
            [
                {
                    "text": CONSISTENCY_JUDGE_PROMPT.format(
                        question=record["question"],
                        task_type=record.get("task_type"),
                        answer_type=record.get("answer_type"),
                        answers=format_answers_for_judge(samples),
                    )
                },
            ],
            max_output_tokens=2048,
            temperature=0,
            top_p=1,
            reasoning_effort="medium",
            timeout=180,
            retries=1,
        )
        result = extract_json_object(raw)
        result["raw_response"] = raw
    verdict = str(result.get("verdict") or "").lower()
    canonical_answer = str(result.get("canonical_answer") or "").strip()
    passed = verdict == "consistent" and bool(result.get("all_answers_consistent")) and bool(canonical_answer)
    out = dict(record)
    answer_generation = dict(answer_generation)
    if canonical_answer:
        answer_generation["final_answer"] = canonical_answer
        answer_generation["normalized_answer"] = normalize_answer(canonical_answer, record.get("answer_type", "short_phrase"))
        answer_generation["consistency_verified"] = passed
        canonical_index = result.get("canonical_answer_index")
        if canonical_index is not None:
            answer_generation["canonical_answer_index"] = canonical_index
            try:
                idx = int(canonical_index) - 1
                if 0 <= idx < len(samples):
                    answer_generation["thinking"] = samples[idx].get("thinking", "")
                    answer_generation["raw_response"] = samples[idx].get("raw_response", "")
            except (TypeError, ValueError):
                pass
        out["answer"] = canonical_answer
        out["answer_normalized"] = answer_generation["normalized_answer"]
    else:
        answer_generation["consistency_verified"] = False
    out["answer_generation"] = answer_generation
    out["answer_judge"] = {
        "model": GEMINI_MODEL,
        "schema_version": ANSWER_VERIFY_SCHEMA_VERSION,
        "protocol": "gemini_native_generateContent",
        "model_config": {
            "maxOutputTokens": 2048,
            "temperature": 0,
            "topP": 1,
            "extra_kwargs": {"reasoning_effort": "medium"},
            "image_input": False,
        },
        **result,
        "passed": passed,
        "dry_run": args.dry_run,
    }
    out["answer_verified"] = passed
    return out


def generate_and_judge(record: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any] | None]:
    answered = generate_answer_samples(record, args)
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
        answered["answer_judge_error"] = f"answer_consistency_failed:{judged.get('answer_judge')}"
        return answered, None
    if last_judge_error:
        answered["answer_judge_error"] = repr(last_judge_error)
        return answered, None
    raise RuntimeError("answer consistency judge failed without error")


def main() -> int:
    args = parse_args()
    done = done_ids(args.verified_out, require_judge=True)
    if not args.retry_failed:
        done |= done_ids(args.raw_out, require_judge=False)
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
                    error_text = repr(exc).lower()
                    if (
                        "answer_extraction_failed" in error_text
                        or "final answer" in error_text
                        or "empty model response" in error_text
                    ):
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
