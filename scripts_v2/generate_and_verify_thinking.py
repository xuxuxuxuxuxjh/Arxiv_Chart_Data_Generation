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
    KIMI_MESSAGES_MODEL,
    append_jsonl,
    extract_final_answer,
    extract_json_object,
    gemini_generate,
    image_part_gemini,
    iter_jsonl,
    kimi_messages_generate,
    normalize_answer,
    write_json,
)


THINKING_PROMPT = """Reason from visible chart evidence only.

Use the verified answer as the required final answer. Do not use paper background, caption-only claims, or external knowledge. The reasoning should explicitly mention the visible chart evidence that supports the answer.

Question: {question}
Task type: {task_type}
Answer type: {answer_type}
Verified answer: {answer}

End with exactly:
Final answer: {answer}
"""


THINKING_JUDGE_PROMPT = """You are checking whether a chart reasoning response is valid.

Use only the visible chart image.

Question: {question}
Task type: {task_type}
Answer type: {answer_type}
Verified answer: {answer}
Kimi final answer: {kimi_final_answer}
Kimi reasoning response:
{kimi_response}

Check:
1. Kimi final answer matches the verified answer.
2. Kimi reasoning is grounded in visible chart evidence.
3. Kimi reasoning does not contradict the image or hallucinate paper context.

Return strict JSON only:
{{
  "verdict": "pass",
  "final_answer_matches": true,
  "reasoning_grounded": true,
  "has_contradiction": false,
  "reason": "..."
}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Kimi thinking and verify it with Gemini.")
    parser.add_argument("--input", type=Path, default=EDIT2_ROOT / "answers_verified.jsonl")
    parser.add_argument("--raw-out", type=Path, default=EDIT2_ROOT / "kimi_thinking_raw.jsonl")
    parser.add_argument("--verified-out", type=Path, default=EDIT2_ROOT / "kimi_thinking_verified.jsonl")
    parser.add_argument("--failures", type=Path, default=EDIT2_ROOT / "logs" / "kimi_thinking_failures.jsonl")
    parser.add_argument("--judge-failures", type=Path, default=EDIT2_ROOT / "logs" / "kimi_thinking_judge_failures.jsonl")
    parser.add_argument("--report", type=Path, default=EDIT2_ROOT / "reports" / "kimi_thinking_verified.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-max-pixels", type=int, default=100000)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--judge-image-max-pixels", type=int, default=350000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {record["id"] for record in iter_jsonl(path)}


def generate_thinking(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    prompt = THINKING_PROMPT.format(
        question=record["question"],
        task_type=record.get("task_type"),
        answer_type=record.get("answer_type"),
        answer=record["answer"],
    )
    if args.dry_run:
        response = f"The visible chart evidence supports the verified answer.\nFinal answer: {record['answer']}"
    else:
        response = kimi_messages_generate(
            image_path=Path(record["image"]),
            text=prompt,
            cache_dir=EDIT2_ROOT / "tmp" / "kimi_thinking_images",
            image_max_pixels=args.image_max_pixels,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
        )
    final_answer = extract_final_answer(response)
    answer_type = record.get("answer_type", "short_phrase")
    final_matches = normalize_answer(final_answer, answer_type) == normalize_answer(record["answer"], answer_type)
    out = dict(record)
    out["kimi_thinking"] = {
        "model": KIMI_MESSAGES_MODEL,
        "protocol": "anthropic_messages",
        "model_config": {
            "max_tokens": args.max_tokens,
            "image_max_pixels": args.image_max_pixels,
            "timeout": args.timeout,
            "retries": args.retries,
        },
        "response": response,
        "final_answer": final_answer,
        "final_answer_matches_verified_answer": final_matches,
        "dry_run": args.dry_run,
    }
    out["messages"] = {
        "qa_direct": [
            {"role": "user", "content": "<image>\n" + record["question"]},
            {"role": "assistant", "content": record["answer"]},
        ],
        "qa_thinking": [
            {"role": "user", "content": "<image>\n" + record["question"]},
            {"role": "assistant", "content": response},
        ],
    }
    return out


def judge_thinking(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    thinking = record.get("kimi_thinking") or {}
    if args.dry_run:
        result = {
            "verdict": "pass",
            "final_answer_matches": True,
            "reasoning_grounded": True,
            "has_contradiction": False,
            "reason": "dry_run",
        }
    else:
        raw = gemini_generate(
            [
                image_part_gemini(
                    Path(record["image"]),
                    cache_dir=EDIT2_ROOT / "tmp" / "gemini_thinking_judge_images",
                    max_pixels=args.judge_image_max_pixels,
                ),
                {
                    "text": THINKING_JUDGE_PROMPT.format(
                        question=record["question"],
                        task_type=record.get("task_type"),
                        answer_type=record.get("answer_type"),
                        answer=record["answer"],
                        kimi_final_answer=thinking.get("final_answer", ""),
                        kimi_response=thinking.get("response", ""),
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
    passed = (
        str(result.get("verdict") or "").lower() == "pass"
        and bool(result.get("final_answer_matches"))
        and bool(result.get("reasoning_grounded"))
        and not bool(result.get("has_contradiction"))
        and bool(thinking.get("final_answer_matches_verified_answer"))
    )
    out = dict(record)
    out["kimi_thinking_judge"] = {
        "model": GEMINI_MODEL,
        "protocol": "gemini_native_generateContent",
        "model_config": {
            "maxOutputTokens": 4096,
            "temperature": 0,
            "topP": 1,
            "extra_kwargs": {"reasoning_effort": "medium"},
            "image_max_pixels": args.judge_image_max_pixels,
        },
        **result,
        "passed": passed,
        "dry_run": args.dry_run,
    }
    out["thinking_verified"] = passed
    return out


def generate_and_judge(record: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any] | None]:
    last: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            raw = generate_thinking(record, args)
            judged = judge_thinking(raw, args)
            if judged.get("thinking_verified"):
                return raw, judged
            raw["thinking_judge_error"] = f"thinking_judge_failed:{judged.get('kimi_thinking_judge')}"
            last = RuntimeError(raw["thinking_judge_error"])
        except Exception as exc:
            last = exc
            if attempt >= args.retries:
                raise
        if attempt < args.retries:
            time.sleep(1 + attempt)
    assert last is not None
    raise last


def main() -> int:
    args = parse_args()
    done = done_ids(args.raw_out) | done_ids(args.verified_out)
    records = [record for record in iter_jsonl(args.input) if record["id"] not in done]
    if args.limit:
        records = records[: args.limit]
    print(f"generating kimi thinking records={len(records)}", flush=True)
    raw_count = verified_count = failures = 0
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        batch_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(generate_and_judge, record, args): record for record in batch}
            for future in as_completed(futures):
                record = futures[future]
                try:
                    raw, verified = future.result()
                except Exception as exc:
                    failures += 1
                    append_jsonl(args.failures, [{"id": record.get("id"), "candidate_id": record.get("candidate_id"), "error": repr(exc)}])
                    continue
                append_jsonl(args.raw_out, [raw])
                raw_count += 1
                if verified:
                    append_jsonl(args.verified_out, [verified])
                    verified_count += 1
                else:
                    append_jsonl(args.judge_failures, [raw])
        print(
            f"kimi raw={raw_count} verified={verified_count} failures={failures} "
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
            "dry_run": args.dry_run,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
