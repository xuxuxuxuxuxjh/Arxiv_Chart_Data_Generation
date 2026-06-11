#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pipeline_common import (
    KIMI_MODEL,
    WORK_ROOT,
    append_jsonl,
    extract_final_answer,
    image_part,
    iter_jsonl,
    kimi_generate,
    normalize_answer,
    write_json,
)


THINKING_PROMPT = """Reason from visible chart evidence only, then end with:
Final answer: {answer}

Do not use paper background or caption-only claims. The final answer must exactly be the given consensus answer.

Question: {question}
Consensus answer: {answer}
Task type: {task_type}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Kimi thinking responses.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--direct-out", type=Path, required=True)
    parser.add_argument("--thinking-out", type=Path, required=True)
    parser.add_argument(
        "--failures",
        type=Path,
        default=WORK_ROOT / "logs" / "kimi_thinking_failures.jsonl",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {r["id"] for r in iter_jsonl(path)}


def direct_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {r["id"] for r in iter_jsonl(path)}


def direct_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        **record,
        "messages": [
            {"role": "user", "content": "<image>\n" + record["question"]},
            {"role": "assistant", "content": record["answer"]},
        ],
    }


def content_parts(image: str, text: str) -> list[dict[str, Any]]:
    return [
        {
            "type": "image_url",
            "image_url": {
                "url": "data:"
                + image_part(Path(image))["inlineData"]["mimeType"]
                + ";base64,"
                + image_part(Path(image))["inlineData"]["data"]
            },
        },
        {"type": "text", "text": text},
    ]


def thinking_one(record: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    prompt = THINKING_PROMPT.format(
        question=record["question"],
        answer=record["answer"],
        task_type=record.get("task_type", ""),
    )
    if dry_run:
        response = f"<think>The visible chart evidence supports the answer.</think>\nFinal answer: {record['answer']}"
    else:
        response = kimi_generate(
            [{"role": "user", "content": content_parts(record["image"], prompt)}],
            max_tokens=250000,
            temperature=1,
            top_p=1,
            top_k=-1,
            timeout=300,
            retries=2,
        )
    answer_type = record.get("answer_type", "short_text")
    ok = normalize_answer(extract_final_answer(response), answer_type) == normalize_answer(record["answer"], answer_type)
    if not ok and not dry_run:
        response = kimi_generate(
            [{"role": "user", "content": content_parts(record["image"], prompt)}],
            max_tokens=250000,
            temperature=1,
            top_p=1,
            top_k=-1,
            timeout=300,
            retries=1,
        )
        ok = normalize_answer(extract_final_answer(response), answer_type) == normalize_answer(record["answer"], answer_type)
    out = dict(record)
    out["thinking_response"] = {
        "model": KIMI_MODEL,
        "model_config": {
            "max_tokens": 250000,
            "temperature": 1,
            "top_p": 1,
            "top_k": -1,
        },
        "response": response,
        "final_answer_matches_consensus": ok,
        "dry_run": dry_run,
    }
    out["thinking_response_failed"] = not ok
    out["messages"] = [
        {"role": "user", "content": "<image>\n" + record["question"]},
        {"role": "assistant", "content": response if ok else record["answer"]},
    ]
    return out


def fallback_thinking_record(record: dict[str, Any], error: Exception) -> dict[str, Any]:
    out = dict(record)
    out["thinking_response"] = {
        "model": KIMI_MODEL,
        "model_config": {
            "max_tokens": 250000,
            "temperature": 1,
            "top_p": 1,
            "top_k": -1,
        },
        "response": record["answer"],
        "final_answer_matches_consensus": True,
        "error": repr(error),
    }
    out["thinking_response_failed"] = True
    out["messages"] = [
        {"role": "user", "content": "<image>\n" + record["question"]},
        {"role": "assistant", "content": record["answer"]},
    ]
    return out


def main() -> int:
    args = parse_args()
    done = already_done(args.thinking_out)
    all_input_records = list(iter_jsonl(args.input))
    records = [r for r in all_input_records if r["id"] not in done]
    direct_existing = direct_done(args.direct_out)
    append_jsonl(args.direct_out, [direct_record(r) for r in all_input_records if r["id"] not in direct_existing])
    print(f"generating Kimi thinking for {len(records)} records", flush=True)
    success = 0
    failed_match = 0
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(thinking_one, record, args.dry_run): record for record in batch}
            for future in as_completed(futures):
                record = futures[future]
                try:
                    out = future.result()
                except Exception as exc:
                    failed_match += 1
                    append_jsonl(args.thinking_out, [fallback_thinking_record(record, exc)])
                    append_jsonl(
                        args.failures,
                        [{"id": record["id"], "question": record.get("question"), "error": repr(exc)}],
                    )
                    continue
                append_jsonl(args.thinking_out, [out])
                success += 1
                failed_match += int(out.get("thinking_response_failed", False))
        print(f"kimi success={success} failed_match={failed_match}", flush=True)
    write_json(
        args.thinking_out.with_suffix(".report.json"),
        {"success": success, "failed_match": failed_match, "dry_run": args.dry_run},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
