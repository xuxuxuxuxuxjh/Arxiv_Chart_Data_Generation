#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pipeline_common import (
    GEMINI_MODEL,
    WORK_ROOT,
    append_jsonl,
    compatible_answers,
    gemini_generate,
    image_part,
    iter_jsonl,
    normalize_answer,
    write_json,
)


ANSWER_PROMPT = """Look only at the chart image.
Answer the question with the shortest correct answer.
Do not explain.
If the exact value is unreadable, answer with an approximate value only when the question allows approximation.
If the question cannot be answered from the image alone, answer "Not answerable from the image alone".

Question: {question}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Answer generated questions with Gemini 3 times.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--failures",
        type=Path,
        default=WORK_ROOT / "logs" / "answer_consensus_failures.jsonl",
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {r["id"] for r in iter_jsonl(path)}


def extract_short_answer(text: str) -> str:
    stripped = text.strip()
    marker = "final answer:"
    lower = stripped.lower()
    if marker in lower:
        return stripped[lower.rfind(marker) + len(marker) :].strip()
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    lines = [
        line
        for line in lines
        if not line.startswith("**")
        and not line.startswith("#")
        and line not in {"```", "```text", "```json"}
    ]
    if lines:
        return lines[-1].strip().strip('"')
    return stripped


def answer_one(record: dict[str, Any], dry_run: bool) -> dict[str, Any]:
    answer_type = record.get("answer_type", "short_text")
    if dry_run:
        runs = ["chart", "chart", "chart"]
    else:
        runs = []
        for _ in range(3):
            raw = gemini_generate(
                [
                    image_part(Path(record["image"])),
                    {"text": ANSWER_PROMPT.format(question=record["question"])},
                ],
                max_output_tokens=64000,
                temperature=1,
                top_p=0.95,
                reasoning_effort="high",
                timeout=240,
                retries=2,
            )
            runs.append(extract_short_answer(raw))
    normalized = [normalize_answer(value, answer_type) for value in runs]
    consensus = len(set(normalized)) == 1 or compatible_answers(normalized, answer_type)
    if not consensus:
        raise ValueError(f"no_consensus runs={runs} normalized={normalized}")
    out = dict(record)
    out["answer"] = runs[0]
    out["answer_generation"] = {
        "model": GEMINI_MODEL,
        "protocol": "gemini_native_generateContent",
        "model_config": {
            "maxOutputTokens": 64000,
            "temperature": 1,
            "topP": 0.95,
            "extra_kwargs": {"reasoning_effort": "high"},
        },
        "runs": runs,
        "normalized_runs": normalized,
        "consensus": True,
        "dry_run": dry_run,
    }
    return out


def main() -> int:
    args = parse_args()
    done = already_done(args.out)
    records = [r for r in iter_jsonl(args.input) if r["id"] not in done]
    print(f"answering {len(records)} questions", flush=True)
    success = 0
    failures = 0
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(answer_one, record, args.dry_run): record for record in batch}
            for future in as_completed(futures):
                record = futures[future]
                try:
                    out = future.result()
                except Exception as exc:
                    failures += 1
                    append_jsonl(
                        args.failures,
                        [{"id": record["id"], "question": record.get("question"), "error": repr(exc)}],
                    )
                    continue
                append_jsonl(args.out, [out])
                success += 1
        print(f"answers success={success} failures={failures}", flush=True)
    total = success + failures
    write_json(
        args.out.with_suffix(".report.json"),
        {
            "success": success,
            "failures": failures,
            "consensus_rate": success / total if total else 0,
            "dry_run": args.dry_run,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
