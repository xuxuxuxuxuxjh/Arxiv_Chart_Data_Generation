#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common_v2 import EDIT2_ROOT, iter_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge verified QA/thinking and verified dense captions.")
    parser.add_argument("--qa", type=Path, default=EDIT2_ROOT / "qa_thinking_sampled.jsonl")
    parser.add_argument("--caption", type=Path, default=EDIT2_ROOT / "dense_caption_verified.jsonl")
    parser.add_argument("--out", type=Path, default=EDIT2_ROOT / "merged.jsonl")
    parser.add_argument("--report", type=Path, default=EDIT2_ROOT / "reports" / "merged.json")
    parser.add_argument("--require-caption", action="store_true")
    return parser.parse_args()


def candidate_id(record: dict[str, Any]) -> str:
    return str(record.get("candidate_id") or (record.get("source") or {}).get("candidate_id"))


def by_candidate_id(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    if not path.exists():
        return records
    for record in iter_jsonl(path):
        cid = candidate_id(record)
        if cid:
            records[cid] = record
    return records


def strip_large(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    out = dict(record)
    return out


def main() -> int:
    args = parse_args()
    captions = by_candidate_id(args.caption)
    merged = []
    missing_caption = 0
    skipped_caption = 0

    for qa in iter_jsonl(args.qa):
        cid = candidate_id(qa)
        caption = captions.get(cid)
        missing_caption += int(caption is None)
        if args.require_caption and caption is None:
            skipped_caption += 1
            continue
        source = qa.get("source") or {}
        merged.append(
            {
                "id": f"edit2_{qa['id']}",
                "candidate_id": cid,
                "image": qa.get("image"),
                "source": source,
                "question": qa.get("question"),
                "answer": qa.get("answer"),
                "task_type": qa.get("task_type"),
                "answer_type": qa.get("answer_type"),
                "difficulty": qa.get("difficulty"),
                "requires_exact_reading": qa.get("requires_exact_reading"),
                "requires_caption_context": qa.get("requires_caption_context"),
                "reasoning_steps_required": qa.get("reasoning_steps_required"),
                "visual_elements_required": qa.get("visual_elements_required"),
                "answer_generation": qa.get("answer_generation"),
                "answer_judge": qa.get("answer_judge"),
                "kimi_thinking": qa.get("kimi_thinking"),
                "kimi_thinking_judge": qa.get("kimi_thinking_judge"),
                "dense_caption": (caption or {}).get("dense_caption"),
                "visible_elements": (caption or {}).get("visible_elements"),
                "uncertainty": (caption or {}).get("uncertainty"),
                "caption_generation": (caption or {}).get("caption_generation"),
                "caption_judge": (caption or {}).get("caption_judge"),
                "messages": {
                    "qa_direct": (qa.get("messages") or {}).get("qa_direct"),
                    "qa_thinking": (qa.get("messages") or {}).get("qa_thinking"),
                    "dense_caption": (caption or {}).get("messages"),
                },
                "raw_records": {
                    "qa_thinking": strip_large(qa),
                    "dense_caption": strip_large(caption),
                },
                "verified": {
                    "answer": bool(qa.get("answer_verified")),
                    "thinking": bool(qa.get("thinking_verified")),
                    "caption": bool((caption or {}).get("caption_verified")),
                },
            }
        )

    write_jsonl(args.out, merged)
    report = {
        "qa_input": str(args.qa),
        "caption_input": str(args.caption),
        "output": str(args.out),
        "count": len(merged),
        "missing_caption": missing_caption,
        "skipped_caption": skipped_caption,
        "require_caption": args.require_caption,
    }
    write_json(args.report, report)
    print(f"wrote {args.out}: {len(merged)} records", flush=True)
    print(f"wrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
