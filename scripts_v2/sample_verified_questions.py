#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from common_v2 import EDIT2_ROOT, TASK_SPECS, iter_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample verified QA/thinking records from the full question candidate pool.")
    parser.add_argument("--input", type=Path, default=EDIT2_ROOT / "kimi_thinking_verified.jsonl")
    parser.add_argument("--out", type=Path, default=EDIT2_ROOT / "qa_thinking_sampled.jsonl")
    parser.add_argument("--report", type=Path, default=EDIT2_ROOT / "reports" / "qa_thinking_sampled.json")
    parser.add_argument("--target", type=int, default=0, help="0 keeps all verified records.")
    parser.add_argument("--max-per-image", type=int, default=3, help="0 disables per-image cap.")
    parser.add_argument("--dedup-similar", action="store_true", help="Avoid near-duplicate questions for the same image.")
    parser.add_argument("--seed", type=int, default=20260611)
    return parser.parse_args()


def task_weight(task_type: str) -> int:
    return int((TASK_SPECS.get(task_type) or {}).get("target_sampling_weight") or 1)


def difficulty_score(record: dict[str, Any]) -> int:
    difficulty = str(record.get("difficulty") or "").lower()
    steps = int(record.get("reasoning_steps_required") or 1)
    if difficulty == "hard":
        return 3 + steps
    if difficulty == "medium":
        return 2 + steps
    return 1 + steps


def question_signature(question: str) -> set[str]:
    text = re.sub(r"[^a-z0-9\s]", " ", question.lower())
    stop = {
        "the",
        "and",
        "or",
        "for",
        "from",
        "with",
        "which",
        "what",
        "chart",
        "plot",
        "figure",
        "shown",
        "value",
        "values",
        "mean",
        "approximate",
    }
    return {token for token in text.split() if len(token) > 2 and token not in stop}


def too_similar(question: str, existing: list[str], threshold: float = 0.72) -> bool:
    sig = question_signature(question)
    if not sig:
        return True
    for old in existing:
        old_sig = question_signature(old)
        if not old_sig:
            continue
        overlap = len(sig & old_sig) / max(len(sig | old_sig), 1)
        if overlap >= threshold:
            return True
    return False


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)
    records = list(iter_jsonl(args.input))
    if not args.target or args.target >= len(records):
        selected = records
    else:
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            by_task[str(record.get("task_type"))].append(record)
        for values in by_task.values():
            rng.shuffle(values)
            values.sort(key=difficulty_score, reverse=True)

        target_by_task = {
            task: max(1, round(args.target * task_weight(task) / sum(task_weight(t) for t in TASK_SPECS)))
            for task in TASK_SPECS
        }
        while sum(target_by_task.values()) > args.target:
            task = max(target_by_task, key=target_by_task.get)
            target_by_task[task] -= 1
        while sum(target_by_task.values()) < args.target:
            task = max(TASK_SPECS, key=task_weight)
            target_by_task[task] += 1

        selected = []
        per_image = Counter()
        questions_by_image: dict[str, list[str]] = defaultdict(list)

        def try_add(record: dict[str, Any]) -> bool:
            cid = (record.get("source") or {}).get("candidate_id") or record.get("candidate_id")
            if args.max_per_image and per_image[cid] >= args.max_per_image:
                return False
            if args.dedup_similar and too_similar(str(record.get("question") or ""), questions_by_image[cid]):
                return False
            selected.append(record)
            per_image[cid] += 1
            questions_by_image[cid].append(str(record.get("question") or ""))
            return True

        for task, desired in target_by_task.items():
            added = 0
            for record in by_task.get(task, []):
                if added >= desired:
                    break
                if try_add(record):
                    added += 1

        if len(selected) < args.target:
            selected_ids = {record["id"] for record in selected}
            leftovers = [record for record in records if record["id"] not in selected_ids]
            rng.shuffle(leftovers)
            leftovers.sort(key=difficulty_score, reverse=True)
            for record in leftovers:
                if len(selected) >= args.target:
                    break
                try_add(record)

    write_jsonl(args.out, selected)
    report = {
        "input": str(args.input),
        "output": str(args.out),
        "input_count": len(records),
        "target": args.target,
        "selected": len(selected),
        "max_per_image": args.max_per_image,
        "dedup_similar": args.dedup_similar,
        "by_task": dict(Counter(str(r.get("task_type")) for r in selected).most_common()),
        "by_answer_type": dict(Counter(str(r.get("answer_type")) for r in selected).most_common()),
        "by_difficulty": dict(Counter(str(r.get("difficulty")) for r in selected).most_common()),
        "unique_images": len({(r.get("source") or {}).get("candidate_id") or r.get("candidate_id") for r in selected}),
    }
    write_json(args.report, report)
    print(f"wrote {args.out}: {len(selected)} records", flush=True)
    print(f"wrote {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
