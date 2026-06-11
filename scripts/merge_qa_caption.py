#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from pipeline_common import WORK_ROOT, iter_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge QA and dense-caption records by candidate_id.")
    parser.add_argument("--work", type=Path, default=WORK_ROOT)
    parser.add_argument("--group", choices=["inclusive", "exclusive", "both"], default="both")
    parser.add_argument("--out-dir", type=Path, default=WORK_ROOT / "merged")
    parser.add_argument("--qa-only", action="store_true", help="Only keep samples with consensus QA.")
    return parser.parse_args()


def by_candidate_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result = {}
    for record in iter_jsonl(path):
        cid = (record.get("source") or {}).get("candidate_id")
        if cid:
            result[cid] = record
    return result


def strip_messages(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {k: v for k, v in record.items() if k != "messages"}


def merge_group(work: Path, out_dir: Path, group: str, qa_only: bool) -> dict[str, Any]:
    group_name = f"charxiv_{group}_50k"
    sample_path = work / f"sample_charxiv_{group}_50k.jsonl"
    direct = by_candidate_id(work / "qa" / f"{group_name}.qa_direct.jsonl")
    thinking = by_candidate_id(work / "qa" / f"{group_name}.qa_thinking.jsonl")
    caption = by_candidate_id(work / "dense_caption" / f"{group_name}.dense_caption.jsonl")

    merged = []
    missing_qa = 0
    missing_caption = 0
    missing_thinking = 0
    failed_thinking = 0
    failed_caption = 0

    for sample in iter_jsonl(sample_path):
        cid = sample["candidate_id"]
        qa_direct = direct.get(cid)
        qa_thinking = thinking.get(cid)
        dense_caption = caption.get(cid)
        if qa_only and not qa_thinking:
            continue

        missing_qa += int(qa_direct is None)
        missing_thinking += int(qa_thinking is None)
        missing_caption += int(dense_caption is None)
        failed_thinking += int(bool((qa_thinking or {}).get("thinking_response_failed")))
        quality = (dense_caption or {}).get("quality") or {}
        failed_caption += int(bool(quality.get("generation_failed")))

        qa_source = qa_thinking or qa_direct or {}
        merged.append(
            {
                "id": f"{group_name}_{cid}",
                "group": group_name,
                "candidate_id": cid,
                "image": sample.get("image_path"),
                "source": {
                    "candidate_id": cid,
                    "paper_id": sample.get("paper_id"),
                    "year": sample.get("year"),
                    "month": sample.get("month"),
                    "figure_index": sample.get("figure_index"),
                    "image_kind": sample.get("image_kind"),
                    "is_charxiv_paper": sample.get("is_charxiv_paper", False),
                    "json_path": sample.get("json_path"),
                    "caption_latex": sample.get("caption_latex", ""),
                },
                "question": qa_source.get("question"),
                "answer": qa_source.get("answer"),
                "task_type": qa_source.get("task_type"),
                "difficulty": qa_source.get("difficulty"),
                "answer_type": qa_source.get("answer_type"),
                "evidence_source": qa_source.get("evidence_source"),
                "requires_exact_reading": qa_source.get("requires_exact_reading"),
                "requires_caption_context": qa_source.get("requires_caption_context"),
                "question_risk": qa_source.get("question_risk"),
                "answer_generation": qa_source.get("answer_generation"),
                "thinking_response": (qa_thinking or {}).get("thinking_response"),
                "thinking_response_failed": (qa_thinking or {}).get("thinking_response_failed"),
                "dense_caption": (dense_caption or {}).get("dense_caption"),
                "visible_elements": (dense_caption or {}).get("visible_elements"),
                "uncertainty": (dense_caption or {}).get("uncertainty"),
                "caption_quality": (dense_caption or {}).get("quality"),
                "messages": {
                    "qa_direct": (qa_direct or {}).get("messages"),
                    "qa_thinking": (qa_thinking or {}).get("messages"),
                    "dense_caption": (dense_caption or {}).get("messages"),
                },
                "raw_records": {
                    "qa_direct": strip_messages(qa_direct),
                    "qa_thinking": strip_messages(qa_thinking),
                    "dense_caption": strip_messages(dense_caption),
                },
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "qa_caption_qa_only" if qa_only else "qa_caption"
    out_path = out_dir / f"{group_name}.{suffix}.jsonl"
    report_path = out_dir / f"{group_name}.{suffix}.report.json"
    write_jsonl(out_path, merged)
    report = {
        "group": group_name,
        "output": str(out_path),
        "count": len(merged),
        "qa_only": qa_only,
        "missing_qa": missing_qa,
        "missing_thinking": missing_thinking,
        "missing_caption": missing_caption,
        "failed_thinking": failed_thinking,
        "failed_caption": failed_caption,
    }
    write_json(report_path, report)
    print(f"wrote {out_path}")
    print(f"wrote {report_path}")
    print(report)
    return report


def main() -> int:
    args = parse_args()
    groups = ["inclusive", "exclusive"] if args.group == "both" else [args.group]
    reports = [merge_group(args.work, args.out_dir, group, args.qa_only) for group in groups]
    write_json(args.out_dir / "merge_qa_caption.summary.json", {"reports": reports})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
