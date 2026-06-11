#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from common_v2 import EDIT2_ROOT, count_jsonl, iter_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate edition2 quality report and review HTML.")
    parser.add_argument("--merged", type=Path, default=EDIT2_ROOT / "merged.jsonl")
    parser.add_argument("--quality-out", type=Path, default=EDIT2_ROOT / "reports" / "quality_stats.json")
    parser.add_argument("--review-out", type=Path, default=EDIT2_ROOT / "review" / "review.html")
    parser.add_argument("--limit", type=int, default=200)
    return parser.parse_args()


def assert_no_image_bytes(paths: list[Path]) -> list[str]:
    bad = []
    patterns = [re.compile(r"base64", re.I), re.compile(r"data:image/", re.I)]
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f, 1):
                if any(pattern.search(line) for pattern in patterns):
                    bad.append(f"{path}:{idx}")
                    break
    return bad


def build_review(records: list[dict[str, Any]], out: Path) -> None:
    rows = []
    for record in records:
        source = record.get("source") or {}
        thinking = ((record.get("kimi_thinking") or {}).get("response") or "")
        caption = record.get("dense_caption") or ""
        rows.append(
            "<section>"
            f"<h2>{html.escape(str(record.get('candidate_id')))}</h2>"
            f"<img src=\"file://{html.escape(str(record.get('image')))}\" loading=\"lazy\">"
            f"<p><b>Task:</b> {html.escape(str(record.get('task_type')))} / {html.escape(str(record.get('answer_type')))} / {html.escape(str(record.get('difficulty')))}</p>"
            f"<p><b>Question:</b> {html.escape(str(record.get('question') or ''))}</p>"
            f"<p><b>Answer:</b> {html.escape(str(record.get('answer') or ''))}</p>"
            f"<p><b>Answer judge:</b> {html.escape(json.dumps(record.get('answer_judge') or {}, ensure_ascii=False)[:1200])}</p>"
            f"<p><b>Kimi thinking:</b> {html.escape(thinking[:2500])}</p>"
            f"<p><b>Thinking judge:</b> {html.escape(json.dumps(record.get('kimi_thinking_judge') or {}, ensure_ascii=False)[:1200])}</p>"
            f"<p><b>Caption:</b> {html.escape(caption[:1200])}</p>"
            f"<p><b>Caption judge:</b> {html.escape(json.dumps(record.get('caption_judge') or {}, ensure_ascii=False)[:1200])}</p>"
            f"<p><b>Caption LaTeX:</b> {html.escape(str(source.get('caption_latex') or '')[:1000])}</p>"
            "</section>"
        )
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>arXiv Chart Edition2 Review</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    section {{ border-top: 1px solid #ccd3dd; padding: 20px 0; }}
    img {{ max-width: 960px; max-height: 680px; display: block; background: #fff; border: 1px solid #ccd3dd; }}
    p {{ max-width: 1100px; line-height: 1.45; }}
  </style>
</head>
<body>
  <h1>arXiv Chart Edition2 Review</h1>
  {"".join(rows)}
</body>
</html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")


def main() -> int:
    args = parse_args()
    records = list(iter_jsonl(args.merged)) if args.merged.exists() else []
    stats = {
        "merged": str(args.merged),
        "count": len(records),
        "jsonl_counts": {
            str(args.merged): count_jsonl(args.merged),
            str(EDIT2_ROOT / "question_candidates.jsonl"): count_jsonl(EDIT2_ROOT / "question_candidates.jsonl"),
            str(EDIT2_ROOT / "answers_verified.jsonl"): count_jsonl(EDIT2_ROOT / "answers_verified.jsonl"),
            str(EDIT2_ROOT / "kimi_thinking_verified.jsonl"): count_jsonl(EDIT2_ROOT / "kimi_thinking_verified.jsonl"),
            str(EDIT2_ROOT / "dense_caption_verified.jsonl"): count_jsonl(EDIT2_ROOT / "dense_caption_verified.jsonl"),
        },
        "no_image_bytes_check_failures": assert_no_image_bytes(
            [
                args.merged,
                EDIT2_ROOT / "question_candidates.jsonl",
                EDIT2_ROOT / "answers_verified.jsonl",
                EDIT2_ROOT / "kimi_thinking_verified.jsonl",
                EDIT2_ROOT / "dense_caption_verified.jsonl",
            ]
        ),
        "by_task": dict(Counter(str(r.get("task_type")) for r in records).most_common()),
        "by_answer_type": dict(Counter(str(r.get("answer_type")) for r in records).most_common()),
        "by_difficulty": dict(Counter(str(r.get("difficulty")) for r in records).most_common()),
        "verified": dict(Counter(json.dumps(r.get("verified") or {}, sort_keys=True) for r in records).most_common()),
    }
    write_json(args.quality_out, stats)
    build_review(records[: args.limit], args.review_out)
    print(f"wrote {args.quality_out}", flush=True)
    print(f"wrote {args.review_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
