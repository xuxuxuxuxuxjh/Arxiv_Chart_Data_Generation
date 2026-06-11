#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pipeline_common import WORK_ROOT, count_jsonl, iter_jsonl, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate quality stats and HTML review page.")
    parser.add_argument("--work", type=Path, default=WORK_ROOT)
    parser.add_argument(
        "--quality-out", type=Path, default=WORK_ROOT / "reports" / "quality_stats.json"
    )
    parser.add_argument(
        "--review-out", type=Path, default=WORK_ROOT / "review" / "pilot_review.html"
    )
    parser.add_argument("--limit", type=int, default=200)
    return parser.parse_args()


def assert_no_image_bytes(paths: list[Path]) -> list[str]:
    bad: list[str] = []
    patterns = [re.compile(r"base64", re.I), re.compile(r"data:image/", re.I)]
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for idx, line in enumerate(f, 1):
                if any(p.search(line) for p in patterns):
                    bad.append(f"{path}:{idx}")
                    break
    return bad


def load_by_source(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    if not path.exists():
        return result
    for record in iter_jsonl(path):
        source = record.get("source", {})
        cid = source.get("candidate_id")
        if cid:
            result[cid] = record
    return result


def review_html(work: Path, out: Path, limit: int) -> None:
    inclusive_sample = work / "sample_charxiv_inclusive_50k.jsonl"
    questions = load_by_source(work / "qa" / "charxiv_inclusive_50k.questions.jsonl")
    answers = load_by_source(work / "qa" / "charxiv_inclusive_50k.consensus.jsonl")
    thinking = load_by_source(work / "qa" / "charxiv_inclusive_50k.qa_thinking.jsonl")
    captions = load_by_source(work / "dense_caption" / "charxiv_inclusive_50k.dense_caption.jsonl")
    rows = []
    if inclusive_sample.exists():
        for sample in iter_jsonl(inclusive_sample):
            if len(rows) >= limit:
                break
            cid = sample["candidate_id"]
            q = questions.get(cid, {})
            a = answers.get(cid, {})
            t = thinking.get(cid, {})
            c = captions.get(cid, {})
            thinking_response = (t.get("thinking_response", {}) or {}).get("response", "")
            thinking_failed = t.get("thinking_response_failed")
            if not (q and a and t and c):
                continue
            if not thinking_response or not c.get("dense_caption"):
                continue
            rows.append(
                "<section>"
                f"<h2>{html.escape(cid)}</h2>"
                f"<img src=\"file://{html.escape(sample['image_path'])}\" loading=\"lazy\">"
                f"<p><b>Caption LaTeX:</b> {html.escape((sample.get('caption_latex') or '')[:1000])}</p>"
                f"<p><b>Question:</b> {html.escape(q.get('question', ''))}</p>"
                f"<p><b>Gemini answers:</b> {html.escape(json.dumps(a.get('answer_generation', {}).get('runs', []), ensure_ascii=False))}</p>"
                f"<p><b>Kimi thinking failed:</b> {html.escape(str(thinking_failed))}</p>"
                f"<p><b>Kimi thinking:</b> {html.escape(thinking_response)}</p>"
                f"<p><b>Dense caption:</b> {html.escape(c.get('dense_caption', ''))}</p>"
                "</section>"
            )
    body = "\n".join(rows)
    doc = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>arXiv Chart Pilot Review</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }}
    section {{ border-top: 1px solid #ccd3dd; padding: 20px 0; }}
    img {{ max-width: 960px; max-height: 680px; display: block; background: #fff; border: 1px solid #ccd3dd; }}
    p {{ max-width: 1100px; line-height: 1.45; }}
  </style>
</head>
<body>
  <h1>arXiv Chart Pilot Review</h1>
  {body}
</body>
</html>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")


def main() -> int:
    args = parse_args()
    work = args.work
    jsonl_paths = [
        work / "sample_charxiv_inclusive_50k.jsonl",
        work / "sample_charxiv_exclusive_50k.jsonl",
        work / "qa" / "charxiv_inclusive_50k.qa_direct.jsonl",
        work / "qa" / "charxiv_inclusive_50k.qa_thinking.jsonl",
        work / "qa" / "charxiv_exclusive_50k.qa_direct.jsonl",
        work / "qa" / "charxiv_exclusive_50k.qa_thinking.jsonl",
        work / "dense_caption" / "charxiv_inclusive_50k.dense_caption.jsonl",
        work / "dense_caption" / "charxiv_exclusive_50k.dense_caption.jsonl",
    ]
    stats = {
        "jsonl_counts": {str(path): count_jsonl(path) for path in jsonl_paths},
        "no_image_bytes_check_failures": assert_no_image_bytes(jsonl_paths),
    }
    classified = work / "candidates_2020_2025.chart_classified.jsonl"
    if classified.exists():
        total = accepted = weak = 0
        for record in iter_jsonl(classified):
            total += 1
            accepted += int(bool(record.get("classifier", {}).get("accepted")))
            weak += int(bool(record.get("classifier", {}).get("weak_accept")))
        stats["classifier"] = {
            "classified": total,
            "accepted": accepted,
            "weak_accept": weak,
            "accepted_rate": accepted / total if total else 0,
        }
    for name in ("inclusive", "exclusive"):
        sample = work / f"sample_charxiv_{name}_50k.jsonl"
        if sample.exists():
            stats[f"{name}_sample"] = {
                "count": count_jsonl(sample),
                "is_charxiv_paper": dict(Counter(str(r.get("is_charxiv_paper")) for r in iter_jsonl(sample)).most_common()),
            }
    write_json(args.quality_out, stats)
    review_html(work, args.review_out, args.limit)
    print(f"wrote {args.quality_out}", flush=True)
    print(f"wrote {args.review_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
