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
    extract_json_object,
    image_part,
    iter_jsonl,
    kimi_generate,
    write_json,
)


CAPTION_PROMPT = """Describe this chart in detail using only visible information.

Requirements:
- Use natural language grounded in the image.
- Include chart type, axes, legend/series, visual encodings, major trends/comparisons, and multi-panel layout when visible.
- Do not invent paper methods, dataset facts, or conclusions that are not visible in the image.
- If text is not legible, say that some labels are not legible.
- The dense_caption must be at least 2 sentences and at most 180 words.

Return strict JSON only:
{{
  "dense_caption": "...",
  "visible_elements": {{
    "chart_types": [],
    "axes": [],
    "series_or_panels": [],
    "main_trends": []
  }},
  "uncertainty": []
}}

Caption LaTeX for terminology only:
{caption_latex}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dense captions with Kimi.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--group", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--failures",
        type=Path,
        default=WORK_ROOT / "logs" / "dense_caption_verifier_failures.jsonl",
    )
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {r["source"]["candidate_id"] for r in iter_jsonl(path)}


def content_parts(image: str, text: str) -> list[dict[str, Any]]:
    inline = image_part(Path(image))["inlineData"]
    return [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{inline['mimeType']};base64,{inline['data']}"
            },
        },
        {"type": "text", "text": text},
    ]


def caption_one(record: dict[str, Any], group: str, idx: int, dry_run: bool) -> dict[str, Any]:
    if dry_run:
        result = {
            "dense_caption": "The image shows a chart with plotted data and visible labels. The main visual elements should be checked manually in the source image.",
            "visible_elements": {
                "chart_types": record.get("classifier", {}).get("chart_types") or [],
                "axes": [],
                "series_or_panels": [],
                "main_trends": [],
            },
            "uncertainty": ["dry_run_result"],
        }
    else:
        prompt = CAPTION_PROMPT.format(caption_latex=(record.get("caption_latex") or "")[:5000])
        raw = kimi_generate(
            [{"role": "user", "content": content_parts(record["image_path"], prompt)}],
            max_tokens=250000,
            temperature=1,
            top_p=1,
            top_k=-1,
            timeout=120,
            retries=1,
        )
        result = extract_json_object(raw)
    dense_caption = result.get("dense_caption", "")
    return {
        "id": f"arxiv_chart_caption_{group}_{idx:06d}",
        "group": group,
        "image": record["image_path"],
        "source": {
            "candidate_id": record["candidate_id"],
            "paper_id": record["paper_id"],
            "year": record["year"],
            "month": record["month"],
            "figure_index": record["figure_index"],
            "image_kind": record["image_kind"],
            "is_charxiv_paper": record.get("is_charxiv_paper", False),
            "json_path": record["json_path"],
            "caption_latex": record.get("caption_latex", ""),
        },
        "task_type": "dense_caption",
        "evidence_source": "image_only",
        "dense_caption": dense_caption,
        "visible_elements": result.get("visible_elements", {}),
        "uncertainty": result.get("uncertainty", []),
        "messages": [
            {"role": "user", "content": "<image>\nDescribe this chart in detail using only visible information."},
            {"role": "assistant", "content": dense_caption},
        ],
        "quality": {
            "caption_model": KIMI_MODEL,
            "verified": False,
            "verifier_model": None,
            "verifier_protocol": None,
            "dry_run": dry_run,
        },
    }


def fallback_caption(record: dict[str, Any], group: str, idx: int, error: Exception) -> dict[str, Any]:
    dense_caption = (
        "The image is a chart selected from an arXiv figure extraction pipeline. "
        "Automatic dense caption generation failed for this item, so this record should be regenerated or filtered before training."
    )
    return {
        "id": f"arxiv_chart_caption_{group}_{idx:06d}",
        "group": group,
        "image": record["image_path"],
        "source": {
            "candidate_id": record["candidate_id"],
            "paper_id": record["paper_id"],
            "year": record["year"],
            "month": record["month"],
            "figure_index": record["figure_index"],
            "image_kind": record["image_kind"],
            "is_charxiv_paper": record.get("is_charxiv_paper", False),
            "json_path": record["json_path"],
            "caption_latex": record.get("caption_latex", ""),
        },
        "task_type": "dense_caption",
        "evidence_source": "image_only",
        "dense_caption": dense_caption,
        "visible_elements": {
            "chart_types": record.get("classifier", {}).get("chart_types") or [],
            "axes": [],
            "series_or_panels": [],
            "main_trends": [],
        },
        "uncertainty": ["dense_caption_generation_failed", repr(error)],
        "messages": [
            {"role": "user", "content": "<image>\nDescribe this chart in detail using only visible information."},
            {"role": "assistant", "content": dense_caption},
        ],
        "quality": {
            "caption_model": KIMI_MODEL,
            "verified": False,
            "verifier_model": None,
            "verifier_protocol": None,
            "generation_failed": True,
        },
    }


def main() -> int:
    args = parse_args()
    done = already_done(args.out)
    records = [r for r in iter_jsonl(args.input) if r["candidate_id"] not in done]
    records = records[: args.limit] if args.limit else records
    print(f"generating dense captions for {len(records)} records", flush=True)
    success = 0
    failures = 0
    indexed = list(enumerate(records, 1))
    for start in range(0, len(indexed), args.batch_size):
        batch = indexed[start : start + args.batch_size]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(caption_one, record, args.group, idx, args.dry_run): record
                for idx, record in batch
            }
            for future in as_completed(futures):
                record = futures[future]
                try:
                    out = future.result()
                except Exception as exc:
                    failures += 1
                    idx = next((i for i, item in batch if item["candidate_id"] == record["candidate_id"]), 0)
                    append_jsonl(args.out, [fallback_caption(record, args.group, idx, exc)])
                    success += 1
                    append_jsonl(
                        args.failures,
                        [{"candidate_id": record["candidate_id"], "group": args.group, "error": repr(exc)}],
                    )
                    continue
                append_jsonl(args.out, [out])
                success += 1
        print(f"dense captions success={success} failures={failures}", flush=True)
    write_json(
        args.out.with_suffix(".report.json"),
        {"success": success, "failures": failures, "dry_run": args.dry_run},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
