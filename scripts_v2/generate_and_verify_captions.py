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
    extract_json_object,
    gemini_generate,
    image_part_gemini,
    iter_jsonl,
    kimi_messages_generate,
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


CAPTION_JUDGE_PROMPT = """You are verifying a dense caption for a chart image.

Use only the visible chart image.

Dense caption:
{dense_caption}

Visible elements JSON:
{visible_elements}

Check:
- Caption is grounded in the image.
- Caption does not hallucinate paper background, methods, datasets, or conclusions.
- Caption covers chart type, axes/scale, legend/series/panels, and major trend/comparison when visible.
- Caption does not state unreadable text as certain.

Return strict JSON only:
{{
  "verdict": "pass",
  "caption_grounded": true,
  "has_hallucination": false,
  "coverage_ok": true,
  "unreadable_text_handled": true,
  "reason": "..."
}}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Kimi dense captions and verify with Gemini.")
    parser.add_argument("--input", type=Path, default=EDIT2_ROOT / "filtered_charts_2020_2025.jsonl")
    parser.add_argument("--raw-out", type=Path, default=EDIT2_ROOT / "dense_caption_raw.jsonl")
    parser.add_argument("--verified-out", type=Path, default=EDIT2_ROOT / "dense_caption_verified.jsonl")
    parser.add_argument("--failures", type=Path, default=EDIT2_ROOT / "logs" / "caption_failures.jsonl")
    parser.add_argument("--judge-failures", type=Path, default=EDIT2_ROOT / "logs" / "caption_judge_failures.jsonl")
    parser.add_argument("--report", type=Path, default=EDIT2_ROOT / "reports" / "dense_caption_verified.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-max-pixels", type=int, default=100000)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--judge-image-max-pixels", type=int, default=350000)
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry records that only exist in raw/judge-failure outputs; verified records are still skipped.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def source_id(record: dict[str, Any]) -> str:
    return str(record.get("candidate_id") or (record.get("source") or {}).get("candidate_id") or record.get("id"))


def done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {source_id(record) for record in iter_jsonl(path)}


def caption_failed(result: dict[str, Any]) -> bool:
    dense_caption = str(result.get("dense_caption") or "").strip()
    return not dense_caption or "Automatic dense caption generation failed" in dense_caption


def generate_caption(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cid = source_id(record)
    if args.dry_run:
        result = {
            "dense_caption": "The image shows a chart with visible axes and plotted data. The caption should be verified against the chart image.",
            "visible_elements": {"chart_types": [], "axes": [], "series_or_panels": [], "main_trends": []},
            "uncertainty": ["dry_run"],
        }
    else:
        raw = kimi_messages_generate(
            image_path=Path(record["image_path"] if "image_path" in record else record["image"]),
            text=CAPTION_PROMPT.format(caption_latex=(record.get("caption_latex") or (record.get("source") or {}).get("caption_latex") or "")[:5000]),
            cache_dir=EDIT2_ROOT / "tmp" / "kimi_caption_images",
            image_max_pixels=args.image_max_pixels,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
        )
        result = extract_json_object(raw)
        result["raw_response"] = raw
    if caption_failed(result):
        raise ValueError("empty_or_failed_caption")
    image = record["image_path"] if "image_path" in record else record["image"]
    out = {
        "id": f"caption_{cid}",
        "candidate_id": cid,
        "image": image,
        "source": {
            "candidate_id": cid,
            "paper_id": record.get("paper_id") or (record.get("source") or {}).get("paper_id"),
            "year": record.get("year") or (record.get("source") or {}).get("year"),
            "month": record.get("month") or (record.get("source") or {}).get("month"),
            "figure_index": record.get("figure_index") or (record.get("source") or {}).get("figure_index"),
            "image_kind": record.get("image_kind") or (record.get("source") or {}).get("image_kind"),
            "is_charxiv_paper": record.get("is_charxiv_paper", (record.get("source") or {}).get("is_charxiv_paper", False)),
            "json_path": record.get("json_path") or (record.get("source") or {}).get("json_path"),
            "caption_latex": record.get("caption_latex") or (record.get("source") or {}).get("caption_latex", ""),
            "classifier": record.get("classifier") or (record.get("source") or {}).get("classifier") or {},
        },
        "task_type": "dense_caption",
        "evidence_source": "image_only",
        "dense_caption": result.get("dense_caption", ""),
        "visible_elements": result.get("visible_elements", {}),
        "uncertainty": result.get("uncertainty", []),
        "caption_generation": {
            "model": KIMI_MESSAGES_MODEL,
            "protocol": "anthropic_messages",
            "model_config": {
                "max_tokens": args.max_tokens,
                "image_max_pixels": args.image_max_pixels,
                "timeout": args.timeout,
                "retries": args.retries,
            },
            "raw_response": result.get("raw_response"),
            "dry_run": args.dry_run,
        },
        "messages": [
            {"role": "user", "content": "<image>\nDescribe this chart in detail using only visible information."},
            {"role": "assistant", "content": result.get("dense_caption", "")},
        ],
    }
    return out


def judge_caption(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.dry_run:
        result = {
            "verdict": "pass",
            "caption_grounded": True,
            "has_hallucination": False,
            "coverage_ok": True,
            "unreadable_text_handled": True,
            "reason": "dry_run",
        }
    else:
        raw = gemini_generate(
            [
                image_part_gemini(
                    Path(record["image"]),
                    cache_dir=EDIT2_ROOT / "tmp" / "gemini_caption_judge_images",
                    max_pixels=args.judge_image_max_pixels,
                ),
                {
                    "text": CAPTION_JUDGE_PROMPT.format(
                        dense_caption=record.get("dense_caption", ""),
                        visible_elements=record.get("visible_elements", {}),
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
        and bool(result.get("caption_grounded"))
        and not bool(result.get("has_hallucination"))
        and bool(result.get("coverage_ok"))
        and bool(result.get("unreadable_text_handled"))
    )
    out = dict(record)
    out["caption_judge"] = {
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
    out["caption_verified"] = passed
    return out


def generate_and_judge(record: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any] | None]:
    last: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            raw = generate_caption(record, args)
            judged = judge_caption(raw, args)
            if judged.get("caption_verified"):
                return raw, judged
            raw["caption_judge_error"] = f"caption_judge_failed:{judged.get('caption_judge')}"
            if attempt >= args.retries:
                return raw, None
            last = RuntimeError(raw["caption_judge_error"])
        except Exception as exc:
            last = exc
            if attempt >= args.retries:
                raise
            time.sleep(1 + attempt)
    assert last is not None
    raise last


def main() -> int:
    args = parse_args()
    done = done_ids(args.verified_out)
    if not args.retry_failed:
        done |= done_ids(args.raw_out)
    records = [record for record in iter_jsonl(args.input) if source_id(record) not in done]
    if args.limit:
        records = records[: args.limit]
    print(f"generating captions records={len(records)}", flush=True)
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
                    append_jsonl(args.failures, [{"candidate_id": source_id(record), "error": repr(exc)}])
                    continue
                append_jsonl(args.raw_out, [raw])
                raw_count += 1
                if verified:
                    append_jsonl(args.verified_out, [verified])
                    verified_count += 1
                else:
                    append_jsonl(args.judge_failures, [raw])
        print(
            f"captions raw={raw_count} verified={verified_count} failures={failures} "
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
