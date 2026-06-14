#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from common_v2 import (
    EDIT2_ROOT,
    KIMI_MESSAGES_MODEL,
    append_jsonl,
    extract_json_object,
    iter_jsonl,
    kimi_messages_generate,
    write_json,
)


CAPTION_SCHEMA_VERSION = "kimi_caption_no_verify_v1"


CAPTION_PROMPT = """Describe this chart in detail using the chart image and the provided caption_latex.

Requirements:
- Use natural language grounded in the image and caption_latex.
- Use caption_latex to preserve domain terms, method names, variable names, panel descriptions, and series names.
- Include chart type, axes, legend/series, visual encodings, major trends/comparisons, and multi-panel layout when visible.
- Do not invent facts that are not supported by either the image or caption_latex.
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

caption_latex:
{caption_latex}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Kimi dense captions without a separate verifier.")
    parser.add_argument("--input", type=Path, default=EDIT2_ROOT / "filtered_charts_2020_2025.jsonl")
    parser.add_argument("--raw-out", type=Path, default=EDIT2_ROOT / "dense_caption_raw.jsonl")
    parser.add_argument("--verified-out", type=Path, default=EDIT2_ROOT / "dense_caption_verified.jsonl")
    parser.add_argument("--failures", type=Path, default=EDIT2_ROOT / "logs" / "caption_failures.jsonl")
    parser.add_argument(
        "--judge-failures",
        type=Path,
        default=EDIT2_ROOT / "logs" / "caption_judge_failures.jsonl",
        help="Compatibility argument; caption verification is disabled in the current pipeline.",
    )
    parser.add_argument("--report", type=Path, default=EDIT2_ROOT / "reports" / "dense_caption_verified.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-max-pixels", type=int, default=1000000)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument(
        "--judge-image-max-pixels",
        type=int,
        default=350000,
        help="Compatibility argument; caption verification is disabled in the current pipeline.",
    )
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
    ids: set[str] = set()
    for record in iter_jsonl(path):
        generation = record.get("caption_generation") or {}
        if generation.get("schema_version") == CAPTION_SCHEMA_VERSION:
            ids.add(source_id(record))
    return ids


def caption_failed(result: dict[str, Any]) -> bool:
    dense_caption = str(result.get("dense_caption") or "").strip()
    return not dense_caption or "Automatic dense caption generation failed" in dense_caption


def caption_latex_text(record: dict[str, Any]) -> str:
    source = record.get("source") or {}
    return str(record.get("caption_latex") or source.get("caption_latex") or "").strip()


def parse_caption_response(raw: str) -> dict[str, Any]:
    try:
        result = extract_json_object(raw)
    except Exception:
        caption = raw.strip()
        if caption.lower().startswith("```"):
            caption = caption.strip("`").strip()
        result = {
            "dense_caption": caption,
            "visible_elements": {"chart_types": [], "axes": [], "series_or_panels": [], "main_trends": []},
            "uncertainty": ["caption_response_was_not_json"],
        }
    result["raw_response"] = raw
    return result


def generate_caption(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cid = source_id(record)
    caption_latex = caption_latex_text(record)
    if args.dry_run:
        result = {
            "dense_caption": "The image shows a chart with visible axes and plotted data. The provided caption_latex supplies figure terminology for the dense caption.",
            "visible_elements": {"chart_types": [], "axes": [], "series_or_panels": [], "main_trends": []},
            "uncertainty": ["dry_run"],
            "raw_response": None,
        }
    else:
        raw = kimi_messages_generate(
            image_path=Path(record["image_path"] if "image_path" in record else record["image"]),
            text=CAPTION_PROMPT.format(caption_latex=caption_latex[:5000]),
            cache_dir=EDIT2_ROOT / "tmp" / "kimi_caption_images",
            image_max_pixels=args.image_max_pixels,
            max_tokens=args.max_tokens,
            timeout=args.timeout,
            retries=args.retries,
        )
        result = parse_caption_response(raw)
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
            "caption_latex": caption_latex,
            "classifier": record.get("classifier") or (record.get("source") or {}).get("classifier") or {},
        },
        "task_type": "dense_caption",
        "evidence_source": "image_and_caption_latex",
        "dense_caption": result.get("dense_caption", ""),
        "visible_elements": result.get("visible_elements", {}),
        "uncertainty": result.get("uncertainty", []),
        "caption_generation": {
            "schema_version": CAPTION_SCHEMA_VERSION,
            "model": KIMI_MESSAGES_MODEL,
            "protocol": "openai_chat_completions_streaming",
            "model_config": {
                "max_tokens": args.max_tokens,
                "image_max_pixels": args.image_max_pixels,
                "timeout": args.timeout,
                "retries": args.retries,
            },
            "caption_latex_used": bool(caption_latex),
            "raw_response": result.get("raw_response"),
            "dry_run": args.dry_run,
        },
        "caption_judge": {
            "schema_version": CAPTION_SCHEMA_VERSION,
            "verdict": "skipped",
            "passed": True,
            "reason": "caption verification disabled; successful Kimi generation is accepted",
            "dry_run": args.dry_run,
        },
        "caption_verified": True,
        "messages": [
            {"role": "user", "content": "<image>\n" + CAPTION_PROMPT.format(caption_latex=caption_latex[:5000])},
            {"role": "assistant", "content": result.get("dense_caption", "")},
        ],
    }
    return out


def generate_and_judge(record: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any] | None]:
    last: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            raw = generate_caption(record, args)
            return raw, raw
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
                    append_jsonl(args.failures, [{"candidate_id": source_id(record), "error": raw.get("caption_judge_error", "caption_generation_failed")}])
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
