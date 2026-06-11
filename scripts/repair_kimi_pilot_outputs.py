#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image
import requests

from pipeline_common import (
    KIMI_MODEL,
    api_key,
    WORK_ROOT,
    append_jsonl,
    extract_final_answer,
    extract_json_object,
    image_part,
    iter_jsonl,
    kimi_generate,
    normalize_answer,
    write_json,
    write_jsonl,
)


MESSAGES_ENDPOINT = "https://models-proxy.stepfun-inc.com/v1/messages"
MESSAGES_MODEL = "kimi-k2.6-aliyun"


THINKING_PROMPT = """Reason from visible chart evidence only, then end with:
Final answer: {answer}

Do not use paper background or caption-only claims. The final answer must exactly be the given consensus answer.

Question: {question}
Consensus answer: {answer}
Task type: {task_type}
"""


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
    parser = argparse.ArgumentParser(description="Repair failed Kimi pilot outputs.")
    parser.add_argument("--kind", choices=["thinking", "caption"], required=True)
    parser.add_argument("--group", choices=["inclusive", "exclusive"], required=True)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--image-max-pixels", type=int, default=100000)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--provider", choices=["chat", "messages"], default="chat")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--work", type=Path, default=WORK_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def jsonl_by_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {record["id"]: record for record in iter_jsonl(path)}


def jsonl_by_candidate_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    for record in iter_jsonl(path):
        cid = (record.get("source") or {}).get("candidate_id")
        if cid:
            result[cid] = record
    return result


def backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = path.with_suffix(path.suffix + f".bak_{stamp}")
    shutil.copy2(path, out)
    return out


def resized_image_path(src: str, cache_dir: Path, max_pixels: int) -> Path:
    source = Path(src)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = source.stem[:80].replace("/", "_")
    dst = cache_dir / f"{stem}_{source.stat().st_size}_{max_pixels}.jpg"
    if dst.exists():
        return dst
    with Image.open(source) as img:
        img = img.convert("RGB")
        width, height = img.size
        scale = min(1.0, math.sqrt(max_pixels / (width * height)))
        new_width = max(1, int(width * scale))
        new_height = max(1, int(height * scale))
        if (new_width, new_height) != (width, height):
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img.save(dst, format="JPEG", quality=90, optimize=True)
    return dst


def content_parts(image: str, text: str, cache_dir: Path, max_pixels: int) -> list[dict[str, Any]]:
    resized = resized_image_path(image, cache_dir, max_pixels)
    inline = image_part(resized)["inlineData"]
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{inline['mimeType']};base64,{inline['data']}"},
        },
        {"type": "text", "text": text},
    ]


def messages_generate(
    image: str,
    text: str,
    *,
    cache_dir: Path,
    image_max_pixels: int,
    max_tokens: int,
    timeout: int,
    retries: int,
) -> str:
    resized = resized_image_path(image, cache_dir, image_max_pixels)
    inline = image_part(resized)["inlineData"]
    payload = {
        "model": MESSAGES_MODEL,
        "max_tokens": max_tokens,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": inline["mimeType"],
                            "data": inline["data"],
                        },
                    },
                    {"type": "text", "text": text},
                ],
            }
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key()}",
        "anthropic-version": "2023-06-01",
    }
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.post(MESSAGES_ENDPOINT, headers=headers, json=payload, timeout=timeout)
            if response.status_code >= 400:
                raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
            data = response.json()
            content = data.get("content") or []
            return "".join(part.get("text", "") for part in content if isinstance(part, dict)).strip()
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
            time.sleep(min(60, 2 + attempt * 5))
    assert last_exc is not None
    raise last_exc


def generate_with_provider(
    provider: str,
    image: str,
    prompt: str,
    *,
    cache_dir: Path,
    image_max_pixels: int,
    max_tokens: int,
    timeout: int,
    retries: int,
) -> str:
    if provider == "messages":
        return messages_generate(
            image,
            prompt,
            cache_dir=cache_dir,
            image_max_pixels=image_max_pixels,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries,
        )
    return kimi_generate(
        [{"role": "user", "content": content_parts(image, prompt, cache_dir, image_max_pixels)}],
        max_tokens=max_tokens,
        temperature=1,
        top_p=1,
        top_k=-1,
        timeout=timeout,
        retries=retries,
    )


def model_name(provider: str) -> str:
    return MESSAGES_MODEL if provider == "messages" else KIMI_MODEL


def direct_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        **record,
        "messages": [
            {"role": "user", "content": "<image>\n" + record["question"]},
            {"role": "assistant", "content": record["answer"]},
        ],
    }


def repair_thinking_one(
    record: dict[str, Any],
    *,
    cache_dir: Path,
    image_max_pixels: int,
    max_tokens: int,
    retries: int,
    timeout: int,
    provider: str,
) -> dict[str, Any]:
    prompt = THINKING_PROMPT.format(
        question=record["question"],
        answer=record["answer"],
        task_type=record.get("task_type", ""),
    )
    response = generate_with_provider(
        provider,
        record["image"],
        prompt,
        cache_dir=cache_dir,
        image_max_pixels=image_max_pixels,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
    )
    answer_type = record.get("answer_type", "short_text")
    ok = normalize_answer(extract_final_answer(response), answer_type) == normalize_answer(record["answer"], answer_type)
    out = dict(record)
    out["thinking_response"] = {
        "model": model_name(provider),
        "model_config": {
            "max_tokens": max_tokens,
            "temperature": 1,
            "top_p": 1,
            "top_k": -1,
            "image_max_pixels": image_max_pixels,
            "provider": provider,
        },
        "response": response,
        "final_answer_matches_consensus": ok,
        "repair_run": True,
    }
    out["thinking_response_failed"] = not ok
    out["messages"] = [
        {"role": "user", "content": "<image>\n" + record["question"]},
        {"role": "assistant", "content": response if ok else record["answer"]},
    ]
    return out


def fallback_thinking_record(record: dict[str, Any], error: Exception, max_tokens: int, image_max_pixels: int, provider: str = "chat") -> dict[str, Any]:
    out = dict(record)
    out["thinking_response"] = {
        "model": model_name(provider),
        "model_config": {
            "max_tokens": max_tokens,
            "temperature": 1,
            "top_p": 1,
            "top_k": -1,
            "image_max_pixels": image_max_pixels,
            "provider": provider,
        },
        "response": record["answer"],
        "final_answer_matches_consensus": True,
        "error": repr(error),
        "repair_run": True,
    }
    out["thinking_response_failed"] = True
    out["messages"] = [
        {"role": "user", "content": "<image>\n" + record["question"]},
        {"role": "assistant", "content": record["answer"]},
    ]
    return out


def write_thinking_merge(
    out_path: Path,
    consensus_records: list[dict[str, Any]],
    existing: dict[str, dict[str, Any]],
    repaired: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    group_name: str,
    targets_count: int,
    backup_path: Path | None,
) -> int:
    merged = []
    failed_after = 0
    for record in consensus_records:
        out = repaired.get(record["id"]) or existing.get(record["id"]) or fallback_thinking_record(
            record, RuntimeError("missing thinking output"), args.max_tokens, args.image_max_pixels, args.provider
        )
        failed_after += int(bool(out.get("thinking_response_failed")))
        merged.append(out)
    write_jsonl(out_path, merged)
    write_json(
        out_path.with_suffix(".report.json"),
        {
            "group": group_name,
            "total": len(merged),
            "repair_targets": targets_count,
            "repaired": len(repaired),
            "failed_after": failed_after,
            "backup": str(backup_path) if backup_path else None,
            "max_tokens": args.max_tokens,
            "image_max_pixels": args.image_max_pixels,
            "workers": args.workers,
            "batch_size": args.batch_size,
            "timeout": args.timeout,
            "provider": args.provider,
        },
    )
    return failed_after


def repair_thinking(args: argparse.Namespace) -> None:
    group_name = f"charxiv_{args.group}_50k"
    qa_dir = args.work / "qa"
    input_path = qa_dir / f"{group_name}.consensus.jsonl"
    out_path = qa_dir / f"{group_name}.qa_thinking.jsonl"
    direct_path = qa_dir / f"{group_name}.qa_direct.jsonl"
    failure_path = args.work / "logs" / "kimi_thinking_repair_failures.jsonl"
    cache_dir = args.work / "tmp" / "resized_kimi_100k" / "thinking" / group_name

    consensus_records = list(iter_jsonl(input_path))
    existing = jsonl_by_id(out_path)
    direct_existing = jsonl_by_id(direct_path)
    if not args.dry_run:
        append_jsonl(direct_path, [direct_record(r) for r in consensus_records if r["id"] not in direct_existing])

    targets = [
        r
        for r in consensus_records
        if r["id"] not in existing or existing[r["id"]].get("thinking_response_failed")
    ]
    if args.limit:
        targets = targets[: args.limit]
    print(f"thinking repair group={group_name} targets={len(targets)} existing={len(existing)}", flush=True)
    if args.dry_run:
        return

    repaired: dict[str, dict[str, Any]] = {}
    failures = 0
    backup_path = backup(out_path)
    for start in range(0, len(targets), args.batch_size):
        batch = targets[start : start + args.batch_size]
        batch_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    repair_thinking_one,
                    record,
                    cache_dir=cache_dir,
                    image_max_pixels=args.image_max_pixels,
                    max_tokens=args.max_tokens,
                    retries=args.retries,
                    timeout=args.timeout,
                    provider=args.provider,
                ): record
                for record in batch
            }
            for future in as_completed(futures):
                record = futures[future]
                try:
                    out = future.result()
                except Exception as exc:
                    failures += 1
                    out = fallback_thinking_record(record, exc, args.max_tokens, args.image_max_pixels)
                    append_jsonl(
                        failure_path,
                        [{"id": record["id"], "group": group_name, "question": record.get("question"), "error": repr(exc)}],
                    )
                repaired[record["id"]] = out
                status = "failed" if out.get("thinking_response_failed") else "ok"
                print(f"thinking {group_name} {status} {record['id']}", flush=True)
                existing[record["id"]] = out
        failed_after = write_thinking_merge(
            out_path,
            consensus_records,
            existing,
            repaired,
            args,
            group_name,
            len(targets),
            backup_path,
        )
        print(
            f"thinking batch group={group_name} done={min(start + len(batch), len(targets))}/{len(targets)} "
            f"elapsed={time.perf_counter() - batch_start:.1f}s failures={failures} failed_after={failed_after}",
            flush=True,
        )

    failed_after = write_thinking_merge(
        out_path,
        consensus_records,
        existing,
        repaired,
        args,
        group_name,
        len(targets),
        backup_path,
    )
    print(f"thinking wrote {out_path} failed_after={failed_after}", flush=True)


def caption_record(record: dict[str, Any], group_name: str, idx: int, result: dict[str, Any], max_tokens: int, image_max_pixels: int) -> dict[str, Any]:
    dense_caption = result.get("dense_caption", "")
    return {
        "id": f"arxiv_chart_caption_{group_name}_{idx:06d}",
        "group": group_name,
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
            "generation_failed": False,
            "repair_run": True,
            "model_config": {
                "max_tokens": max_tokens,
                "temperature": 1,
                "top_p": 1,
                "top_k": -1,
                "image_max_pixels": image_max_pixels,
            },
        },
    }


def repair_caption_one(
    record: dict[str, Any],
    group_name: str,
    idx: int,
    *,
    cache_dir: Path,
    image_max_pixels: int,
    max_tokens: int,
    retries: int,
    timeout: int,
    provider: str,
) -> dict[str, Any]:
    prompt = CAPTION_PROMPT.format(caption_latex=(record.get("caption_latex") or "")[:5000])
    raw = generate_with_provider(
        provider,
        record["image_path"],
        prompt,
        cache_dir=cache_dir,
        image_max_pixels=image_max_pixels,
        max_tokens=max_tokens,
        timeout=timeout,
        retries=retries,
    )
    result = extract_json_object(raw)
    return caption_record(record, group_name, idx, result, max_tokens, image_max_pixels)


def fallback_caption_record(
    record: dict[str, Any],
    group_name: str,
    idx: int,
    error: Exception,
    max_tokens: int,
    image_max_pixels: int,
) -> dict[str, Any]:
    dense_caption = (
        "The image is a chart selected from an arXiv figure extraction pipeline. "
        "Automatic dense caption generation failed for this item, so this record should be regenerated or filtered before training."
    )
    out = caption_record(
        record,
        group_name,
        idx,
        {
            "dense_caption": dense_caption,
            "visible_elements": {
                "chart_types": record.get("classifier", {}).get("chart_types") or [],
                "axes": [],
                "series_or_panels": [],
                "main_trends": [],
            },
            "uncertainty": ["dense_caption_generation_failed", repr(error)],
        },
        max_tokens,
        image_max_pixels,
    )
    out["quality"]["generation_failed"] = True
    out["quality"]["error"] = repr(error)
    return out


def caption_failed(record: dict[str, Any] | None) -> bool:
    if record is None:
        return True
    quality = record.get("quality") or {}
    if quality.get("generation_failed"):
        return True
    dense_caption = record.get("dense_caption") or ""
    return "Automatic dense caption generation failed" in dense_caption


def write_caption_merge(
    out_path: Path,
    indexed: list[tuple[int, dict[str, Any]]],
    existing_by_cid: dict[str, dict[str, Any]],
    repaired: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    group_name: str,
    targets_count: int,
    backup_path: Path | None,
) -> int:
    merged = []
    failed_after = 0
    for idx, record in indexed:
        cid = record["candidate_id"]
        out = repaired.get(cid) or existing_by_cid.get(cid) or fallback_caption_record(
            record, group_name, idx, RuntimeError("missing dense caption output"), args.max_tokens, args.image_max_pixels
        )
        failed_after += int(caption_failed(out))
        merged.append(out)
    write_jsonl(out_path, merged)
    write_json(
        out_path.with_suffix(".report.json"),
        {
            "group": group_name,
            "total": len(merged),
            "repair_targets": targets_count,
            "repaired": len(repaired),
            "failed_after": failed_after,
            "backup": str(backup_path) if backup_path else None,
            "max_tokens": args.max_tokens,
            "image_max_pixels": args.image_max_pixels,
            "workers": args.workers,
            "batch_size": args.batch_size,
            "timeout": args.timeout,
            "provider": args.provider,
        },
    )
    return failed_after


def repair_caption(args: argparse.Namespace) -> None:
    group_name = f"charxiv_{args.group}_50k"
    sample_path = args.work / f"sample_charxiv_{args.group}_50k.jsonl"
    out_path = args.work / "dense_caption" / f"{group_name}.dense_caption.jsonl"
    failure_path = args.work / "logs" / "dense_caption_repair_failures.jsonl"
    cache_dir = args.work / "tmp" / "resized_kimi_100k" / "caption" / group_name

    sample_records = list(iter_jsonl(sample_path))
    existing_by_cid = jsonl_by_candidate_id(out_path)
    indexed = [(idx, record) for idx, record in enumerate(sample_records, 1)]
    targets = [
        (idx, record)
        for idx, record in indexed
        if caption_failed(existing_by_cid.get(record["candidate_id"]))
    ]
    if args.limit:
        targets = targets[: args.limit]
    print(f"caption repair group={group_name} targets={len(targets)} existing={len(existing_by_cid)}", flush=True)
    if args.dry_run:
        return

    repaired: dict[str, dict[str, Any]] = {}
    failures = 0
    backup_path = backup(out_path)
    for start in range(0, len(targets), args.batch_size):
        batch = targets[start : start + args.batch_size]
        batch_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    repair_caption_one,
                    record,
                    group_name,
                    idx,
                    cache_dir=cache_dir,
                    image_max_pixels=args.image_max_pixels,
                    max_tokens=args.max_tokens,
                    retries=args.retries,
                    timeout=args.timeout,
                    provider=args.provider,
                ): (idx, record)
                for idx, record in batch
            }
            for future in as_completed(futures):
                idx, record = futures[future]
                cid = record["candidate_id"]
                try:
                    out = future.result()
                except Exception as exc:
                    failures += 1
                    out = fallback_caption_record(record, group_name, idx, exc, args.max_tokens, args.image_max_pixels)
                    append_jsonl(
                        failure_path,
                        [{"candidate_id": cid, "group": group_name, "error": repr(exc)}],
                    )
                repaired[cid] = out
                existing_by_cid[cid] = out
                status = "failed" if out.get("quality", {}).get("generation_failed") else "ok"
                print(f"caption {group_name} {status} {cid}", flush=True)
        failed_after = write_caption_merge(
            out_path,
            indexed,
            existing_by_cid,
            repaired,
            args,
            group_name,
            len(targets),
            backup_path,
        )
        print(
            f"caption batch group={group_name} done={min(start + len(batch), len(targets))}/{len(targets)} "
            f"elapsed={time.perf_counter() - batch_start:.1f}s failures={failures} failed_after={failed_after}",
            flush=True,
        )

    failed_after = write_caption_merge(
        out_path,
        indexed,
        existing_by_cid,
        repaired,
        args,
        group_name,
        len(targets),
        backup_path,
    )
    print(f"caption wrote {out_path} failed_after={failed_after}", flush=True)


def main() -> int:
    args = parse_args()
    if args.kind == "thinking":
        repair_thinking(args)
    else:
        repair_caption(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
