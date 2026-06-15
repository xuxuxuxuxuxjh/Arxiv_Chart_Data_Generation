#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import warnings
from pathlib import Path
from typing import Any

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build static review HTML from edition2 merged QA/caption JSONL.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-side", type=int, default=1400)
    parser.add_argument("--max-pixels", type=int, default=1800000)
    return parser.parse_args()


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            records.append(record)
    return records


def save_web_image(src: Path, dst: Path, max_side: int, max_pixels: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(src) as img:
            img.load()
            img = img.convert("RGBA")
            width, height = img.size
            scale = min(1.0, max_side / max(width, height))
            if max_pixels > 0:
                scale = min(scale, math.sqrt(max_pixels / max(width * height, 1)))
            if scale < 1.0:
                img = img.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            bg.alpha_composite(img)
            bg.convert("RGB").save(dst, "JPEG", quality=90, optimize=True)


def esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def thinking_text(record: dict[str, Any]) -> str:
    thinking = record.get("kimi_thinking") or {}
    return str(thinking.get("response") or "")


def answer_reasoning(record: dict[str, Any]) -> str:
    generation = record.get("answer_generation") or {}
    return str(generation.get("thinking") or "")


def caption_reasoning(record: dict[str, Any]) -> str:
    generation = record.get("caption_generation") or {}
    thinking = str(record.get("caption_thinking") or generation.get("thinking") or "")
    if thinking and "<think>" not in thinking:
        return f"<think>\n{thinking}\n</think>"
    return thinking


def compact_json(value: Any, limit: int = 1200) -> str:
    text = json.dumps(value or {}, ensure_ascii=False, indent=2)
    return text[:limit]


def build_html(records: list[dict[str, Any]], image_rels: list[str], failures: list[str]) -> str:
    rows: list[str] = []
    for idx, (record, image_rel) in enumerate(zip(records, image_rels), 1):
        verified = record.get("verified") or {}
        meta = " | ".join(
            [
                f"candidate={record.get('candidate_id')}",
                f"task={record.get('task_type')}",
                f"answer_type={record.get('answer_type')}",
                f"difficulty={record.get('difficulty')}",
                f"verified={verified}",
            ]
        )
        rows.append(
            f"""
    <article class="review-item" id="item-{idx}">
      <header class="item-head">
        <div>
          <div class="item-index">#{idx:03d}</div>
          <h2>{esc(record.get("question"))}</h2>
          <p class="meta">{esc(meta)}</p>
        </div>
        <a class="jump" href="#top">Top</a>
      </header>
      <div class="content-grid">
        <figure class="chart-frame">
          <img src="{esc(image_rel)}" loading="lazy" alt="chart {idx}">
          <figcaption>{esc(record.get("image"))}</figcaption>
        </figure>
        <section class="qa-panel">
          <div class="field">
            <div class="label">Answer</div>
            <div class="answer">{esc(record.get("answer"))}</div>
          </div>
          <details class="field" open>
            <summary>Gemini Answer Thinking</summary>
            <pre>{esc(answer_reasoning(record))}</pre>
          </details>
          <details class="field thinking" open>
            <summary>Kimi Thinking</summary>
            <pre>{esc(thinking_text(record))}</pre>
          </details>
          <div class="field">
            <div class="label">Dense Caption</div>
            <p>{esc(record.get("dense_caption"))}</p>
          </div>
          <details class="field thinking">
            <summary>Kimi Caption Thinking</summary>
            <pre>{esc(caption_reasoning(record))}</pre>
          </details>
          <details class="field">
            <summary>Judges</summary>
            <pre>{esc(compact_json({"answer_judge": record.get("answer_judge"), "thinking_judge": record.get("kimi_thinking_judge"), "caption_judge": record.get("caption_judge")}))}</pre>
          </details>
        </section>
      </div>
    </article>"""
        )

    failure_html = ""
    if failures:
        failure_html = (
            "<section class=\"failures\"><h2>Image Copy Failures</h2><pre>"
            + esc("\n".join(failures))
            + "</pre></section>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>arXiv Chart Edition2 10-Image Review</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #1d252d;
      --muted: #667380;
      --line: #d7dde4;
      --accent: #2563a8;
      --accent-soft: #e9f2ff;
      --answer: #116149;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Arial, Helvetica, sans-serif; line-height: 1.45; }}
    .topbar {{ position: sticky; top: 0; z-index: 10; display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 14px 22px; border-bottom: 1px solid var(--line); background: rgba(255, 255, 255, 0.96); }}
    .title h1 {{ margin: 0; font-size: 20px; font-weight: 700; letter-spacing: 0; }}
    .title p {{ margin: 2px 0 0; color: var(--muted); font-size: 13px; }}
    .nav {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; justify-content: flex-end; }}
    .nav a, .jump {{ color: var(--accent); background: var(--accent-soft); border: 1px solid #c7dcf7; text-decoration: none; padding: 6px 9px; border-radius: 6px; font-size: 13px; }}
    main {{ max-width: 1560px; margin: 0 auto; padding: 18px 22px 48px; }}
    .review-item {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; margin: 0 0 18px; overflow: hidden; }}
    .item-head {{ display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; padding: 14px 16px; border-bottom: 1px solid var(--line); background: #fbfcfe; }}
    .item-index {{ color: var(--muted); font-size: 13px; margin-bottom: 4px; }}
    h2 {{ margin: 0; font-size: 18px; font-weight: 700; letter-spacing: 0; }}
    .meta {{ margin: 6px 0 0; color: var(--muted); font-size: 12px; }}
    .content-grid {{ display: grid; grid-template-columns: minmax(420px, 0.9fr) minmax(460px, 1.1fr); gap: 16px; padding: 16px; align-items: start; }}
    .chart-frame {{ margin: 0; position: sticky; top: 86px; }}
    .chart-frame img {{ display: block; width: 100%; max-height: 78vh; object-fit: contain; border: 1px solid var(--line); background: #fff; }}
    figcaption {{ margin-top: 8px; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }}
    .qa-panel {{ display: grid; gap: 12px; }}
    .field {{ border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fff; }}
    .label, summary {{ color: var(--muted); font-size: 13px; font-weight: 700; margin-bottom: 8px; cursor: pointer; }}
    .answer {{ color: var(--answer); font-size: 18px; font-weight: 700; }}
    p {{ margin: 0; }}
    pre {{ margin: 8px 0 0; white-space: pre-wrap; overflow-wrap: anywhere; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; line-height: 1.5; }}
    .thinking pre {{ max-height: 480px; overflow: auto; }}
    .failures {{ margin-bottom: 18px; padding: 14px; border: 1px solid #e2b7b7; background: #fff7f7; border-radius: 8px; }}
    @media (max-width: 980px) {{
      .topbar {{ align-items: flex-start; flex-direction: column; }}
      .nav {{ justify-content: flex-start; }}
      .content-grid {{ grid-template-columns: 1fr; }}
      .chart-frame {{ position: static; }}
    }}
  </style>
</head>
<body>
  <header class="topbar" id="top">
    <div class="title">
      <h1>arXiv Chart Edition2 10-Image Review</h1>
      <p>{len(records)} merged QA/thinking/caption records · static local images</p>
    </div>
    <nav class="nav">
      <a href="#item-1">First</a>
      <a href="#item-{max(1, len(records) // 2)}">Middle</a>
      <a href="#item-{len(records)}">Last</a>
    </nav>
  </header>
  <main>
    {failure_html}
    {"".join(rows)}
  </main>
</body>
</html>
"""


def main() -> int:
    args = parse_args()
    records = iter_jsonl(args.input)
    if args.limit:
        records = records[: args.limit]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "assets").mkdir(parents=True, exist_ok=True)

    image_rels: list[str] = []
    failures: list[str] = []
    for idx, record in enumerate(records, 1):
        src = Path(str(record.get("image") or ""))
        rel = f"assets/chart_{idx:04d}.jpg"
        try:
            save_web_image(src, args.out_dir / rel, args.max_side, args.max_pixels)
            image_rels.append(rel)
        except Exception as exc:
            failures.append(f"{idx}: {src}: {exc!r}")
            image_rels.append("")

    (args.out_dir / "index.html").write_text(build_html(records, image_rels, failures), encoding="utf-8")
    (args.out_dir / "records.jsonl").write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + ("\n" if records else ""),
        encoding="utf-8",
    )
    (args.out_dir / "image_failures.txt").write_text("\n".join(failures), encoding="utf-8")
    print(f"wrote {args.out_dir / 'index.html'}", flush=True)
    print(f"images copied: {len(records) - len(failures)}/{len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
