#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import warnings
from pathlib import Path
from urllib.parse import unquote

from PIL import Image

from common_v2 import EDIT2_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export edition2 review HTML with local resized image assets.")
    parser.add_argument("--html", type=Path, default=EDIT2_ROOT / "review" / "review.html")
    parser.add_argument("--out", type=Path, default=EDIT2_ROOT / "review_static")
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--max-pixels", type=int, default=100000)
    return parser.parse_args()


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
            bg.convert("RGB").save(dst, "JPEG", quality=92, optimize=True)


def main() -> int:
    args = parse_args()
    if not args.html.exists():
        raise FileNotFoundError(args.html)
    args.out.mkdir(parents=True, exist_ok=True)
    assets = args.out / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    html = args.html.read_text(encoding="utf-8", errors="ignore")
    paths = re.findall(r'src="file://([^"]+)"', html)
    mapping: dict[str, str] = {}
    failed: list[tuple[str, str]] = []
    for idx, raw in enumerate(paths, 1):
        src = Path(unquote(raw))
        rel = f"assets/img_{idx:04d}.jpg"
        try:
            save_web_image(src, args.out / rel, args.max_side, args.max_pixels)
            mapping[raw] = rel
        except Exception as exc:
            failed.append((str(src), repr(exc)))
            mapping[raw] = ""

    def repl(match: re.Match[str]) -> str:
        raw = match.group(1)
        rel = mapping.get(raw, "")
        return f'src="{rel}"' if rel else 'src=""'

    html = re.sub(r'src="file://([^"]+)"', repl, html)
    out_html = args.out / "review_static.html"
    out_html.write_text(html, encoding="utf-8")
    (args.out / "copy_failures.txt").write_text(
        "\n".join(f"{path}\t{err}" for path, err in failed),
        encoding="utf-8",
    )
    print(f"wrote {out_html}", flush=True)
    print(f"copied {len(paths) - len(failed)}/{len(paths)} images", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
