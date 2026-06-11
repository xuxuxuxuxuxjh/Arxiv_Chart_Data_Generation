#!/usr/bin/env python3
"""
Merge extracted figure panels back into one figure image.

Usage:
    python merge.py /home/i-xujiahao/arxiv_data/arXiv_src_2107_050.tar
    python merge.py /home/i-xujiahao/arxiv_data --dry-run

The script scans for directories named ``*_extracted_figs``. In each directory,
records with the same ``figure_index`` are treated as panels of one LaTeX
figure. Their corresponding JSON ``figure_tex`` is used to recover the
``\\includegraphics`` order and explicit LaTeX row breaks. Merged images and a
merged JSON record are written under ``/home/i-xujiahao/arxiv_data/img_merged``
while preserving the input path index.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, ImageOps


DATA_ROOT = Path("/home/i-xujiahao/arxiv_data").resolve()
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "img_merged"
SKIP_DIR_NAMES = {"img_merged", ".viewer_cache", "__pycache__"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
LINE_WIDTH_UNITS = ("textwidth", "columnwidth", "linewidth", "hsize")


@dataclass
class IncludeRef:
    path: str
    norm: str
    stem_norm: str
    start: int
    end: int
    options: str


@dataclass
class FigRecord:
    json_path: Path
    image_path: Path
    data: dict[str, Any]
    figure_key: str
    image_index: int
    original_graphics_path: str

    @property
    def stem(self) -> str:
        return self.image_path.stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multi-image extracted figures by figure_tex layout."
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Root directory to scan, e.g. /home/i-xujiahao/arxiv_data/arXiv_src_2107_050.tar",
    )
    parser.add_argument(
        "-o",
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root. Default: {DEFAULT_OUTPUT_ROOT}",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=16,
        help="White gap in pixels between merged panels.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=0,
        help="Outer white padding in pixels.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print merge candidates; do not write files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing merged images/json.",
    )
    return parser.parse_args()


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def normalize_graphics_path(value: str) -> str:
    value = (value or "").strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    value = str(PurePosixPath(value))
    return value.lower()


def strip_known_suffix(value: str) -> str:
    suffix = PurePosixPath(value).suffix.lower()
    if suffix in {".pdf", ".png", ".jpg", ".jpeg", ".eps", ".ps", ".webp", ".bmp", ".tif", ".tiff"}:
        return value[: -len(suffix)]
    return value


def norm_stem(value: str) -> str:
    return strip_known_suffix(normalize_graphics_path(value))


def find_extracted_fig_dirs(root: Path) -> list[Path]:
    root = root.resolve()
    dirs: list[Path] = []
    for path in root.rglob("*_extracted_figs"):
        if not path.is_dir():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        dirs.append(path)
    return sorted(dirs)


def fallback_figure_key(json_path: Path) -> str:
    match = re.match(r"fig_(\d+)_img_\d+_", json_path.stem)
    if match:
        return str(int(match.group(1)))
    return json_path.stem


def find_image_for_record(fig_dir: Path, json_path: Path, data: dict[str, Any]) -> Path | None:
    rel = data.get("output_image_rel_to_output_dir") or ""
    if rel:
        candidate = fig_dir / rel
        if candidate.exists():
            return candidate

    same_stem = json_path.with_suffix(".png")
    if same_stem.exists():
        return same_stem

    for suffix in IMAGE_SUFFIXES:
        candidate = json_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def load_records(fig_dir: Path) -> list[FigRecord]:
    records: list[FigRecord] = []
    for json_path in sorted(fig_dir.glob("fig_*.json")):
        if json_path.name == "extraction_summary.json":
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            data = json.loads(json_path.read_text(encoding="latin1"))
        except Exception as exc:
            print(f"[WARN] skip unreadable json: {json_path} ({exc})")
            continue

        image_path = find_image_for_record(fig_dir, json_path, data)
        if image_path is None:
            print(f"[WARN] no image for json: {json_path}")
            continue

        figure_index = data.get("figure_index")
        figure_key = str(figure_index) if figure_index is not None else fallback_figure_key(json_path)
        records.append(
            FigRecord(
                json_path=json_path,
                image_path=image_path,
                data=data,
                figure_key=figure_key,
                image_index=safe_int(data.get("image_index_in_figure"), 0),
                original_graphics_path=data.get("original_graphics_path") or "",
            )
        )
    return records


def extract_includegraphics(figure_tex: str) -> list[IncludeRef]:
    includes: list[IncludeRef] = []
    pattern = re.compile(r"\\includegraphics\s*(?:\[(?P<opts>[^\]]*)\])?\s*\{(?P<path>[^{}]+)\}", re.S)
    for match in pattern.finditer(figure_tex or ""):
        path = match.group("path").strip()
        includes.append(
            IncludeRef(
                path=path,
                norm=normalize_graphics_path(path),
                stem_norm=norm_stem(path),
                start=match.start(),
                end=match.end(),
                options=match.group("opts") or "",
            )
        )
    return includes


def record_stem_candidates(record: FigRecord) -> set[str]:
    candidates = {
        normalize_graphics_path(record.original_graphics_path),
        norm_stem(record.original_graphics_path),
    }

    output_rel = record.data.get("output_image_rel_to_output_dir") or record.image_path.name
    candidates.add(normalize_graphics_path(output_rel))
    candidates.add(norm_stem(output_rel))

    image_name = record.image_path.name
    candidates.add(normalize_graphics_path(image_name))
    candidates.add(norm_stem(image_name))

    # Extracted files are often named fig_0001_img_01_<original_stem>.png.
    match = re.match(r"fig_\d+_img_\d+_(.+)$", record.image_path.stem)
    if match:
        suffix_stem = match.group(1)
        candidates.add(normalize_graphics_path(suffix_stem))
        candidates.add(norm_stem(suffix_stem))
    return {item for item in candidates if item}


def match_records_to_includes(records: list[FigRecord], includes: list[IncludeRef]) -> list[tuple[FigRecord, IncludeRef | None]]:
    unmatched = set(range(len(includes)))
    matched: list[tuple[FigRecord, IncludeRef | None]] = []

    for record in sorted(records, key=lambda item: item.image_index):
        candidates = record_stem_candidates(record)
        best_idx: int | None = None
        for idx in sorted(unmatched):
            include = includes[idx]
            if include.norm in candidates or include.stem_norm in candidates:
                best_idx = idx
                break
        if best_idx is None:
            matched.append((record, None))
        else:
            unmatched.remove(best_idx)
            matched.append((record, includes[best_idx]))

    if any(include is None for _, include in matched):
        return [(record, None) for record in sorted(records, key=lambda item: item.image_index)]

    return sorted(matched, key=lambda pair: pair[1].start if pair[1] else pair[0].image_index)


def has_explicit_row_break(gap_tex: str) -> bool:
    if not gap_tex:
        return False
    # LaTeX row break command. This intentionally ignores plain blank lines,
    # because subfigure environments often contain blank lines after
    # includegraphics while panels are still arranged horizontally.
    if re.search(r"(?<!\\)\\\\(?![A-Za-z])", gap_tex):
        return True
    if re.search(r"\\(?:newline|linebreak|par|vskip|vspace)\b", gap_tex):
        return True

    # Old-style \subfigure{...} sources often use a blank line between rows.
    # Do not treat blank lines inside a subfigure environment or before a
    # \label as row breaks.
    blank_match = re.search(r"\n\s*\n", gap_tex)
    if blank_match:
        before_blank = gap_tex[: blank_match.start()]
        if not re.search(r"\\(?:label|end\{subfigure\})", before_blank):
            return True
    return False


def option_value(options: str, key: str) -> str | None:
    pattern = re.compile(
        rf"(?:^|,)\s*{re.escape(key)}\s*=\s*(?:\{{(?P<braced>[^{{}}]+)\}}|(?P<plain>[^,\]]+))",
        re.S,
    )
    match = pattern.search(options or "")
    if not match:
        return None
    return (match.group("braced") or match.group("plain") or "").strip()


def parse_line_width_reference(value: str) -> tuple[float, str] | None:
    """Return ``(factor, unit)`` for LaTeX widths tied to line/text width."""
    value = re.sub(r"\s+", "", value or "")
    if not value:
        return None

    unit_pattern = "|".join(re.escape(unit) for unit in LINE_WIDTH_UNITS)
    pattern = re.compile(
        rf"(?P<factor>[+-]?(?:\d+(?:\.\d+)?|\.\d+)?)\*?\\(?P<unit>{unit_pattern})"
    )
    match = pattern.search(value)
    if not match:
        return None

    factor_text = match.group("factor")
    if factor_text in {"", "+", "-"}:
        factor = -1.0 if factor_text == "-" else 1.0
    else:
        try:
            factor = float(factor_text)
        except ValueError:
            return None
    return factor, match.group("unit")


def parse_fraction_of_linewidth(value: str) -> float | None:
    parsed = parse_line_width_reference(value)
    if parsed is None:
        return None
    return parsed[0]


def enclosing_local_width_fraction(figure_tex: str, position: int) -> float | None:
    """Find local minipage/subfigure width that redefines \linewidth."""
    prefix = figure_tex[:position]
    candidates: list[tuple[int, float]] = []
    env_pattern = re.compile(
        r"\\begin\{(?P<env>minipage|subfigure|subtable)\}\s*(?:\[[^\]]*\]\s*)*\{(?P<width>[^{}]+)\}",
        re.S,
    )
    for match in env_pattern.finditer(prefix):
        env_name = match.group("env")
        if re.search(rf"\\end\{{{re.escape(env_name)}\}}", prefix[match.end() :]):
            continue
        parsed = parse_line_width_reference(match.group("width"))
        if parsed is None:
            continue
        width_fraction, _unit = parsed
        candidates.append((match.start(), width_fraction))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def include_width_fraction(include: IncludeRef, figure_tex: str) -> float | None:
    width = option_value(include.options, "width")
    if not width:
        return None
    parsed = parse_line_width_reference(width)
    if parsed is None:
        return None
    width_fraction, unit = parsed
    if unit in {"linewidth", "hsize"}:
        local_width = enclosing_local_width_fraction(figure_tex, include.start)
        if local_width is not None:
            return width_fraction * local_width
    return width_fraction


def should_break_by_width(current_width: float | None, include: IncludeRef | None, figure_tex: str) -> bool:
    if include is None:
        return False
    next_width = include_width_fraction(include, figure_tex)
    if next_width is None:
        return False
    if next_width >= 0.99:
        return True
    if current_width is None:
        return False
    return current_width + next_width > 1.01


def infer_layout(records: list[FigRecord]) -> tuple[list[list[FigRecord]], dict[str, Any]]:
    representative = next((record for record in records if record.data.get("figure_tex")), records[0])
    figure_tex = representative.data.get("figure_tex") or ""
    includes = extract_includegraphics(figure_tex)
    ordered_pairs = match_records_to_includes(records, includes)

    rows: list[list[FigRecord]] = []
    current_row: list[FigRecord] = []
    previous_include: IncludeRef | None = None
    current_width = 0.0
    current_width_known = True

    for record, include in ordered_pairs:
        if include is not None and previous_include is not None:
            gap_tex = figure_tex[previous_include.end : include.start]
            known_width = current_width if current_width_known else None
            if (has_explicit_row_break(gap_tex) or should_break_by_width(known_width, include, figure_tex)) and current_row:
                rows.append(current_row)
                current_row = []
                current_width = 0.0
                current_width_known = True
        current_row.append(record)
        if include is not None:
            width_fraction = include_width_fraction(include, figure_tex)
            if width_fraction is None:
                current_width_known = False
            else:
                current_width += width_fraction
            previous_include = include

    if current_row:
        rows.append(current_row)

    if not rows:
        rows = [sorted(records, key=lambda item: item.image_index)]

    layout_info = {
        "row_count": len(rows),
        "row_lengths": [len(row) for row in rows],
        "source": "figure_tex_includegraphics" if includes else "image_index_fallback",
        "includegraphics_paths": [include.path for include in includes],
        "includegraphics_widths": [include_width_fraction(include, figure_tex) for include in includes],
    }
    return rows, layout_info


def load_panel(path: Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"}:
        background = Image.new("RGB", image.size, "white")
        background.paste(image, mask=image.getchannel("A"))
        image = background
    elif image.mode != "RGB":
        image = image.convert("RGB")
    return image


def merge_row(images: list[Image.Image], gap: int) -> Image.Image:
    if len(images) == 1:
        return images[0]
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), "white")
    x = 0
    for image in images:
        y = (height - image.height) // 2
        canvas.paste(image, (x, y))
        x += image.width + gap
    return canvas


def merge_grid(rows: list[list[FigRecord]], gap: int, padding: int) -> Image.Image:
    row_images: list[Image.Image] = []
    for row in rows:
        panels = [load_panel(record.image_path) for record in row]
        row_images.append(merge_row(panels, gap=gap))

    width = max(image.width for image in row_images)
    height = sum(image.height for image in row_images) + gap * (len(row_images) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for image in row_images:
        x = (width - image.width) // 2
        canvas.paste(image, (x, y))
        y += image.height + gap

    if padding > 0:
        padded = Image.new("RGB", (canvas.width + padding * 2, canvas.height + padding * 2), "white")
        padded.paste(canvas, (padding, padding))
        return padded
    return canvas


def input_relative_output_dir(input_root: Path, fig_dir: Path, output_root: Path) -> Path:
    input_root = input_root.resolve()
    fig_dir = fig_dir.resolve()
    output_root = output_root.resolve()

    if input_root.is_relative_to(DATA_ROOT):
        rel = input_root.relative_to(DATA_ROOT) / fig_dir.relative_to(input_root)
    else:
        rel = Path(input_root.name) / fig_dir.relative_to(input_root)
    return output_root / rel


def merged_file_stem(records: list[FigRecord]) -> str:
    first = records[0]
    idx = safe_int(first.data.get("figure_index"), 0)
    if idx:
        return f"fig_{idx:04d}_merged"
    match = re.match(r"(fig_\d+)_img_", first.json_path.stem)
    if match:
        return f"{match.group(1)}_merged"
    return f"{first.figure_key}_merged"


def build_merged_json(
    records: list[FigRecord],
    rows: list[list[FigRecord]],
    layout_info: dict[str, Any],
    output_image_name: str,
    output_json_name: str,
) -> dict[str, Any]:
    representative = records[0].data.copy()
    representative.update(
        {
            "status": "merged_success",
            "merged_image": output_image_name,
            "merged_json": output_json_name,
            "output_image_rel_to_output_dir": output_image_name,
            "output_json_rel_to_output_dir": output_json_name,
            "image_index_in_figure": None,
            "merged_panel_count": len(records),
            "merge_layout": layout_info,
            "merged_rows": [[record.image_path.name for record in row] for row in rows],
            "merged_from": [
                {
                    "json": record.json_path.name,
                    "image": record.image_path.name,
                    "source_json": record.json_path.name,
                    "source_image": record.image_path.name,
                    "figure_index": record.data.get("figure_index"),
                    "image_index_in_figure": record.data.get("image_index_in_figure"),
                    "original_graphics_path": record.original_graphics_path,
                    "output_image_rel_to_output_dir": record.data.get("output_image_rel_to_output_dir"),
                }
                for record in records
            ],
        }
    )
    return representative


def process_group(
    input_root: Path,
    fig_dir: Path,
    records: list[FigRecord],
    output_root: Path,
    gap: int,
    padding: int,
    dry_run: bool,
    overwrite: bool,
) -> bool:
    records = sorted(records, key=lambda item: item.image_index)
    rows, layout_info = infer_layout(records)
    out_dir = input_relative_output_dir(input_root, fig_dir, output_root)
    stem = merged_file_stem(records)
    out_image = out_dir / f"{stem}.png"
    out_json = out_dir / f"{stem}.json"

    print(
        f"[MERGE] {fig_dir} figure={records[0].figure_key} panels={len(records)} "
        f"layout={layout_info['row_lengths']} -> {out_image}"
    )

    if dry_run:
        return True
    if (out_image.exists() or out_json.exists()) and not overwrite:
        print(f"[SKIP] exists, use --overwrite to replace: {out_image}")
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    merged_image = merge_grid(rows, gap=gap, padding=padding)
    merged_image.save(out_image)

    merged_json = build_merged_json(
        records=records,
        rows=rows,
        layout_info=layout_info,
        output_image_name=out_image.name,
        output_json_name=out_json.name,
    )
    out_json.write_text(json.dumps(merged_json, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def main() -> None:
    args = parse_args()
    input_root = args.root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not input_root.exists() or not input_root.is_dir():
        raise SystemExit(f"Input root does not exist or is not a directory: {input_root}")

    fig_dirs = find_extracted_fig_dirs(input_root)
    print(f"[INFO] input_root={input_root}")
    print(f"[INFO] output_root={output_root}")
    print(f"[INFO] extracted_fig_dirs={len(fig_dirs)}")

    candidate_count = 0
    merged_count = 0
    for fig_dir in fig_dirs:
        records = load_records(fig_dir)
        groups: dict[str, list[FigRecord]] = {}
        for record in records:
            groups.setdefault(record.figure_key, []).append(record)

        for group_records in groups.values():
            if len(group_records) <= 1:
                continue
            candidate_count += 1
            if process_group(
                input_root=input_root,
                fig_dir=fig_dir,
                records=group_records,
                output_root=output_root,
                gap=args.gap,
                padding=args.padding,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            ):
                merged_count += 1

    print(f"[DONE] candidates={candidate_count}, {'would_merge' if args.dry_run else 'merged'}={merged_count}")


if __name__ == "__main__":
    main()
