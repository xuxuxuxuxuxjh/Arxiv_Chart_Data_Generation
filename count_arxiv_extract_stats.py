#!/usr/bin/env python3
import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".svg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count papers and images in arxiv figure extraction output."
    )
    parser.add_argument(
        "--src",
        default="/mnt/lvhaoran-jfs/cyy/arxiv_fig_extract_0608_addmerge",
        help="Root directory to scan.",
    )
    parser.add_argument(
        "--out",
        default="/home/i-xujiahao/arxiv_data/arxiv_fig_extract_stats.json",
        help="JSON output path for detailed stats.",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def list_dirs(path: Path) -> list[Path]:
    try:
        entries = os.scandir(path)
    except OSError as exc:
        print(f"skip unreadable directory: {path} ({exc})", flush=True)
        return []

    dirs = []
    with entries:
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    dirs.append(Path(entry.path))
            except OSError:
                continue
    return sorted(dirs)


def count_images_in_dir(path: Path) -> tuple[int, Counter[str]]:
    image_count = 0
    suffix_counts: Counter[str] = Counter()

    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in files:
            suffix = Path(name).suffix.lower()
            if suffix in IMAGE_SUFFIXES:
                image_count += 1
                suffix_counts[suffix] += 1

    return image_count, suffix_counts


def scan_group(group: Path) -> dict:
    stats = {
        "group": str(group),
        "paper_count": 0,
        "papers_with_extracted": 0,
        "papers_with_merged": 0,
        "extracted_image_count": 0,
        "merged_image_count": 0,
        "other_image_count": 0,
        "image_suffix_counts": Counter(),
    }

    tar_dirs = [path for path in list_dirs(group) if path.name.startswith("arXiv_src_")]
    for tar_dir in tar_dirs:
        for paper_dir in list_dirs(tar_dir):
            stats["paper_count"] += 1
            has_extracted = False
            has_merged = False
            expected_extracted = f"{paper_dir.name}_extracted_figs"
            expected_merged = f"{paper_dir.name}_extracted_figs_merged"

            for child in list_dirs(paper_dir):
                if child.name == expected_extracted:
                    has_extracted = True
                    count, suffix_counts = count_images_in_dir(child)
                    stats["extracted_image_count"] += count
                    stats["image_suffix_counts"].update(suffix_counts)
                elif child.name == expected_merged:
                    has_merged = True
                    count, suffix_counts = count_images_in_dir(child)
                    stats["merged_image_count"] += count
                    stats["image_suffix_counts"].update(suffix_counts)
                elif child.name.endswith("_extracted_figs") or child.name.endswith("_extracted_figs_merged"):
                    count, suffix_counts = count_images_in_dir(child)
                    stats["other_image_count"] += count
                    stats["image_suffix_counts"].update(suffix_counts)

            if has_extracted:
                stats["papers_with_extracted"] += 1
            if has_merged:
                stats["papers_with_merged"] += 1

    return stats


def merge_stats(group_stats: list[dict]) -> dict:
    merged = {
        "paper_count": 0,
        "papers_with_extracted": 0,
        "papers_with_merged": 0,
        "extracted_image_count": 0,
        "merged_image_count": 0,
        "other_image_count": 0,
        "image_suffix_counts": Counter(),
        "groups": [],
    }

    for stats in group_stats:
        merged["groups"].append(
            {
                "group": stats["group"],
                "paper_count": stats["paper_count"],
                "papers_with_extracted": stats["papers_with_extracted"],
                "papers_with_merged": stats["papers_with_merged"],
                "extracted_image_count": stats["extracted_image_count"],
                "merged_image_count": stats["merged_image_count"],
                "other_image_count": stats["other_image_count"],
            }
        )
        for key in (
            "paper_count",
            "papers_with_extracted",
            "papers_with_merged",
            "extracted_image_count",
            "merged_image_count",
            "other_image_count",
        ):
            merged[key] += stats[key]
        merged["image_suffix_counts"].update(stats["image_suffix_counts"])

    merged["total_image_count"] = (
        merged["extracted_image_count"]
        + merged["merged_image_count"]
        + merged["other_image_count"]
    )
    merged["image_suffix_counts"] = dict(sorted(merged["image_suffix_counts"].items()))
    merged["groups"] = sorted(merged["groups"], key=lambda item: item["group"])
    return merged


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    out = Path(args.out)

    if not src.is_dir():
        raise SystemExit(f"source directory not found: {src}")

    groups = [path for path in list_dirs(src) if path.name.startswith("arxiv_")]
    if not groups:
        raise SystemExit(f"no arxiv_* groups found under {src}")

    group_stats = []
    max_workers = min(args.workers, len(groups))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(scan_group, group): group for group in groups}
        for index, future in enumerate(as_completed(futures), 1):
            stats = future.result()
            group_stats.append(stats)
            if args.verbose:
                total = stats["extracted_image_count"] + stats["merged_image_count"] + stats["other_image_count"]
                print(
                    f"[{index}/{len(groups)}] {stats['group']}: "
                    f"papers={stats['paper_count']}, images={total}",
                    flush=True,
                )

    merged = merge_stats(group_stats)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"source: {src}")
    print(f"output: {out}")
    print(f"paper_count: {merged['paper_count']}")
    print(f"papers_with_extracted: {merged['papers_with_extracted']}")
    print(f"papers_with_merged: {merged['papers_with_merged']}")
    print(f"extracted_image_count: {merged['extracted_image_count']}")
    print(f"merged_image_count: {merged['merged_image_count']}")
    print(f"other_image_count: {merged['other_image_count']}")
    print(f"total_image_count: {merged['total_image_count']}")
    print(f"image_suffix_counts: {merged['image_suffix_counts']}")


if __name__ == "__main__":
    main()
