#!/usr/bin/env python3
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find merged figure directories under an arxiv extraction root."
    )
    parser.add_argument(
        "--src",
        default="/mnt/lvhaoran-jfs/cyy/arxiv_fig_extract_0608_addmerge",
        help="Root directory to scan.",
    )
    parser.add_argument(
        "--out",
        default="/home/i-xujiahao/arxiv_data/merged_files_path.json",
        help="JSON output path.",
    )
    parser.add_argument(
        "--pattern",
        default="*_extracted_figs_merged",
        help="Directory name glob to match.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=4,
        help="Maximum directory depth below src to scan. Default matches root/arxiv_xxx/tar/paper/merged.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress while scanning top-level arxiv groups.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of top-level arxiv groups to scan in parallel.",
    )
    return parser.parse_args()


def list_dirs(path: Path) -> list[Path]:
    try:
        entries = os.scandir(path)
    except OSError as exc:
        print(f"skip unreadable directory: {path} ({exc})")
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


def scan_group(group: Path, pattern: str) -> list[str]:
    matches: list[str] = []
    tar_dirs = [path for path in list_dirs(group) if path.name.startswith("arXiv_src_")]
    for tar_dir in tar_dirs:
        for paper_dir in list_dirs(tar_dir):
            for child in list_dirs(paper_dir):
                if child.match(pattern):
                    matches.append(str(child))
    return sorted(matches)


def scan_arxiv_layout(src: Path, pattern: str, verbose: bool, workers: int) -> list[str]:
    matches: list[str] = []
    groups = [path for path in list_dirs(src) if path.name.startswith("arxiv_")]

    if workers <= 1:
        for group_index, group in enumerate(groups, 1):
            group_matches = scan_group(group, pattern)
            matches.extend(group_matches)
            if verbose:
                print(
                    f"[{group_index}/{len(groups)}] {group}: +{len(group_matches)}, total={len(matches)}",
                    flush=True,
                )
        return sorted(matches)

    completed = 0
    max_workers = min(workers, len(groups)) if groups else 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_group = {executor.submit(scan_group, group, pattern): group for group in groups}
        for future in as_completed(future_to_group):
            group = future_to_group[future]
            group_matches = future.result()
            matches.extend(group_matches)
            completed += 1
            if verbose:
                print(
                    f"[{completed}/{len(groups)}] {group}: +{len(group_matches)}, total={len(matches)}",
                    flush=True,
                )

    return sorted(matches)


def scan_dirs_by_depth(src: Path, pattern: str, max_depth: int) -> list[str]:
    matches: list[str] = []
    stack: list[tuple[Path, int]] = [(src, 0)]

    while stack:
        current, depth = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError as exc:
            print(f"skip unreadable directory: {current} ({exc})")
            continue

        for entry in entries:
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                continue
            if not is_dir:
                continue

            path = Path(entry.path)
            next_depth = depth + 1
            if path.match(pattern):
                matches.append(str(path))
                continue
            if next_depth < max_depth:
                stack.append((path, next_depth))

    return sorted(matches)


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    out = Path(args.out)

    if not src.is_dir():
        raise SystemExit(f"source directory not found: {src}")

    paths = scan_arxiv_layout(src, args.pattern, args.verbose, args.workers)
    if not paths:
        print("no matches found with arxiv layout scan; falling back to depth scan")
        paths = scan_dirs_by_depth(src, args.pattern, args.max_depth)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(paths, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"source: {src}")
    print(f"output: {out}")
    print(f"count: {len(paths)}")
    if paths:
        print(f"first: {paths[0]}")
        print(f"last: {paths[-1]}")


if __name__ == "__main__":
    main()
