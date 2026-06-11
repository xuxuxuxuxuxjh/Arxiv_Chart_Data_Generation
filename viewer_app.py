#!/usr/bin/env python3
import argparse
import json
import mimetypes
import os
import re
import shutil
import subprocess
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path, PurePosixPath
from urllib.parse import parse_qs, quote, unquote, urlparse


DATA_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = DATA_ROOT / "viewer_static"
CACHE_ROOT = DATA_ROOT / ".viewer_cache"
DEFAULT_PAPER = DATA_ROOT / "arXiv_src_2107_050.tar" / "2107.08430"


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_data_path(value: str) -> Path:
    if not value:
        return DEFAULT_PAPER.resolve()
    value = unquote(value)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = DATA_ROOT / path
    resolved = path.resolve()
    if not is_relative_to(resolved, DATA_ROOT.resolve()):
        raise ValueError(f"path must stay under {DATA_ROOT}")
    return resolved


def file_url(path: Path) -> str:
    return "/file?path=" + quote(str(path.resolve()), safe="")


def read_text(path: Path) -> str:
    for enc in ("utf-8", "latin1", "gb18030", "gbk"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="ignore")


def strip_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        out = []
        escaped = False
        for ch in line:
            if ch == "\\":
                escaped = not escaped
                out.append(ch)
            elif ch == "%" and not escaped:
                break
            else:
                escaped = False
                out.append(ch)
        lines.append("".join(out))
    return "\n".join(lines)


def command_arg(text: str, command: str) -> str:
    needle = "\\" + command
    start = text.find(needle)
    if start < 0:
        return ""
    brace = text.find("{", start + len(needle))
    if brace < 0:
        return ""
    depth = 0
    for idx in range(brace, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[brace + 1:idx]
    return ""


def unwrap_simple_commands(text: str) -> str:
    commands = [
        "emph", "textit", "textbf", "texttt", "textrm", "textsc", "textcolor",
        "small", "footnotesize", "scriptsize", "large", "Large", "url",
        "href", "textsuperscript", "underline",
    ]
    changed = True
    while changed:
        changed = False
        for cmd in commands:
            pattern = re.compile(r"\\" + cmd + r"(?:\[[^\]]*\])?\{([^{}]*)\}")
            text, n = pattern.subn(r"\1", text)
            changed = changed or n > 0
    return text


def latex_to_text(text: str) -> str:
    text = strip_comments(text)
    text = re.sub(r"\\(begin|end)\{[^}]+\}", " ", text)
    text = re.sub(r"\\(section|subsection|subsubsection|paragraph|subparagraph)\*?\{([^{}]*)\}", r"\2", text)
    text = re.sub(r"\\caption(?:\[[^\]]*\])?\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1", text)
    text = re.sub(r"\\label\{[^}]+\}", " ", text)
    text = re.sub(r"\\cite[a-zA-Z]*\{[^}]+\}", " [cite] ", text)
    text = re.sub(r"\\(Cref|cref|autoref|ref|eqref)\{([^}]+)\}", r"\2", text)
    text = re.sub(r"\\footnote\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", " ", text)
    text = unwrap_simple_commands(text)
    replacements = {
        r"\%": "%",
        r"\&": "&",
        r"\_": "_",
        r"\#": "#",
        r"\$": "$",
        r"\{": "{",
        r"\}": "}",
        r"~": " ",
        r"\,": "",
        r"\;": " ",
        r"\:": " ",
        r"\!": "",
        r"---": "-",
        r"--": "-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", text)
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def latex_to_readable(text: str) -> str:
    readable = latex_to_text(text)
    if readable:
        return readable
    compact = strip_comments(text)
    compact = re.sub(r"\s+", " ", compact).strip()
    return compact


def normalize_latex(text: str) -> str:
    text = strip_comments(text)
    text = text.replace("\\ ", "")
    return re.sub(r"\s+", "", text)


def split_blocks_from_text(text: str, start_index: int) -> tuple[list[dict], int]:
    blocks = []
    block_index = start_index
    text = remove_document_control_commands(text)

    heading_re = re.compile(r"\\(section|subsection|subsubsection|paragraph)\*?(?:\[[^\]]*\])?\{([^{}]+)\}")
    pos = 0
    chunks = []
    for match in heading_re.finditer(text):
        before = text[pos:match.start()]
        if before.strip():
            chunks.append(("text", before))
        chunks.append((match.group(1), match.group(2)))
        pos = match.end()
    tail = text[pos:]
    if tail.strip():
        chunks.append(("text", tail))

    for kind, chunk in chunks:
        if kind != "text":
            level = {"section": 1, "subsection": 2, "subsubsection": 3, "paragraph": 4}.get(kind, 2)
            blocks.append({
                "id": f"b{block_index}",
                "type": "heading",
                "level": level,
                "raw": chunk,
                "text": latex_to_text(chunk),
            })
            block_index += 1
            continue

        paragraphs = re.split(r"\n\s*\n", chunk)
        for para in paragraphs:
            raw = para.strip()
            if not raw:
                continue
            if re.fullmatch(r"\\(setcounter|vspace|hspace|pagestyle|thispagestyle|newpage|clearpage).*", raw, re.S):
                continue
            text_value = latex_to_readable(raw)
            if not text_value or len(text_value) < 2:
                continue
            blocks.append({
                "id": f"b{block_index}",
                "type": "paragraph",
                "raw": raw,
                "text": text_value,
            })
            block_index += 1
    return blocks, block_index


def remove_balanced_command(text: str, command: str) -> str:
    needle = "\\" + command
    pos = 0
    out = []
    while True:
        start = text.find(needle, pos)
        if start < 0:
            out.append(text[pos:])
            break
        out.append(text[pos:start])
        brace = text.find("{", start + len(needle))
        if brace < 0:
            line_end = text.find("\n", start)
            pos = len(text) if line_end < 0 else line_end + 1
            continue
        depth = 0
        end = brace
        while end < len(text):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    end += 1
                    break
            end += 1
        pos = end
    return "".join(out)


def remove_document_control_commands(text: str) -> str:
    for command in ("title", "author"):
        text = remove_balanced_command(text, command)
    text = re.sub(r"\\maketitle", "\n\n", text)
    text = re.sub(r"\\renewcommand\\twocolumn\[1\]\[\]\{#1\}", "\n\n", text)
    text = re.sub(r"\\twocolumn\s*\[\s*\{", "\n\n", text)
    text = re.sub(r"^\s*\}\s*\]\s*$", "\n\n", text, flags=re.M)
    text = re.sub(r"\\blfootnote\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", r"\1", text)
    return text


def extract_caption_from_env(raw: str) -> str:
    match = re.search(r"\\caption(?:\[[^\]]*\])?\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", raw, re.S)
    return latex_to_text(match.group(1)) if match else ""


def split_latex_cells(row: str) -> list[str]:
    cells = []
    buf = []
    depth = 0
    escaped = False
    for ch in row:
        if ch == "\\":
            escaped = not escaped
            buf.append(ch)
            continue
        if ch == "{" and not escaped:
            depth += 1
        elif ch == "}" and not escaped:
            depth = max(0, depth - 1)
        if ch == "&" and depth == 0 and not escaped:
            cells.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
        escaped = False
    if buf or row:
        cells.append("".join(buf).strip())
    return cells


def clean_table_cell(cell: str) -> tuple[str, int]:
    span = 1
    multi_col = re.search(r"\\multicolumn\{(\d+)\}\{[^{}]*\}\{(.+)\}\s*$", cell, re.S)
    if multi_col:
        span = max(1, int(multi_col.group(1)))
        cell = multi_col.group(2)
    multi_row = re.search(r"\\multirow\{[^{}]*\}\{[^{}]*\}\{(.+)\}\s*$", cell, re.S)
    if multi_row:
        cell = multi_row.group(1)
    cell = re.sub(r"\\color\{[^{}]*\}", "", cell)
    cell = re.sub(r"\\cellcolor\{[^{}]*\}", "", cell)
    cell = latex_to_text(cell)
    return cell.strip(), span


def consume_balanced_group(text: str, start: int, open_char: str, close_char: str) -> int:
    if start >= len(text) or text[start] != open_char:
        return start
    depth = 0
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "\\":
            escaped = not escaped
            continue
        if ch == open_char and not escaped:
            depth += 1
        elif ch == close_char and not escaped:
            depth -= 1
            if depth == 0:
                return idx + 1
        escaped = False
    return start


def extract_tabular_body(raw: str) -> str:
    match = re.search(r"\\begin\{(tabular\*?|tabularx|array)\}", raw)
    if not match:
        return ""
    env_name = match.group(1)
    pos = match.end()

    while pos < len(raw) and raw[pos].isspace():
        pos += 1
    if pos < len(raw) and raw[pos] == "[":
        pos = consume_balanced_group(raw, pos, "[", "]")

    required_groups = 2 if env_name in {"tabular*", "tabularx"} else 1
    for _ in range(required_groups):
        while pos < len(raw) and raw[pos].isspace():
            pos += 1
        if pos < len(raw) and raw[pos] == "{":
            pos = consume_balanced_group(raw, pos, "{", "}")

    body_start = pos
    end_match = re.search(r"\\end\{" + re.escape(env_name) + r"\}", raw[body_start:], re.S)
    if not end_match:
        return ""
    return raw[body_start:body_start + end_match.start()]


def parse_latex_table(raw: str) -> list[list[str]]:
    body = extract_tabular_body(raw)
    if not body:
        return []

    body = re.sub(r"\\(?:toprule|midrule|bottomrule|hline)\b", "\n", body)
    body = re.sub(r"\\(?:cmidrule|cline)(?:\([^)]*\))?\{[^{}]*\}", "\n", body)
    body = re.sub(r"\\addlinespace(?:\[[^\]]*\])?", "\n", body)
    raw_rows = re.split(r"(?<!\\)\\\\(?:\s*\[[^\]]*\])?", body)
    rows = []
    for raw_row in raw_rows:
        row = raw_row.strip()
        if not row:
            continue
        row = re.sub(r"^\s*&+", "", row)
        cells = []
        for cell in split_latex_cells(row):
            cleaned, span = clean_table_cell(cell)
            cells.append(cleaned)
            cells.extend([""] * (span - 1))
        if any(cell for cell in cells):
            rows.append(cells)

    if not rows:
        return []
    max_cols = max(len(row) for row in rows)
    return [row + [""] * (max_cols - len(row)) for row in rows]


def build_document(tex: str, figures: list[dict]) -> dict:
    clean_tex = strip_comments(tex)
    figure_groups = {}
    for fig in figures:
        figure_groups.setdefault(str(fig.get("figure_index", "")), []).append(fig)

    title = latex_to_text(command_arg(clean_tex, "title"))
    author = latex_to_text(command_arg(clean_tex, "author"))
    content = clean_tex
    begin = content.find(r"\begin{document}")
    end = content.rfind(r"\end{document}")
    if begin >= 0:
        content = content[begin + len(r"\begin{document}"):]
    if end >= 0:
        content = content[:end]

    env_re = re.compile(r"\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}", re.S)
    blocks = []
    block_index = 0

    if title:
        blocks.append({"id": f"b{block_index}", "type": "title", "raw": title, "text": title})
        block_index += 1
    if author:
        blocks.append({"id": f"b{block_index}", "type": "author", "raw": author, "text": author})
        block_index += 1

    last = 0
    figure_order = 0
    for match in env_re.finditer(content):
        before = content[last:match.start()]
        new_blocks, block_index = split_blocks_from_text(before, block_index)
        blocks.extend(new_blocks)

        raw = match.group(0).strip()
        env_name = match.group(1)
        if env_name.startswith("figure"):
            figure_order += 1
            group = figure_groups.get(str(figure_order), [])
            caption = group[0]["caption_text"] if group else extract_caption_from_env(raw)
            blocks.append({
                "id": f"b{block_index}",
                "type": "figure",
                "raw": raw,
                "text": caption,
                "figure_index": figure_order,
                "images": [{"src": item["image_url"], "name": item["image_name"]} for item in group],
                "caption": caption,
            })
        else:
            blocks.append({
                "id": f"b{block_index}",
                "type": "table",
                "raw": raw,
                "text": latex_to_text(raw)[:1200],
                "caption": extract_caption_from_env(raw),
                "table_rows": parse_latex_table(raw),
            })
        block_index += 1
        last = match.end()

    tail_blocks, block_index = split_blocks_from_text(content[last:], block_index)
    blocks.extend(tail_blocks)

    norm_by_block = {block["id"]: normalize_latex(block.get("raw", "")) for block in blocks}
    for fig in figures:
        ref_ids = []
        for ref in fig.get("reference_paragraphs_latex", []):
            ref_norm = normalize_latex(ref)
            if not ref_norm:
                continue
            for block_id, block_norm in norm_by_block.items():
                if ref_norm in block_norm or block_norm in ref_norm:
                    ref_ids.append(block_id)
        fig["reference_block_ids"] = sorted(set(ref_ids), key=ref_ids.index)
        fig["figure_block_id"] = next(
            (block["id"] for block in blocks if block.get("type") == "figure" and block.get("figure_index") == fig.get("figure_index")),
            "",
        )

    pages = paginate_blocks(blocks)
    return {"pages": pages, "blocks": blocks, "stats": build_document_stats(tex, blocks)}


def build_document_stats(tex: str, blocks: list[dict]) -> dict:
    raw_chars = sum(len(block.get("raw", "")) for block in blocks)
    text_chars = sum(len(block.get("text", "")) for block in blocks)
    return {
        "total_tex_chars": len(tex),
        "block_raw_chars": raw_chars,
        "block_text_chars": text_chars,
        "num_blocks": len(blocks),
    }


def block_cost(block: dict) -> int:
    kind = block.get("type")
    text_len = len(block.get("text", ""))
    if kind == "title":
        return 150
    if kind == "author":
        return 90
    if kind == "heading":
        return 70
    if kind == "figure":
        return 360 + 120 * max(0, len(block.get("images", [])) - 1)
    if kind == "table":
        rows = block.get("table_rows") or []
        return 180 + min(700, max(len(rows), 3) * 42)
    return max(56, 24 + text_len // 2)


def paginate_blocks(blocks: list[dict]) -> list[dict]:
    pages = []
    current = []
    used = 0
    limit = 2200
    for block in blocks:
        cost = block_cost(block)
        if current and used + cost > limit:
            pages.append({"number": len(pages) + 1, "blocks": current})
            current = []
            used = 0
        current.append(block)
        used += cost
    if current:
        pages.append({"number": len(pages) + 1, "blocks": current})
    return pages


def minimal_sty_content(name: str) -> str:
    basename = re.sub(r"[^A-Za-z0-9_:-]", "", name)
    return rf"""
\NeedsTeXFormat{{LaTeX2e}}
\ProvidesPackage{{{basename}}}[local viewer fallback]
\providecommand{{\cvprfinalcopy}}{{}}
\providecommand{{\cvprPaperID}}[1]{{}}
\providecommand{{\httilde}}{{\textasciitilde}}
\providecommand{{\Checkmark}}{{\checkmark}}
\providecommand{{\XSolidBrush}}{{\texttimes}}
"""


def minimal_cls_content(name: str) -> str:
    basename = re.sub(r"[^A-Za-z0-9_:-]", "", name)
    return rf"""
\NeedsTeXFormat{{LaTeX2e}}
\ProvidesClass{{{basename}}}[local viewer fallback]
\LoadClass{{article}}
\RequirePackage{{graphicx}}
\RequirePackage{{amsmath}}
\RequirePackage{{amssymb}}
\providecommand{{\address}}[1]{{\par\smallskip{{\small #1}}\par}}
\providecommand{{\bodymatter}}{{}}
\providecommand{{\refcite}}[1]{{\cite{{#1}}}}
"""


def patch_tex_for_preview(tex: str, cache_dir: Path, figures: list[dict]) -> str:
    patched = tex
    for fig in figures:
        original = fig.get("original_graphics_path") or ""
        image_path = fig.get("image_path")
        if not original or not image_path:
            continue
        source = Path(image_path)
        if not source.exists():
            continue
        normalized_original = original.replace("\\", "/").lstrip("./")
        posix = PurePosixPath(normalized_original)
        target = cache_dir / posix.with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
        replacement = str(posix.with_suffix(".png"))
        candidates = {
            original,
            original.lstrip("./"),
            "./" + original.lstrip("./"),
            str(posix),
        }
        for candidate in sorted(candidates, key=len, reverse=True):
            patched = patched.replace(candidate, replacement)
    return patched


def ensure_pdf_preview(paper_dir: Path, paper_id: str, tex: str, figures: list[dict]) -> Path | None:
    if shutil.which("pdflatex") is None:
        return None
    cache_dir = CACHE_ROOT / re.sub(r"[^A-Za-z0-9_.-]+", "_", str(paper_dir.relative_to(DATA_ROOT)))
    pdf_path = cache_dir / "paper.pdf"
    tex_path = cache_dir / "paper.tex"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        return pdf_path

    patched = patch_tex_for_preview(tex, cache_dir, figures)
    tex_path.write_text(patched, encoding="utf-8")
    (cache_dir / "cvpr.sty").write_text(minimal_sty_content("cvpr"), encoding="utf-8")

    missing_re = re.compile(r"File `([^']+\.(?:sty|cls))' not found")
    last_output = ""
    for _ in range(10):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "paper.tex"],
            cwd=cache_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=90,
        )
        last_output = proc.stdout
        missing = missing_re.search(last_output)
        if missing:
            missing_name = missing.group(1)
            missing_path = cache_dir / missing_name
            if missing_name.endswith(".cls"):
                missing_path.write_text(minimal_cls_content(missing_name[:-4]), encoding="utf-8")
            else:
                missing_path.write_text(minimal_sty_content(missing_name[:-4]), encoding="utf-8")
            continue
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "paper.tex"],
                cwd=cache_dir,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=90,
            )
            return pdf_path
        break

    (cache_dir / "compile.log").write_text(last_output, encoding="utf-8", errors="ignore")
    return pdf_path if pdf_path.exists() and pdf_path.stat().st_size > 0 else None


def load_figures(figs_dir: Path) -> list[dict]:
    figures = []
    for json_path in sorted(figs_dir.glob("fig_*.json")):
        try:
            data = json.loads(read_text(json_path))
        except Exception:
            continue
        image_rel = data.get("output_image_rel_to_output_dir") or (json_path.stem + ".png")
        image_path = figs_dir / image_rel
        if not image_path.exists():
            candidate = json_path.with_suffix(".png")
            image_path = candidate if candidate.exists() else image_path
        caption = data.get("caption_latex") or ""
        refs = data.get("reference_paragraphs_latex") or []
        figures.append({
            "id": json_path.stem,
            "paper_id": data.get("paper_id", ""),
            "figure_index": data.get("figure_index"),
            "image_index_in_figure": data.get("image_index_in_figure"),
            "figure_env": data.get("figure_env", ""),
            "figure_line_start": data.get("figure_line_start"),
            "figure_line_end": data.get("figure_line_end"),
            "figure_tex": data.get("figure_tex", ""),
            "labels": data.get("labels", []),
            "caption_latex": caption,
            "caption_text": latex_to_text(caption),
            "reference_paragraphs_latex": refs,
            "reference_paragraphs_text": [latex_to_text(ref) for ref in refs],
            "graphics_command": data.get("graphics_command", ""),
            "includegraphics_options": data.get("includegraphics_options", ""),
            "original_graphics_path": data.get("original_graphics_path", ""),
            "resolved_source_path_rel_to_paper": data.get("resolved_source_path_rel_to_paper", ""),
            "image_name": image_path.name,
            "image_path": str(image_path.resolve()) if image_path.exists() else "",
            "image_url": file_url(image_path) if image_path.exists() else "",
            "json_name": json_path.name,
        })
    return figures


def build_paper_payload(paper_dir: Path) -> dict:
    if not paper_dir.exists() or not paper_dir.is_dir():
        raise FileNotFoundError(f"paper directory not found: {paper_dir}")
    paper_id = paper_dir.name
    total_tex_path = paper_dir / f"{paper_id}_total_tex" / f"{paper_id}_total.tex"
    figs_dir = paper_dir / f"{paper_id}_extracted_figs"
    if not total_tex_path.exists():
        raise FileNotFoundError(f"total tex not found: {total_tex_path}")
    if not figs_dir.exists():
        raise FileNotFoundError(f"extracted figs directory not found: {figs_dir}")

    tex = read_text(total_tex_path)
    figures = load_figures(figs_dir)
    document = build_document(tex, figures)
    pdf_path = ensure_pdf_preview(paper_dir, paper_id, tex, figures)
    summary_path = figs_dir / "extraction_summary.json"
    summary = {}
    if summary_path.exists():
        try:
            raw_summary = json.loads(read_text(summary_path))
            summary = {
                "num_figures": raw_summary.get("num_figures"),
                "num_success_images": raw_summary.get("num_success_images"),
                "total_tex_kind": raw_summary.get("total_tex_kind"),
                "main_tex_rel_to_paper": raw_summary.get("main_tex_rel_to_paper"),
            }
        except Exception:
            summary = {}

    return {
        "paper_id": paper_id,
        "paper_dir": str(paper_dir),
        "total_tex_path": str(total_tex_path),
        "source_tex": tex,
        "figs_dir": str(figs_dir),
        "pdf_url": file_url(pdf_path) if pdf_path else "",
        "figures": figures,
        "document": document,
        "summary": summary,
    }


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "ArxivFigureViewer/0.1"

    def send_bytes(self, body: bytes, content_type: str, status: int = 200, cache_control: str = "no-store") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: dict, status: int = 200) -> None:
        self.send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                index = STATIC_ROOT / "index.html"
                self.send_bytes(index.read_bytes(), "text/html; charset=utf-8")
            elif parsed.path.startswith("/static/"):
                rel = parsed.path.removeprefix("/static/").lstrip("/")
                target = (STATIC_ROOT / rel).resolve()
                if not is_relative_to(target, STATIC_ROOT.resolve()) or not target.exists():
                    self.send_error(404)
                    return
                content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                self.send_bytes(target.read_bytes(), content_type)
            elif parsed.path == "/file":
                query = parse_qs(parsed.query)
                target = resolve_data_path(query.get("path", [""])[0])
                if not target.exists() or not target.is_file():
                    self.send_error(404)
                    return
                content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
                self.send_bytes(target.read_bytes(), content_type, cache_control="public, max-age=86400")
            elif parsed.path == "/api/paper":
                query = parse_qs(parsed.query)
                paper_dir = resolve_data_path(query.get("path", [""])[0])
                payload = build_paper_payload(paper_dir)
                self.send_json(payload)
            else:
                self.send_error(404)
        except Exception as exc:
            self.send_json({"error": str(exc), "type": exc.__class__.__name__}, 500)

    def log_message(self, fmt: str, *args) -> None:
        print("%s - %s" % (self.address_string(), fmt % args))


def main() -> None:
    parser = argparse.ArgumentParser(description="Local arXiv figure/paper viewer")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    CACHE_ROOT.mkdir(exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    print(f"Serving arXiv viewer on http://{args.host}:{args.port}")
    print(f"Default paper: {DEFAULT_PAPER}")
    server.serve_forever()


if __name__ == "__main__":
    main()
