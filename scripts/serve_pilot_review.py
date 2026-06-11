#!/usr/bin/env python3
from __future__ import annotations

import argparse
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


DEFAULT_HTML = Path("/home/i-xujiahao/arxiv_data/work/review/pilot_review.html")
ALLOWED_ROOTS = [
    Path("/mnt/lvhaoran-jfs"),
    Path("/mnt/xjh"),
    Path("/home/i-xujiahao/arxiv_data"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve pilot review HTML and images.")
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8899)
    return parser.parse_args()


def is_allowed(path: Path) -> bool:
    try:
        resolved = path.resolve()
    except OSError:
        return False
    for root in ALLOWED_ROOTS:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def rewrite_file_urls(html: str) -> str:
    def repl(match: re.Match[str]) -> str:
        path = unquote(match.group(1))
        return f'src="/file?path={quote(path, safe="")}"'

    return re.sub(r'src="file://([^"]+)"', repl, html)


class Handler(BaseHTTPRequestHandler):
    html_path: Path = DEFAULT_HTML

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)

    def send_text(self, status: int, text: str, content_type: str = "text/plain") -> None:
        data = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/pilot_review.html"}:
            if not self.html_path.exists():
                self.send_text(404, f"missing review HTML: {self.html_path}\n")
                return
            html = self.html_path.read_text(encoding="utf-8", errors="ignore")
            self.send_text(200, rewrite_file_urls(html), "text/html")
            return

        if parsed.path == "/file":
            query = parse_qs(parsed.query)
            values = query.get("path") or []
            if not values:
                self.send_text(400, "missing path\n")
                return
            path = Path(unquote(values[0]))
            if not is_allowed(path) or not path.is_file():
                self.send_text(404, f"not allowed or missing: {path}\n")
                return
            mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_text(404, "not found\n")


def main() -> int:
    args = parse_args()
    Handler.html_path = args.html
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Serving {args.html} at http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
