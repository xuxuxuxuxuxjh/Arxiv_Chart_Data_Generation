#!/usr/bin/env python3
from __future__ import annotations

import base64
import ast
import hashlib
import json
import math
import mimetypes
import os
import random
import re
import string
import time
import unicodedata
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import requests
from PIL import Image


ARXIV_ROOT = Path("/mnt/lvhaoran-jfs/cyy/arxiv_fig_extract_0608_addmerge")
CHARXIV_ROOT = Path("/mnt/stepeval/datasets/VL_datasets/CharXiv/data")
WORK_ROOT = Path("/home/i-xujiahao/arxiv_data/work")

GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_ENDPOINT = (
    "https://models-proxy.stepfun-inc.com/gemini/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
KIMI_MODEL = "kimi-k2.6-aliyun2kimi"
KIMI_BASE_URL = "https://models-proxy.stepfun-inc.com/v1"
KIMI_ENDPOINT = f"{KIMI_BASE_URL}/chat/completions"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
PAPER_ID_RE = re.compile(r"^(20|21|22|23|24|25)[0-9]{2}\.[0-9]+")

POSITIVE_KEYWORDS = [
    "accuracy",
    "loss",
    "score",
    "performance",
    "comparison",
    "baseline",
    "ablation",
    "epoch",
    "step",
    "training",
    "validation",
    "test",
    "metric",
    "benchmark",
    "precision",
    "recall",
    "f1",
    "auc",
    "roc",
    "pr curve",
    "error",
    "rmse",
    "mae",
    "distribution",
    "histogram",
    "cumulative",
    "density",
    "frequency",
    "scatter",
    "correlation",
    "heatmap",
    "confusion matrix",
    "latency",
    "throughput",
    "speed",
    "time",
    "memory",
    "parameter",
    "scaling",
    "redshift",
    "flux",
    "spectrum",
    "mass",
    "temperature",
    "energy",
    "residual",
    "profile",
]

NEGATIVE_KEYWORDS = [
    "architecture",
    "framework",
    "pipeline",
    "overview",
    "workflow",
    "algorithm",
    "pseudocode",
    "screenshot",
    "user interface",
    " ui ",
    "qualitative examples",
    "generated samples",
    "input image",
    "output image",
    "segmentation result",
    "detection example",
    "reconstruction example",
    "network structure",
    "computational graph",
    "model diagram",
]


def ensure_work_dirs() -> None:
    for rel in ("qa", "dense_caption", "logs", "reports", "review"):
        (WORK_ROOT / rel).mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: JSONL record is not an object")
            yield value


def append_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")
            count += 1
    return count


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            f.write("\n")
            count += 1
    return count


def list_dirs(path: Path) -> list[Path]:
    try:
        entries = os.scandir(path)
    except OSError as exc:
        print(f"skip unreadable directory: {path} ({exc})", flush=True)
        return []

    dirs: list[Path] = []
    with entries:
        for entry in entries:
            try:
                if entry.is_dir(follow_symlinks=False):
                    dirs.append(Path(entry.path))
            except OSError:
                continue
    return sorted(dirs)


def iter_files_with_suffix(path: Path, suffixes: set[str]) -> Iterable[Path]:
    for root, dirs, files in os.walk(path):
        dirs.sort()
        for name in sorted(files):
            if Path(name).suffix.lower() in suffixes:
                yield Path(root) / name


def image_info(path: Path) -> tuple[int, int, str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        img = Image.open(path)
    with img:
        width, height = img.size
        fmt = img.format or path.suffix.lstrip(".").upper()
    return width, height, fmt


def normalize_output_image(record: dict[str, Any], json_path: Path) -> str:
    value = record.get("output_image") or record.get("image_path") or ""
    if isinstance(value, str) and value:
        candidate = Path(value)
        if candidate.exists():
            return str(candidate)
        sibling = json_path.with_name(candidate.name)
        if sibling.exists():
            return str(sibling)
    sibling = json_path.with_suffix(".png")
    if sibling.exists():
        return str(sibling)
    for suffix in IMAGE_SUFFIXES:
        sibling = json_path.with_suffix(suffix)
        if sibling.exists():
            return str(sibling)
    return str(Path(value)) if isinstance(value, str) else ""


def infer_month_year_from_path(path: Path) -> tuple[str | None, int | None, str | None]:
    parts = path.parts
    for part in parts:
        if re.fullmatch(r"arxiv_[0-9]{4}", part):
            month = part.replace("arxiv_", "")
            return month, 2000 + int(month[:2]), part
    return None, None, None


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(clean_text(item) for item in value)
    return str(value)


def metadata_keyword_score(record: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    haystack = " ".join(
        [
            clean_text(record.get("caption_latex")),
            clean_text(record.get("labels")),
            clean_text(record.get("figure_tex")),
            clean_text(record.get("image_path")),
            clean_text(record.get("reference_paragraphs_latex"))[:4000],
        ]
    )
    lowered = f" {haystack.lower()} "
    positives = [kw for kw in POSITIVE_KEYWORDS if kw in lowered]
    negatives = [kw.strip() for kw in NEGATIVE_KEYWORDS if kw in lowered]
    score = 2 * len(positives) - 3 * len(negatives)
    return score, {"positive_keywords": positives, "negative_keywords": negatives}


def simple_phash(path: Path, hash_size: int = 8) -> str:
    with Image.open(path) as img:
        img = img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
        pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for pixel in pixels:
        bits = (bits << 1) | int(pixel >= avg)
    return f"{bits:0{hash_size * hash_size // 4}x}"


def hamming_hex(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def image_part(path: Path) -> dict[str, Any]:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    with path.open("rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return {"inlineData": {"mimeType": mime, "data": data}}


def api_key() -> str:
    key = (
        os.environ.get("MODELPROXY_APIKEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("STEPMODEL_API_KEY")
    )
    if key:
        return key
    api_py = Path("/home/i-xujiahao/api.py")
    if api_py.exists():
        text = api_py.read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"OPENAI_API_KEY\s*=\s*['\"]([^'\"]+)['\"]", text)
        if match:
            return match.group(1)
    raise RuntimeError("MODELPROXY_APIKEY or OPENAI_API_KEY is required")


def request_with_retry(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    json_payload: dict[str, Any],
    timeout: int,
    retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                json=json_payload,
                timeout=timeout,
            )
            if response.status_code >= 400:
                message = response.text[:1000]
                if response.status_code in {408, 409, 424, 429, 500, 502, 503, 504}:
                    raise RuntimeError(f"RETRYABLE HTTP {response.status_code}: {message}")
                raise RuntimeError(f"HTTP {response.status_code}: {message}")
            return response.json()
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                break
            text = repr(exc).lower()
            wait = retry_sleep * (2**attempt) + random.random()
            retry_after = re.search(r"try again after (\d+(?:\.\d+)?) seconds", text)
            if retry_after:
                wait = max(wait, float(retry_after.group(1)) + 1.0)
            if "429" in text or "rate limit" in text:
                wait = max(wait, 5.0 + attempt * 3.0 + random.random())
            time.sleep(wait)
    assert last_exc is not None
    raise last_exc


def gemini_generate(
    parts: list[dict[str, Any]],
    *,
    max_output_tokens: int,
    temperature: float,
    top_p: float,
    reasoning_effort: str,
    timeout: int = 180,
    retries: int = 2,
) -> str:
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "temperature": temperature,
            "topP": top_p,
        },
        "extra_kwargs": {"reasoning_effort": reasoning_effort},
    }
    headers = {"Content-Type": "application/json", "x-google-api-key": api_key()}
    data = request_with_retry(
        "POST",
        GEMINI_ENDPOINT,
        headers=headers,
        json_payload=payload,
        timeout=timeout,
        retries=retries,
        retry_sleep=2,
    )
    try:
        candidates = data["candidates"]
        parts_out = candidates[0]["content"]["parts"]
        return "".join(part.get("text", "") for part in parts_out).strip()
    except Exception as exc:
        raise RuntimeError(f"Unexpected Gemini response: {data}") from exc


def kimi_generate(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 250000,
    temperature: float = 1,
    top_p: float = 1,
    top_k: int = -1,
    timeout: int = 300,
    retries: int = 5,
) -> str:
    payload = {
        "model": KIMI_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key()}"}
    data = request_with_retry(
        "POST",
        KIMI_ENDPOINT,
        headers=headers,
        json_payload=payload,
        timeout=timeout,
        retries=retries,
        retry_sleep=2,
    )
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        raise RuntimeError(f"Unexpected Kimi response: {data}") from exc


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.I | re.S)
    if fence:
        stripped = fence.group(1).strip()
    elif stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        snippet = stripped[start : end + 1]
        try:
            value = json.loads(snippet)
        except json.JSONDecodeError:
            try:
                escaped = re.sub(r"\\(?![\"\\/bfnrtu])", r"\\\\", snippet)
                value = json.loads(escaped)
            except json.JSONDecodeError:
                value = ast.literal_eval(snippet)
    if not isinstance(value, dict):
        raise ValueError("Expected JSON object")
    return value


def strip_punctuation(text: str) -> str:
    table = str.maketrans("", "", string.punctuation + "，。！？；：“”‘’、（）【】《》")
    return text.translate(table)


def normalize_answer(value: Any, answer_type: str = "short_text") -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        items = [normalize_answer(item, "short_text") for item in value]
        items = [item for item in items if item]
        if answer_type == "list":
            return "|".join(sorted(items))
        return " ".join(items)
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = strip_punctuation(text)
    text = re.sub(r"\s+", " ", text).strip()
    if answer_type in {"numeric_exact", "numeric_approx", "number_in_chart"}:
        nums = parse_numbers(text)
        if nums:
            return format_float(nums[0])
    if answer_type == "yes_no":
        if text in {"yes", "y", "true"}:
            return "yes"
        if text in {"no", "n", "false"}:
            return "no"
    return text


def parse_numbers(text: str) -> list[float]:
    results: list[float] = []
    for match in re.finditer(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:e[-+]?\d+)?\s*%?", text, re.I):
        token = match.group(0).strip()
        pct = token.endswith("%")
        token = token.rstrip("%").replace(",", "")
        try:
            value = float(token)
        except ValueError:
            continue
        if pct:
            value = value / 100.0
        results.append(value)
    return results


def format_float(value: float) -> str:
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:.8g}"


def compatible_answers(values: list[str], answer_type: str) -> bool:
    if not values:
        return False
    if len(set(values)) == 1:
        return True
    if answer_type == "numeric_approx":
        nums: list[float] = []
        for value in values:
            parsed = parse_numbers(value)
            if not parsed:
                return False
            nums.append(parsed[0])
        scale = max(max(abs(n) for n in nums), 1.0)
        return max(nums) - min(nums) <= max(0.02 * scale, 0.05) or (
            max(nums) - min(nums)
        ) / scale <= 0.05
    return False


def extract_final_answer(text: str) -> str:
    match = re.search(r"final answer\s*:\s*(.+)", text, re.I | re.S)
    if match:
        return match.group(1).strip()
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for line in f if line.strip())


def summarize_jsonl(path: Path, keys: list[str]) -> dict[str, Any]:
    total = 0
    counters: dict[str, Counter[str]] = {key: Counter() for key in keys}
    for record in iter_jsonl(path):
        total += 1
        for key in keys:
            value: Any = record
            for part in key.split("."):
                if isinstance(value, dict):
                    value = value.get(part)
                else:
                    value = None
                    break
            if isinstance(value, list):
                for item in value:
                    counters[key][str(item)] += 1
            else:
                counters[key][str(value)] += 1
    return {
        "total": total,
        "counters": {key: dict(counter.most_common()) for key, counter in counters.items()},
    }


def nested_defaultdict_counter() -> defaultdict[str, Counter[str]]:
    return defaultdict(Counter)


def counter_to_dict(value: Any) -> Any:
    if isinstance(value, Counter):
        return dict(value.most_common())
    if isinstance(value, defaultdict):
        return {key: counter_to_dict(val) for key, val in sorted(value.items())}
    if isinstance(value, dict):
        return {key: counter_to_dict(val) for key, val in sorted(value.items())}
    return value
