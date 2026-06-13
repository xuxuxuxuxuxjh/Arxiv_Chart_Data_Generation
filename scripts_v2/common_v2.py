#!/usr/bin/env python3
from __future__ import annotations

import ast
import base64
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
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import requests
from PIL import Image


ARXIV_ROOT = Path(os.environ.get("ARXIV_ROOT", "/mnt/lvhaoran-jfs/cyy/arxiv_fig_extract_0608_addmerge"))
CHARXIV_ROOT = Path(os.environ.get("CHARXIV_ROOT", "/mnt/stepeval/datasets/VL_datasets/CharXiv/data"))
WORK_ROOT = Path(os.environ.get("ARXIV_CHART_WORK", "/home/i-xujiahao/arxiv_data/work"))
EDIT2_ROOT = Path(os.environ.get("ARXIV_CHART_EDIT2", str(WORK_ROOT / "edit2")))

GEMINI_MODEL = os.environ.get("ARXIV_CHART_GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_ENDPOINT = (
    "https://models-proxy.stepfun-inc.com/gemini/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)
KIMI_MODEL = os.environ.get("ARXIV_CHART_KIMI_MODEL", "kimi-k2.6-qianli")
KIMI_BASE_URL = os.environ.get("ARXIV_CHART_KIMI_BASE_URL", "https://models-proxy.stepfun-inc.com/v1")
KIMI_ENDPOINT = f"{KIMI_BASE_URL}/chat/completions"
KIMI_THINKING_BUDGET_TOKENS = int(os.environ.get("ARXIV_CHART_KIMI_THINKING_BUDGET_TOKENS", "2048"))
# Backward-compatible names for the v2 scripts. The transport is now OpenAI-style
# chat/completions with streaming reasoning, not Anthropic messages.
KIMI_MESSAGES_MODEL = KIMI_MODEL
KIMI_MESSAGES_ENDPOINT = KIMI_ENDPOINT

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

CHART_TYPES = (
    "line_chart",
    "bar_chart",
    "scatter_plot",
    "histogram",
    "density_plot",
    "heatmap",
    "confusion_matrix",
    "box_plot",
    "violin_plot",
    "area_chart",
    "roc_curve",
    "pr_curve",
    "calibration_curve",
    "ablation_curve",
    "error_bar_plot",
    "contour_plot",
    "surface_plot",
    "matrix_plot",
    "other_chart",
)

ANSWER_TYPES = (
    "numeric_exact",
    "numeric_approx",
    "integer",
    "choice",
    "ranked_list",
    "trend_label",
    "boolean",
    "short_phrase",
)

TASK_SPECS: dict[str, dict[str, Any]] = {
    "cross_element_comparison": {
        "target_sampling_weight": 18,
        "answer_types": ("choice", "short_phrase", "numeric_approx", "ranked_list"),
        "description": "Compare values, ranks, differences, or relative changes across curves, bars, panels, categories, or conditions.",
    },
    "approximate_value_estimation": {
        "target_sampling_weight": 15,
        "answer_types": ("numeric_approx",),
        "description": "Estimate a visible value from an axis, colorbar, bar, curve, scatter point, or annotated mark with reasonable tolerance.",
    },
    "fine_grained_visual_reading": {
        "target_sampling_weight": 10,
        "answer_types": ("short_phrase", "numeric_exact", "numeric_approx", "choice"),
        "description": "Locate a specific panel, series, x-range, condition, or visual mark before reading a detail; never title-only or axis-only.",
    },
    "multi_element_synthesis": {
        "target_sampling_weight": 15,
        "answer_types": ("short_phrase", "choice", "ranked_list"),
        "description": "Combine legend, axes, panels, series, color, shape, or spatial position to infer the answer.",
    },
    "trend_pattern_analysis": {
        "target_sampling_weight": 15,
        "answer_types": ("trend_label", "short_phrase", "choice"),
        "description": "Determine increasing, decreasing, stable, non-monotonic, crossing, convergence, divergence, saturation, or similar trends.",
    },
    "anomaly_extrema_detection": {
        "target_sampling_weight": 10,
        "answer_types": ("short_phrase", "numeric_approx", "choice"),
        "description": "Find peaks, valleys, outliers, extrema, abrupt changes, or abnormal regions.",
    },
    "complex_calculation": {
        "target_sampling_weight": 10,
        "answer_types": ("numeric_approx", "numeric_exact", "integer"),
        "description": "Compute differences, ratios, totals, averages, ranks, or rates of change from visible chart values.",
    },
    "hypothetical_reasoning": {
        "target_sampling_weight": 7,
        "answer_types": ("numeric_approx", "choice", "short_phrase", "boolean"),
        "description": "Infer or estimate under an explicit hypothetical condition, such as a changed threshold, x-value, or removed method.",
    },
}

TASK_TYPES = tuple(TASK_SPECS)
TASK_SEQUENCE = tuple(
    task_type
    for task_type, spec in TASK_SPECS.items()
    for _ in range(int(spec["target_sampling_weight"]))
)

LOW_VALUE_PATTERNS = (
    re.compile(r"\b(title|caption)\s+(of|for)\s+(the\s+)?(chart|plot|figure)\b", re.I),
    re.compile(r"\bwhat\s+is\s+(the\s+)?(chart|plot|figure)\s+title\b", re.I),
    re.compile(r"\bwhat\s+is\s+(the\s+)?([xy]|horizontal|vertical)[-\s]?axis\s+label\b", re.I),
    re.compile(r"\b(label|name)\s+(of|for)\s+(the\s+)?([xy]|horizontal|vertical)[-\s]?axis\b", re.I),
    re.compile(r"\bwhat\s+does\s+(the\s+)?(legend|colorbar|color\s+bar)\s+(show|represent|indicate)\b", re.I),
    re.compile(r"\bwhat\s+is\s+(the\s+)?(legend|colorbar|color\s+bar)\s+label\b", re.I),
    re.compile(r"\bwhich\s+panel\s+is\s+labeled\s+\(?[a-z]\)?\b", re.I),
    re.compile(r"\bwhat\s+is\s+shown\s+in\s+(the\s+)?(chart|plot|figure)\b", re.I),
)

REASONING_TERMS = re.compile(
    r"\b(compare|larger|smaller|higher|lower|highest|lowest|largest|smallest|"
    r"difference|ratio|twice|half|increase|decrease|trend|peak|valley|outlier|"
    r"approximately|estimate|between|across|relative|if|would|change|rank|most|least)\b",
    re.I,
)


def ensure_edit2_dirs() -> None:
    for rel in ("logs", "reports", "tmp", "review"):
        (EDIT2_ROOT / rel).mkdir(parents=True, exist_ok=True)


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


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as f:
        return sum(1 for line in f if line.strip())


def stable_id(prefix: str, value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:16]
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80].strip("_")
    return f"{prefix}_{safe}_{digest}" if safe else f"{prefix}_{digest}"


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
    raise RuntimeError("MODELPROXY_APIKEY, OPENAI_API_KEY, or STEPMODEL_API_KEY is required")


def request_with_retry(
    method: str,
    url: str,
    *,
    headers: dict[str, str],
    json_payload: dict[str, Any],
    timeout: int,
    retries: int,
    retry_sleep: float = 2.0,
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
    )
    try:
        return "".join(part.get("text", "") for part in data["candidates"][0]["content"]["parts"]).strip()
    except Exception as exc:
        raise RuntimeError(f"Unexpected Gemini response: {data}") from exc


def _model_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_model_text(item) for item in value)
    if isinstance(value, dict):
        for key in ("text", "content", "reasoning_content", "reasoning", "output_text"):
            if key in value:
                text = _model_text(value.get(key))
                if text:
                    return text
        return ""
    return str(value)


def _stream_chat_completion(
    payload: dict[str, Any],
    headers: dict[str, str],
    *,
    timeout: int,
    retries: int,
    retry_sleep: float = 2.0,
) -> str:
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            stream_payload = dict(payload)
            stream_payload["stream"] = True
            content_parts: list[str] = []
            reasoning_parts: list[str] = []
            with requests.post(
                KIMI_ENDPOINT,
                headers=headers,
                json=stream_payload,
                timeout=timeout,
                stream=True,
            ) as response:
                if response.status_code >= 400:
                    message = response.text[:1000]
                    if response.status_code in {408, 409, 424, 429, 500, 502, 503, 504}:
                        raise RuntimeError(f"RETRYABLE HTTP {response.status_code}: {message}")
                    raise RuntimeError(f"HTTP {response.status_code}: {message}")

                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        piece = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    for choice in piece.get("choices") or []:
                        delta = choice.get("delta") or {}
                        content = _model_text(delta.get("content")) or _model_text(choice.get("text"))
                        reasoning = (
                            _model_text(delta.get("reasoning_content"))
                            or _model_text(delta.get("reasoning"))
                            or _model_text(choice.get("reasoning_content"))
                            or _model_text(choice.get("reasoning"))
                            or _model_text(choice.get("reasoning_details"))
                        )
                        if content:
                            content_parts.append(content)
                        if reasoning:
                            reasoning_parts.append(reasoning)

            content = "".join(content_parts).strip()
            reasoning = "".join(reasoning_parts).strip()
            if reasoning:
                return f"<think>\n{reasoning}\n</think>\n\n{content}".strip()
            return content
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


def kimi_generate(
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 8192,
    temperature: float = 1,
    top_p: float = 0.95,
    top_k: int = -1,
    timeout: int = 300,
    retries: int = 2,
    extra_kwargs: dict[str, Any] | None = None,
    stream: bool = True,
) -> str:
    payload = {
        "model": KIMI_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }
    if extra_kwargs:
        payload.update(extra_kwargs)
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key()}"}
    if stream:
        return _stream_chat_completion(
            payload,
            headers,
            timeout=timeout,
            retries=retries,
            retry_sleep=2,
        )

    data = request_with_retry(
        "POST",
        KIMI_ENDPOINT,
        headers=headers,
        json_payload=payload,
        timeout=timeout,
        retries=retries,
    )
    try:
        choice = data["choices"][0]
        message = choice.get("message") or {}
        content = _model_text(message.get("content")) or _model_text(choice.get("text"))
        reasoning = (
            _model_text(message.get("reasoning_content"))
            or _model_text(choice.get("reasoning_content"))
            or _model_text(message.get("reasoning"))
            or _model_text(choice.get("reasoning"))
            or _model_text(message.get("reasoning_details"))
            or _model_text(choice.get("reasoning_details"))
        )
        if reasoning:
            return f"<think>\n{reasoning.strip()}\n</think>\n\n{content.strip()}".strip()
        return content.strip()
    except Exception as exc:
        raise RuntimeError(f"Unexpected Kimi response: {data}") from exc


def content_parts_openai(image_path: Path, text: str, *, cache_dir: Path, max_pixels: int) -> list[dict[str, Any]]:
    resized = resized_image_path(image_path, cache_dir, max_pixels)
    inline = image_part_inline(resized)["inlineData"]
    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{inline['mimeType']};base64,{inline['data']}"},
        },
        {"type": "text", "text": text},
    ]


def kimi_messages_generate(
    *,
    image_path: Path,
    text: str,
    cache_dir: Path,
    image_max_pixels: int = 100000,
    max_tokens: int = 8192,
    timeout: int = 120,
    retries: int = 1,
) -> str:
    return kimi_generate(
        [
            {
                "role": "user",
                "content": content_parts_openai(
                    image_path,
                    text,
                    cache_dir=cache_dir,
                    max_pixels=image_max_pixels,
                ),
            }
        ],
        max_tokens=max_tokens,
        temperature=1,
        top_p=0.95,
        top_k=-1,
        timeout=timeout,
        retries=retries,
        extra_kwargs={"thinking": {"type": "enabled", "budget_tokens": KIMI_THINKING_BUDGET_TOKENS}},
        stream=True,
    )


def resized_image_path(src: Path, cache_dir: Path, max_pixels: int) -> Path:
    if max_pixels <= 0:
        return src
    source = Path(src)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stat = source.stat()
    key = hashlib.sha1(
        f"{source.resolve()}:{stat.st_size}:{stat.st_mtime_ns}:{max_pixels}".encode("utf-8")
    ).hexdigest()[:16]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", source.stem)[:80]
    dst = cache_dir / f"{stem}_{key}_{max_pixels}.jpg"
    if dst.exists():
        return dst
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(source) as img:
            img.load()
            img = img.convert("RGB")
            width, height = img.size
            scale = min(1.0, math.sqrt(max_pixels / max(width * height, 1)))
            if scale < 1.0:
                img = img.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.LANCZOS)
            img.save(dst, format="JPEG", quality=90, optimize=True)
    return dst


def image_part_inline(path: Path) -> dict[str, Any]:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    with path.open("rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return {"inlineData": {"mimeType": mime, "data": data}}


def image_part_gemini(path: Path, *, cache_dir: Path | None = None, max_pixels: int = 0) -> dict[str, Any]:
    source = resized_image_path(path, cache_dir, max_pixels) if cache_dir and max_pixels > 0 else path
    return image_part_inline(source)


def image_info(path: Path) -> tuple[int, int, str]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as img:
            width, height = img.size
            fmt = img.format or path.suffix.lstrip(".").upper()
    return width, height, fmt


def simple_phash(path: Path, hash_size: int = 8) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        Image.MAX_IMAGE_PIXELS = None
        with Image.open(path) as img:
            img = img.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
            pixels = list(img.getdata())
    avg = sum(pixels) / len(pixels)
    bits = 0
    for pixel in pixels:
        bits = (bits << 1) | int(pixel >= avg)
    return f"{bits:0{hash_size * hash_size // 4}x}"


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


def extract_final_answer(text: str) -> str:
    match = re.search(r"final answer\s*:\s*(.+)", text, re.I | re.S)
    if match:
        answer = match.group(1).strip()
        answer = re.sub(r"</?answer>", "", answer, flags=re.I).strip()
        return answer.splitlines()[0].strip() if "\n" in answer else answer
    return text.strip().splitlines()[-1].strip() if text.strip() else ""


def extract_thinking_and_final_answer(text: str) -> dict[str, str]:
    raw = text.strip()
    if not raw:
        raise ValueError("empty model response")
    try:
        data = extract_json_object(raw)
        final_answer = data.get("final_answer") or data.get("answer")
        thinking = data.get("thinking") or data.get("reasoning") or data.get("explanation") or ""
        if final_answer:
            return {
                "thinking": str(thinking).strip(),
                "final_answer": str(final_answer).strip(),
                "format": "json",
            }
    except Exception:
        pass
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", raw, re.I | re.S)
    final_answer = extract_final_answer(raw)
    if not final_answer:
        raise ValueError("missing final answer")
    thinking = think_match.group(1).strip() if think_match else raw[: raw.lower().rfind("final answer")].strip()
    return {"thinking": thinking, "final_answer": final_answer, "format": "think_final"}


def strip_punctuation(text: str) -> str:
    table = str.maketrans("", "", string.punctuation + "，。！？；：“”‘’、（）【】《》")
    return text.translate(table)


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


def normalize_answer(value: Any, answer_type: str = "short_phrase") -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        items = [normalize_answer(item, "short_phrase") for item in value]
        items = [item for item in items if item]
        if answer_type == "ranked_list":
            return "|".join(items)
        return " ".join(items)
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = strip_punctuation(text)
    text = re.sub(r"\s+", " ", text).strip()
    if answer_type in {"numeric_exact", "numeric_approx", "integer"}:
        nums = parse_numbers(text)
        if nums:
            return str(int(round(nums[0]))) if answer_type == "integer" else format_float(nums[0])
    if answer_type == "boolean":
        if text in {"yes", "y", "true", "correct"}:
            return "yes"
        if text in {"no", "n", "false", "incorrect"}:
            return "no"
    return text


def question_is_low_value(question: str) -> bool:
    stripped = question.strip()
    if not stripped:
        return True
    has_reasoning = bool(REASONING_TERMS.search(stripped))
    return any(pattern.search(stripped) for pattern in LOW_VALUE_PATTERNS) and not has_reasoning


def enum_or_default(value: Any, allowed: Iterable[str], default: str) -> str:
    text = str(value or "").strip()
    return text if text in set(allowed) else default


def list_counter(records: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        value: Any = record
        for part in key.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        if isinstance(value, list):
            counter.update(str(item) for item in value)
        else:
            counter[str(value)] += 1
    return dict(counter.most_common())
