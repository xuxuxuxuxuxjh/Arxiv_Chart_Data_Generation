#!/usr/bin/env bash
set -euo pipefail

cd "${ARXIV_CHART_PROJECT_ROOT:-/home/i-xujiahao/arxiv_data}"

: "${MODELPROXY_APIKEY:?MODELPROXY_APIKEY is required}"
export ARXIV_CHART_GEMINI_MODEL="${ARXIV_CHART_GEMINI_MODEL:-gemini-3.5-flash}"
export ARXIV_CHART_KIMI_MODEL="${ARXIV_CHART_KIMI_MODEL:-kimi-k2.6-qianli}"
export ARXIV_CHART_EDIT2="${ARXIV_CHART_EDIT2:-/mnt/xjh/data/arxiv_chart/work/edit2_charxiv}"
export PYTHONUNBUFFERED=1

ROOT="$ARXIV_CHART_EDIT2"
IMAGE_DIR="${CHARXIV_IMAGE_DIR:-/home/i-xujiahao/arxiv_data/charxiv_image_arxiv}"
LOGDIR="$ROOT/run_logs"

Q_WORKERS="${Q_WORKERS:-8}"
A_WORKERS="${A_WORKERS:-16}"
T_WORKERS="${T_WORKERS:-24}"
C_WORKERS="${C_WORKERS:-24}"
Q_BATCH="${Q_BATCH:-16}"
A_BATCH="${A_BATCH:-64}"
T_BATCH="${T_BATCH:-32}"
C_BATCH="${C_BATCH:-32}"
Q_IMAGE_MAX_PIXELS="${Q_IMAGE_MAX_PIXELS:-350000}"
A_IMAGE_MAX_PIXELS="${A_IMAGE_MAX_PIXELS:-350000}"
T_IMAGE_MAX_PIXELS="${T_IMAGE_MAX_PIXELS:-0}"
T_JUDGE_IMAGE_MAX_PIXELS="${T_JUDGE_IMAGE_MAX_PIXELS:-350000}"
C_IMAGE_MAX_PIXELS="${C_IMAGE_MAX_PIXELS:-0}"
KIMI_MAX_TOKENS="${KIMI_MAX_TOKENS:-64000}"
KIMI_TIMEOUT="${KIMI_TIMEOUT:-300}"
QUESTION_TARGET="${QUESTION_TARGET:-8000}"
QUESTION_GROUP_ROUNDS="${QUESTION_GROUP_ROUNDS:-3}"
QUESTION_SINGLE_ROUNDS="${QUESTION_SINGLE_ROUNDS:-2}"

mkdir -p "$ROOT"/logs "$ROOT"/reports "$ROOT"/tmp "$LOGDIR"

count_file() {
  python3 - "$1" <<'PY'
import sys
from pathlib import Path

p = Path(sys.argv[1])
print(sum(1 for line in p.open("rb") if line.strip()) if p.exists() else 0)
PY
}

echo "CHARXIV_PIPELINE_START $(date -Is)" | tee -a "$LOGDIR/status.log"
echo "CONFIG image_dir=$IMAGE_DIR root=$ROOT q_workers=$Q_WORKERS a_workers=$A_WORKERS t_workers=$T_WORKERS c_workers=$C_WORKERS q_batch=$Q_BATCH a_batch=$A_BATCH t_batch=$T_BATCH c_batch=$C_BATCH q_image_max_pixels=$Q_IMAGE_MAX_PIXELS a_image_max_pixels=$A_IMAGE_MAX_PIXELS t_image_max_pixels=$T_IMAGE_MAX_PIXELS c_image_max_pixels=$C_IMAGE_MAX_PIXELS kimi_max_tokens=$KIMI_MAX_TOKENS kimi_timeout=$KIMI_TIMEOUT question_target=$QUESTION_TARGET question_group_rounds=$QUESTION_GROUP_ROUNDS question_single_rounds=$QUESTION_SINGLE_ROUNDS" | tee -a "$LOGDIR/status.log"

python3 - "$IMAGE_DIR" "$ROOT/filtered_charts_2020_2025.jsonl" <<'PY'
import json
import re
import sys
from pathlib import Path

image_dir = Path(sys.argv[1])
out = Path(sys.argv[2])
suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
records = []
for idx, path in enumerate(sorted(p for p in image_dir.iterdir() if p.suffix.lower() in suffixes), 1):
    stem = path.stem
    paper_match = re.match(r"(\d{4}\.\d{4,5})", stem)
    paper_id = paper_match.group(1) if paper_match else stem
    year = int("20" + paper_id[:2]) if re.match(r"\d{4}\.", paper_id) else None
    month = int(paper_id[2:4]) if re.match(r"\d{4}\.", paper_id) else None
    records.append(
        {
            "candidate_id": stem,
            "edition2_input_id": stem,
            "image_path": str(path),
            "paper_id": paper_id,
            "year": year,
            "month": month,
            "figure_index": idx,
            "image_kind": "charxiv_image_arxiv",
            "is_charxiv_paper": True,
            "json_path": "",
            "caption_latex": "",
            "classifier": {
                "accepted": True,
                "chart_type": "other_chart",
                "chart_confidence": 1.0,
                "is_multi_panel": False,
                "panel_count": 1,
                "source": "charxiv_image_arxiv_directory",
            },
        }
    )
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    for record in records:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
print(f"wrote {out}: {len(records)} records", flush=True)
if len(records) != 1000:
    raise SystemExit(f"expected 1000 images, got {len(records)}")
PY

echo "QUESTION_START $(date -Is)" | tee -a "$LOGDIR/status.log"
for round in $(seq 1 "$QUESTION_GROUP_ROUNDS"); do
  current=$(count_file "$ROOT/question_candidates.jsonl")
  if [ "$current" -ge "$QUESTION_TARGET" ]; then break; fi
  echo "QUESTION_GROUP_ROUND_START round=$round current=$current $(date -Is)" | tee -a "$LOGDIR/status.log"
  python3 scripts_v2/generate_question_candidates.py \
    --input "$ROOT/filtered_charts_2020_2025.jsonl" \
    --out "$ROOT/question_candidates.jsonl" \
    --failures "$ROOT/logs/question_generation_failures.jsonl" \
    --report "$ROOT/reports/question_candidates.json" \
    --group-tasks-per-image \
    --workers "$Q_WORKERS" \
    --batch-size "$Q_BATCH" \
    --retries 1 \
    --image-max-pixels "$Q_IMAGE_MAX_PIXELS" \
    2>&1 | tee -a "$LOGDIR/question.log"
done

for round in $(seq 1 "$QUESTION_SINGLE_ROUNDS"); do
  current=$(count_file "$ROOT/question_candidates.jsonl")
  if [ "$current" -ge "$QUESTION_TARGET" ]; then break; fi
  echo "QUESTION_SINGLE_ROUND_START round=$round current=$current $(date -Is)" | tee -a "$LOGDIR/status.log"
  python3 scripts_v2/generate_question_candidates.py \
    --input "$ROOT/filtered_charts_2020_2025.jsonl" \
    --out "$ROOT/question_candidates.jsonl" \
    --failures "$ROOT/logs/question_generation_failures.jsonl" \
    --report "$ROOT/reports/question_candidates.json" \
    --workers "$Q_WORKERS" \
    --batch-size "$Q_BATCH" \
    --retries 1 \
    --image-max-pixels "$Q_IMAGE_MAX_PIXELS" \
    2>&1 | tee -a "$LOGDIR/question.log"
done
echo "QUESTION_DONE $(date -Is) count=$(count_file "$ROOT/question_candidates.jsonl")" | tee -a "$LOGDIR/status.log"

echo "ANSWER_START $(date -Is)" | tee -a "$LOGDIR/status.log"
python3 scripts_v2/generate_and_verify_answers.py \
  --input "$ROOT/question_candidates.jsonl" \
  --raw-out "$ROOT/answers_raw.jsonl" \
  --verified-out "$ROOT/answers_verified.jsonl" \
  --failures "$ROOT/logs/answer_failures.jsonl" \
  --extraction-failures "$ROOT/logs/answer_extraction_failures.jsonl" \
  --judge-failures "$ROOT/logs/answer_judge_failures.jsonl" \
  --report "$ROOT/reports/answers_verified.json" \
  --workers "$A_WORKERS" \
  --batch-size "$A_BATCH" \
  --answer-samples 3 \
  --answer-retries 1 \
  --judge-retries 1 \
  --image-max-pixels "$A_IMAGE_MAX_PIXELS" \
  2>&1 | tee "$LOGDIR/answer.log"
echo "ANSWER_DONE $(date -Is) verified=$(count_file "$ROOT/answers_verified.jsonl") raw=$(count_file "$ROOT/answers_raw.jsonl")" | tee -a "$LOGDIR/status.log"

echo "THINKING_START $(date -Is)" | tee -a "$LOGDIR/status.log"
python3 scripts_v2/generate_and_verify_thinking.py \
  --input "$ROOT/answers_verified.jsonl" \
  --raw-out "$ROOT/kimi_thinking_raw.jsonl" \
  --verified-out "$ROOT/kimi_thinking_verified.jsonl" \
  --failures "$ROOT/logs/kimi_thinking_failures.jsonl" \
  --judge-failures "$ROOT/logs/kimi_thinking_judge_failures.jsonl" \
  --report "$ROOT/reports/kimi_thinking_verified.json" \
  --workers "$T_WORKERS" \
  --batch-size "$T_BATCH" \
  --image-max-pixels "$T_IMAGE_MAX_PIXELS" \
  --max-tokens "$KIMI_MAX_TOKENS" \
  --timeout "$KIMI_TIMEOUT" \
  --retries 1 \
  --judge-image-max-pixels "$T_JUDGE_IMAGE_MAX_PIXELS" \
  2>&1 | tee "$LOGDIR/thinking.log"
echo "THINKING_DONE $(date -Is) verified=$(count_file "$ROOT/kimi_thinking_verified.jsonl") raw=$(count_file "$ROOT/kimi_thinking_raw.jsonl")" | tee -a "$LOGDIR/status.log"

echo "CAPTION_START $(date -Is)" | tee -a "$LOGDIR/status.log"
python3 scripts_v2/generate_and_verify_captions.py \
  --input "$ROOT/filtered_charts_2020_2025.jsonl" \
  --raw-out "$ROOT/dense_caption_raw.jsonl" \
  --verified-out "$ROOT/dense_caption_verified.jsonl" \
  --failures "$ROOT/logs/caption_failures.jsonl" \
  --judge-failures "$ROOT/logs/caption_judge_failures.jsonl" \
  --report "$ROOT/reports/dense_caption_verified.json" \
  --workers "$C_WORKERS" \
  --batch-size "$C_BATCH" \
  --image-max-pixels "$C_IMAGE_MAX_PIXELS" \
  --max-tokens "$KIMI_MAX_TOKENS" \
  --timeout "$KIMI_TIMEOUT" \
  --retries 1 \
  2>&1 | tee "$LOGDIR/caption.log"
echo "CAPTION_DONE $(date -Is) verified=$(count_file "$ROOT/dense_caption_verified.jsonl") raw=$(count_file "$ROOT/dense_caption_raw.jsonl")" | tee -a "$LOGDIR/status.log"

python3 scripts_v2/sample_verified_questions.py \
  --input "$ROOT/kimi_thinking_verified.jsonl" \
  --out "$ROOT/qa_thinking_sampled.jsonl" \
  --report "$ROOT/reports/qa_thinking_sampled.json" \
  --target 0 \
  --max-per-image 0 \
  2>&1 | tee "$LOGDIR/sample.log"

python3 scripts_v2/merge_verified_outputs.py \
  --qa "$ROOT/qa_thinking_sampled.jsonl" \
  --caption "$ROOT/dense_caption_verified.jsonl" \
  --out "$ROOT/merged.jsonl" \
  --report "$ROOT/reports/merged.json" \
  --require-caption \
  2>&1 | tee "$LOGDIR/merge.log"

echo "MERGE_DONE $(date -Is) merged=$(count_file "$ROOT/merged.jsonl")" | tee -a "$LOGDIR/status.log"
echo "CHARXIV_PIPELINE_DONE $(date -Is)" | tee -a "$LOGDIR/status.log"
