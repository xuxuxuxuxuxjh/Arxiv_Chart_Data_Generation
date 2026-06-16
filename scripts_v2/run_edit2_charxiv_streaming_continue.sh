#!/usr/bin/env bash
set -euo pipefail

cd "${ARXIV_CHART_PROJECT_ROOT:-/home/i-xujiahao/arxiv_data}"

export ARXIV_CHART_GEMINI_MODEL="${ARXIV_CHART_GEMINI_MODEL:-gemini-3.5-flash}"
export ARXIV_CHART_KIMI_MODEL="${ARXIV_CHART_KIMI_MODEL:-kimi-k2.6-qianli}"
export ARXIV_CHART_EDIT2="${ARXIV_CHART_EDIT2:-/mnt/xjh/data/arxiv_chart/work/edit2_charxiv}"
export PYTHONUNBUFFERED=1

ROOT="$ARXIV_CHART_EDIT2"
LOGDIR="$ROOT/run_logs_streaming_continue"

A_WORKERS="${A_WORKERS:-64}"
A_BATCH="${A_BATCH:-256}"
T_WORKERS="${T_WORKERS:-48}"
T_BATCH="${T_BATCH:-128}"
A_IMAGE_MAX_PIXELS="${A_IMAGE_MAX_PIXELS:-350000}"
T_IMAGE_MAX_PIXELS="${T_IMAGE_MAX_PIXELS:-0}"
T_JUDGE_IMAGE_MAX_PIXELS="${T_JUDGE_IMAGE_MAX_PIXELS:-350000}"
KIMI_MAX_TOKENS="${KIMI_MAX_TOKENS:-64000}"
KIMI_TIMEOUT="${KIMI_TIMEOUT:-300}"
T_INPUT_CHUNK_LINES="${T_INPUT_CHUNK_LINES:-512}"
POLL_SECONDS="${POLL_SECONDS:-15}"

mkdir -p "$ROOT"/logs "$ROOT"/reports "$ROOT"/tmp "$LOGDIR"

count_file() {
  python3 - "$1" <<'PY'
import sys
from pathlib import Path

p = Path(sys.argv[1])
print(sum(1 for line in p.open("rb") if line.strip()) if p.exists() else 0)
PY
}

repair_jsonl_if_needed() {
  python3 - "$1" <<'PY'
import json
import shutil
import sys
import time
from pathlib import Path

p = Path(sys.argv[1])
if not p.exists():
    print(f"JSONL_OK missing {p}", flush=True)
    raise SystemExit(0)
good: list[bytes] = []
bad = 0
with p.open("rb") as f:
    for line_no, line in enumerate(f, 1):
        if not line.strip():
            continue
        try:
            json.loads(line)
        except Exception as exc:
            bad += 1
            print(f"JSONL_BAD {p}:{line_no}: {exc}", flush=True)
            continue
        good.append(line if line.endswith(b"\n") else line + b"\n")
if bad:
    backup = p.with_name(f"{p.name}.corrupt_{time.strftime('%Y%m%d_%H%M%S')}.bak")
    shutil.copy2(p, backup)
    with p.open("wb") as w:
        w.writelines(good)
    print(f"JSONL_REPAIRED {p} kept={len(good)} bad={bad} backup={backup}", flush=True)
else:
    print(f"JSONL_OK {p} records={len(good)}", flush=True)
PY
}

write_new_complete_lines() {
  local src="$1" dst="$2" offset_file="$3" limit="$4"
  python3 - "$src" "$dst" "$offset_file" "$limit" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
off = Path(sys.argv[3])
limit = int(sys.argv[4])
start = int(off.read_text().strip()) if off.exists() and off.read_text().strip() else 0
lines: list[str] = []
next_offset = start
if src.exists():
    with src.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx < start:
                continue
            if not line.endswith("\n"):
                break
            next_offset = idx + 1
            if line.strip():
                lines.append(line)
                if limit > 0 and len(lines) >= limit:
                    break
dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as w:
    w.writelines(lines)
off.parent.mkdir(parents=True, exist_ok=True)
off.with_suffix(off.suffix + ".next").write_text(str(next_offset), encoding="utf-8")
print(len(lines))
PY
}

commit_new_lines() {
  local offset_file="$1"
  if [ -f "${offset_file}.next" ]; then
    mv "${offset_file}.next" "$offset_file"
  fi
}

answer_loop() {
  echo "ANSWER_STREAM_START $(date -Is)" | tee -a "$LOGDIR/status.log"
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
    --image-max-pixels "$A_IMAGE_MAX_PIXELS"
  echo "ANSWER_STREAM_DONE $(date -Is) verified=$(count_file "$ROOT/answers_verified.jsonl") raw=$(count_file "$ROOT/answers_raw.jsonl")" | tee -a "$LOGDIR/status.log"
}

thinking_loop() {
  local input="$ROOT/tmp/answers_verified_for_thinking_stream.jsonl"
  local offset_file="$ROOT/tmp/answers_verified_for_thinking_stream.offset"
  echo "THINKING_STREAM_START $(date -Is)" | tee -a "$LOGDIR/status.log"
  while true; do
    local n
    n=$(write_new_complete_lines "$ROOT/answers_verified.jsonl" "$input" "$offset_file" "$T_INPUT_CHUNK_LINES")
    if [ "$n" != "0" ]; then
      python3 scripts_v2/generate_and_verify_thinking.py \
        --input "$input" \
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
        --judge-image-max-pixels "$T_JUDGE_IMAGE_MAX_PIXELS"
      commit_new_lines "$offset_file"
    fi
    echo "THINKING_STREAM_PROGRESS $(date -Is) new_answers=$n answer_verified=$(count_file "$ROOT/answers_verified.jsonl") thinking_verified=$(count_file "$ROOT/kimi_thinking_verified.jsonl") thinking_raw=$(count_file "$ROOT/kimi_thinking_raw.jsonl")" | tee -a "$LOGDIR/status.log"
    if [ -f "$LOGDIR/answer.done" ] && [ "$n" = "0" ]; then
      break
    fi
    if [ -f "$LOGDIR/answer.failed" ] && [ "$n" = "0" ]; then
      echo "THINKING_STREAM_STOP answer_failed $(date -Is)" | tee -a "$LOGDIR/status.log"
      return 1
    fi
    sleep "$POLL_SECONDS"
  done
  echo "THINKING_STREAM_DONE $(date -Is) verified=$(count_file "$ROOT/kimi_thinking_verified.jsonl") raw=$(count_file "$ROOT/kimi_thinking_raw.jsonl")" | tee -a "$LOGDIR/status.log"
}

echo "CHARXIV_STREAMING_CONTINUE_START $(date -Is)" | tee -a "$LOGDIR/status.log"
echo "CONFIG root=$ROOT a_workers=$A_WORKERS a_batch=$A_BATCH t_workers=$T_WORKERS t_batch=$T_BATCH t_input_chunk_lines=$T_INPUT_CHUNK_LINES kimi_max_tokens=$KIMI_MAX_TOKENS kimi_timeout=$KIMI_TIMEOUT" | tee -a "$LOGDIR/status.log"

for f in \
  "$ROOT/question_candidates.jsonl" \
  "$ROOT/answers_raw.jsonl" \
  "$ROOT/answers_verified.jsonl" \
  "$ROOT/kimi_thinking_raw.jsonl" \
  "$ROOT/kimi_thinking_verified.jsonl" \
  "$ROOT/dense_caption_verified.jsonl"; do
  repair_jsonl_if_needed "$f" | tee -a "$LOGDIR/jsonl_repair.log"
done

rm -f "$LOGDIR/answer.done" "$LOGDIR/answer.failed" "$LOGDIR/thinking.done" "$LOGDIR/thinking.failed"

(
  set +e
  answer_loop > "$LOGDIR/answer_stream.log" 2>&1
  rc=$?
  if [ "$rc" = 0 ]; then
    touch "$LOGDIR/answer.done"
  else
    touch "$LOGDIR/answer.failed"
  fi
  exit "$rc"
) &
answer_pid=$!

(
  set +e
  thinking_loop > "$LOGDIR/thinking_stream.log" 2>&1
  rc=$?
  if [ "$rc" = 0 ]; then
    touch "$LOGDIR/thinking.done"
  else
    touch "$LOGDIR/thinking.failed"
  fi
  exit "$rc"
) &
thinking_pid=$!

failed=0
if ! wait "$answer_pid"; then
  echo "ANSWER_PROCESS_FAILED $(date -Is)" | tee -a "$LOGDIR/status.log"
  failed=1
fi
if ! wait "$thinking_pid"; then
  echo "THINKING_PROCESS_FAILED $(date -Is)" | tee -a "$LOGDIR/status.log"
  failed=1
fi
if [ "$failed" != 0 ]; then
  exit 1
fi

python3 scripts_v2/sample_verified_questions.py \
  --input "$ROOT/kimi_thinking_verified.jsonl" \
  --out "$ROOT/qa_thinking_sampled.jsonl" \
  --report "$ROOT/reports/qa_thinking_sampled.json" \
  --target 0 \
  --max-per-image 0 \
  > "$LOGDIR/sample.log" 2>&1

python3 scripts_v2/merge_verified_outputs.py \
  --qa "$ROOT/qa_thinking_sampled.jsonl" \
  --caption "$ROOT/dense_caption_verified.jsonl" \
  --out "$ROOT/merged.jsonl" \
  --report "$ROOT/reports/merged.json" \
  --require-caption \
  > "$LOGDIR/merge.log" 2>&1

echo "MERGE_DONE $(date -Is) merged=$(count_file "$ROOT/merged.jsonl")" | tee -a "$LOGDIR/status.log"
python3 - "$ROOT" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
for name in [
    "filtered_charts_2020_2025.jsonl",
    "question_candidates.jsonl",
    "answers_raw.jsonl",
    "answers_verified.jsonl",
    "kimi_thinking_raw.jsonl",
    "kimi_thinking_verified.jsonl",
    "dense_caption_raw.jsonl",
    "dense_caption_verified.jsonl",
    "qa_thinking_sampled.jsonl",
    "merged.jsonl",
]:
    p = root / name
    count = sum(1 for line in p.open("rb") if line.strip()) if p.exists() else 0
    print(f"{name}: {count}", flush=True)
PY
echo "CHARXIV_STREAMING_CONTINUE_DONE $(date -Is)" | tee -a "$LOGDIR/status.log"
