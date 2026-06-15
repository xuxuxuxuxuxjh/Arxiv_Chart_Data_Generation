#!/usr/bin/env bash
set -euo pipefail

cd "${ARXIV_CHART_PROJECT_ROOT:-/home/i-xujiahao/arxiv_data}"

: "${MODELPROXY_APIKEY:?MODELPROXY_APIKEY is required}"
export ARXIV_CHART_GEMINI_MODEL="${ARXIV_CHART_GEMINI_MODEL:-gemini-3.5-flash}"
export ARXIV_CHART_KIMI_MODEL="${ARXIV_CHART_KIMI_MODEL:-kimi-k2.6-qianli}"
export ARXIV_CHART_WORK="${ARXIV_CHART_WORK:-/mnt/xjh/data/arxiv_chart/work}"
export ARXIV_CHART_EDIT2="${ARXIV_CHART_EDIT2:-$ARXIV_CHART_WORK/edit2_50k_run}"
export PYTHONUNBUFFERED=1

ROOT="$ARXIV_CHART_EDIT2"
SOURCE_ROOT="${ARXIV_CHART_SOURCE_ROOT:-$ARXIV_CHART_WORK/edit2_full}"
LOGDIR="$ROOT/run_logs_streaming"
TARGET_RECORDS="${TARGET_RECORDS:-50000}"
SHARDS="${SHARDS:-8}"
Q_WORKERS="${Q_WORKERS:-8}"
A_WORKERS="${A_WORKERS:-4}"
T_WORKERS="${T_WORKERS:-8}"
C_WORKERS="${C_WORKERS:-8}"
Q_BATCH="${Q_BATCH:-16}"
A_BATCH="${A_BATCH:-16}"
T_BATCH="${T_BATCH:-8}"
C_BATCH="${C_BATCH:-8}"
C_IMAGE_MAX_PIXELS="${C_IMAGE_MAX_PIXELS:-0}"
C_MAX_TOKENS="${C_MAX_TOKENS:-64000}"
C_TIMEOUT="${C_TIMEOUT:-300}"
A_TARGET_VERIFIED="${A_TARGET_VERIFIED:-0}"
T_TARGET_VERIFIED="${T_TARGET_VERIFIED:-0}"
C_TARGET_VERIFIED="${C_TARGET_VERIFIED:-0}"
A_INPUT_CHUNK_LINES="${A_INPUT_CHUNK_LINES:-1024}"
T_INPUT_CHUNK_LINES="${T_INPUT_CHUNK_LINES:-1024}"

mkdir -p "$ROOT"/logs "$ROOT"/reports "$ROOT"/tmp "$ROOT"/shards "$LOGDIR"

count_file() {
  python3 - "$1" <<'PY'
import sys
from pathlib import Path

p = Path(sys.argv[1])
print(sum(1 for line in p.open("rb") if line.strip()) if p.exists() else 0)
PY
}

target_for_shard() {
  local total="$1"
  local i="$2"
  if [ "$total" = "0" ]; then
    echo 0
    return
  fi
  local base=$((total / SHARDS))
  local rem=$((total % SHARDS))
  if [ "$i" -lt "$rem" ]; then
    echo $((base + 1))
  else
    echo "$base"
  fi
}

prepare_if_needed() {
  if [ "$(count_file "$ROOT/filtered_charts_2020_2025.jsonl")" = "$TARGET_RECORDS" ]; then
    echo "PREPARE_SKIP existing $TARGET_RECORDS" | tee -a "$LOGDIR/status.log"
    return
  fi

  echo "PREPARE_START $(date -Is)" | tee -a "$LOGDIR/status.log"
  rm -f "$ROOT/filtered_charts_2020_2025.jsonl"
  python3 - "$SOURCE_ROOT" "$ROOT/filtered_charts_2020_2025.jsonl" "$TARGET_RECORDS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
target = int(sys.argv[3])
out.parent.mkdir(parents=True, exist_ok=True)
patterns = [
    "shards8b/classified_shard8b_*.jsonl",
    "shards/classified_shard_*.jsonl",
    "shards16/classified_shard16_*.jsonl",
    "candidates_2020_2025.chart_classified_v2.jsonl",
]
seen = set()
written = accepted_seen = readable = 0
with out.open("w", encoding="utf-8") as w:
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            print(f"PREPARE reading {path}", flush=True)
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    cid = rec.get("candidate_id") or rec.get("edition2_input_id") or rec.get("image_path")
                    if not cid or cid in seen:
                        continue
                    seen.add(cid)
                    cls = rec.get("classifier") or {}
                    if not (cls.get("accepted") or rec.get("status") == "accepted"):
                        continue
                    accepted_seen += 1
                    image = rec.get("image_path") or rec.get("image")
                    if not image or not Path(image).exists():
                        continue
                    readable += 1
                    rec["image_path"] = image
                    rec["status"] = "accepted"
                    w.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
                    written += 1
                    if written % 1000 == 0:
                        print(f"PREPARE written={written}", flush=True)
                    if written >= target:
                        break
            if written >= target:
                break
        if written >= target:
            break
print(f"PREPARE_DONE written={written} accepted_seen={accepted_seen} readable={readable} unique_seen={len(seen)}", flush=True)
if written < target:
    raise SystemExit(f"Only prepared {written}, target={target}")
PY
}

split_filtered_if_needed() {
  local total=0
  for i in $(seq 0 $((SHARDS - 1))); do
    total=$((total + $(count_file "$ROOT/shards/filtered_shard_${i}.jsonl")))
  done
  if [ "$total" = "$TARGET_RECORDS" ]; then
    echo "SPLIT_SKIP existing total=$total" | tee -a "$LOGDIR/status.log"
    return
  fi

  echo "SPLIT_START $(date -Is)" | tee -a "$LOGDIR/status.log"
  rm -f "$ROOT"/shards/filtered_shard_*.jsonl
  python3 - "$ROOT" "$SHARDS" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
shards = int(sys.argv[2])
writers = []
for i in range(shards):
    p = root / "shards" / f"filtered_shard_{i}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    writers.append(p.open("w", encoding="utf-8"))
try:
    with (root / "filtered_charts_2020_2025.jsonl").open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                writers[idx % shards].write(line)
finally:
    for w in writers:
        w.close()
for i in range(shards):
    p = root / "shards" / f"filtered_shard_{i}.jsonl"
    print(f'SHARD filtered {i} {sum(1 for line in p.open("rb") if line.strip())}', flush=True)
PY
}

write_new_lines() {
  local src="$1" dst="$2" offset_file="$3"
  local limit="${4:-0}"
  python3 - "$src" "$dst" "$offset_file" "$limit" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
off = Path(sys.argv[3])
limit = int(sys.argv[4])
start = int(off.read_text().strip()) if off.exists() and off.read_text().strip() else 0
lines = []
if src.exists():
    with src.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= start and line.strip():
                lines.append(line)
                if limit > 0 and len(lines) >= limit:
                    break
dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as w:
    w.writelines(lines)
off.parent.mkdir(parents=True, exist_ok=True)
off.with_suffix(off.suffix + ".next").write_text(str(start + len(lines)), encoding="utf-8")
print(len(lines))
PY
}

commit_new_lines() {
  local offset_file="$1"
  if [ -f "${offset_file}.next" ]; then
    mv "${offset_file}.next" "$offset_file"
  fi
}

question_loop() {
  local i="$1"
  local name="question_shard_${i}"
  echo "LOOP_START $name $(date -Is)" >> "$LOGDIR/status.log"
  while true; do
    python3 scripts_v2/generate_question_candidates.py \
      --input "$ROOT/shards/filtered_shard_${i}.jsonl" \
      --out "$ROOT/shards/question_candidates_shard_${i}.jsonl" \
      --failures "$ROOT/logs/question_generation_failures_shard_${i}.jsonl" \
      --report "$ROOT/reports/question_candidates_shard_${i}.json" \
      --group-tasks-per-image \
      --workers "$Q_WORKERS" \
      --batch-size "$Q_BATCH" \
      --retries 1 \
      --image-max-pixels 350000
    local qcount expected
    qcount=$(count_file "$ROOT/shards/question_candidates_shard_${i}.jsonl")
    expected=$(( $(count_file "$ROOT/shards/filtered_shard_${i}.jsonl") * 8 ))
    echo "LOOP_PROGRESS $name qcount=$qcount expected=$expected $(date -Is)" >> "$LOGDIR/status.log"
    if [ "$qcount" -ge "$expected" ]; then break; fi
    sleep 30
  done
  touch "$LOGDIR/${name}.done"
  echo "LOOP_DONE $name $(date -Is)" >> "$LOGDIR/status.log"
}

caption_loop() {
  local i="$1"
  local name="caption_shard_${i}"
  local target
  target=$(target_for_shard "$C_TARGET_VERIFIED" "$i")
  echo "LOOP_START $name $(date -Is)" >> "$LOGDIR/status.log"
  while true; do
    local current
    current=$(count_file "$ROOT/shards/dense_caption_verified_shard_${i}.jsonl")
    if [ "$target" != "0" ] && [ "$current" -ge "$target" ]; then
      echo "LOOP_TARGET_REACHED $name verified=$current target=$target $(date -Is)" >> "$LOGDIR/status.log"
      break
    fi
    python3 scripts_v2/generate_and_verify_captions.py \
      --input "$ROOT/shards/filtered_shard_${i}.jsonl" \
      --raw-out "$ROOT/shards/dense_caption_raw_shard_${i}.jsonl" \
      --verified-out "$ROOT/shards/dense_caption_verified_shard_${i}.jsonl" \
      --failures "$ROOT/logs/caption_failures_shard_${i}.jsonl" \
      --judge-failures "$ROOT/logs/caption_judge_failures_shard_${i}.jsonl" \
      --report "$ROOT/reports/dense_caption_verified_shard_${i}.json" \
      --workers "$C_WORKERS" \
      --batch-size "$C_BATCH" \
      --image-max-pixels "$C_IMAGE_MAX_PIXELS" \
      --max-tokens "$C_MAX_TOKENS" \
      --timeout "$C_TIMEOUT" \
      --retries 1
    local ccount expected
    ccount=$(count_file "$ROOT/shards/dense_caption_verified_shard_${i}.jsonl")
    expected=$(count_file "$ROOT/shards/filtered_shard_${i}.jsonl")
    echo "LOOP_PROGRESS $name ccount=$ccount expected=$expected $(date -Is)" >> "$LOGDIR/status.log"
    if [ "$ccount" -ge "$expected" ]; then break; fi
    sleep 30
  done
  touch "$LOGDIR/${name}.done"
  echo "LOOP_DONE $name $(date -Is)" >> "$LOGDIR/status.log"
}

answer_loop() {
  local i="$1"
  local name="answer_shard_${i}"
  local target
  target=$(target_for_shard "$A_TARGET_VERIFIED" "$i")
  echo "LOOP_START $name $(date -Is)" >> "$LOGDIR/status.log"
  while true; do
    local current
    current=$(count_file "$ROOT/shards/answers_verified_shard_${i}.jsonl")
    if [ "$target" != "0" ] && [ "$current" -ge "$target" ]; then
      echo "LOOP_TARGET_REACHED $name verified=$current target=$target $(date -Is)" >> "$LOGDIR/status.log"
      break
    fi
    local new_input="$ROOT/shards/question_candidates_for_answer_new_shard_${i}.jsonl"
    local offset_file="$ROOT/shards/question_candidates_for_answer_shard_${i}.offset"
    local n
    if [ "$target" = "0" ]; then
      n=$(write_new_lines "$ROOT/shards/question_candidates_shard_${i}.jsonl" "$new_input" "$offset_file")
    else
      local remaining limit
      remaining=$((target - current))
      limit="$A_INPUT_CHUNK_LINES"
      if [ "$remaining" -lt "$limit" ]; then limit="$remaining"; fi
      n=$(write_new_lines "$ROOT/shards/question_candidates_shard_${i}.jsonl" "$new_input" "$offset_file" "$limit")
    fi
    if [ "$n" != "0" ]; then
      python3 scripts_v2/generate_and_verify_answers.py \
        --input "$new_input" \
        --raw-out "$ROOT/shards/answers_raw_shard_${i}.jsonl" \
        --verified-out "$ROOT/shards/answers_verified_shard_${i}.jsonl" \
        --failures "$ROOT/logs/answer_failures_shard_${i}.jsonl" \
        --extraction-failures "$ROOT/logs/answer_extraction_failures_shard_${i}.jsonl" \
        --judge-failures "$ROOT/logs/answer_judge_failures_shard_${i}.jsonl" \
        --report "$ROOT/reports/answers_verified_shard_${i}.json" \
        --workers "$A_WORKERS" \
        --batch-size "$A_BATCH" \
        --answer-samples 3 \
        --answer-retries 1 \
        --judge-retries 1 \
        --image-max-pixels 350000
      commit_new_lines "$offset_file"
    fi
    echo "LOOP_PROGRESS $name new_questions=$n raw=$(count_file "$ROOT/shards/answers_raw_shard_${i}.jsonl") verified=$(count_file "$ROOT/shards/answers_verified_shard_${i}.jsonl") $(date -Is)" >> "$LOGDIR/status.log"
    if [ -f "$LOGDIR/question_shard_${i}.done" ] && [ "$n" = "0" ]; then break; fi
    sleep 20
  done
  touch "$LOGDIR/${name}.done"
  echo "LOOP_DONE $name $(date -Is)" >> "$LOGDIR/status.log"
}

thinking_loop() {
  local i="$1"
  local name="thinking_shard_${i}"
  local target
  target=$(target_for_shard "$T_TARGET_VERIFIED" "$i")
  echo "LOOP_START $name $(date -Is)" >> "$LOGDIR/status.log"
  while true; do
    local current
    current=$(count_file "$ROOT/shards/kimi_thinking_verified_shard_${i}.jsonl")
    if [ "$target" != "0" ] && [ "$current" -ge "$target" ]; then
      echo "LOOP_TARGET_REACHED $name verified=$current target=$target $(date -Is)" >> "$LOGDIR/status.log"
      break
    fi
    local new_input="$ROOT/shards/answers_verified_for_thinking_new_shard_${i}.jsonl"
    local offset_file="$ROOT/shards/answers_verified_for_thinking_shard_${i}.offset"
    local n
    if [ "$target" = "0" ]; then
      n=$(write_new_lines "$ROOT/shards/answers_verified_shard_${i}.jsonl" "$new_input" "$offset_file")
    else
      local remaining limit
      remaining=$((target - current))
      limit="$T_INPUT_CHUNK_LINES"
      if [ "$remaining" -lt "$limit" ]; then limit="$remaining"; fi
      n=$(write_new_lines "$ROOT/shards/answers_verified_shard_${i}.jsonl" "$new_input" "$offset_file" "$limit")
    fi
    if [ "$n" != "0" ]; then
      python3 scripts_v2/generate_and_verify_thinking.py \
        --input "$new_input" \
        --raw-out "$ROOT/shards/kimi_thinking_raw_shard_${i}.jsonl" \
        --verified-out "$ROOT/shards/kimi_thinking_verified_shard_${i}.jsonl" \
        --failures "$ROOT/logs/kimi_thinking_failures_shard_${i}.jsonl" \
        --judge-failures "$ROOT/logs/kimi_thinking_judge_failures_shard_${i}.jsonl" \
        --report "$ROOT/reports/kimi_thinking_verified_shard_${i}.json" \
        --workers "$T_WORKERS" \
        --batch-size "$T_BATCH" \
        --image-max-pixels 0 \
        --max-tokens 64000 \
        --timeout 300 \
        --retries 1 \
        --judge-image-max-pixels 350000
      commit_new_lines "$offset_file"
    fi
    echo "LOOP_PROGRESS $name new_answers=$n raw=$(count_file "$ROOT/shards/kimi_thinking_raw_shard_${i}.jsonl") verified=$(count_file "$ROOT/shards/kimi_thinking_verified_shard_${i}.jsonl") $(date -Is)" >> "$LOGDIR/status.log"
    if [ -f "$LOGDIR/answer_shard_${i}.done" ] && [ "$n" = "0" ]; then break; fi
    sleep 20
  done
  touch "$LOGDIR/${name}.done"
  echo "LOOP_DONE $name $(date -Is)" >> "$LOGDIR/status.log"
}

prepare_if_needed | tee -a "$LOGDIR/prepare.log"
split_filtered_if_needed | tee -a "$LOGDIR/split.log"
echo "STREAMING_START $(date -Is)" | tee -a "$LOGDIR/status.log"
echo "STREAMING_CONFIG shards=$SHARDS q_workers=$Q_WORKERS q_batch=$Q_BATCH a_workers=$A_WORKERS a_batch=$A_BATCH t_workers=$T_WORKERS t_batch=$T_BATCH c_workers=$C_WORKERS c_batch=$C_BATCH c_image_max_pixels=$C_IMAGE_MAX_PIXELS c_max_tokens=$C_MAX_TOKENS c_timeout=$C_TIMEOUT a_target_verified=$A_TARGET_VERIFIED t_target_verified=$T_TARGET_VERIFIED c_target_verified=$C_TARGET_VERIFIED a_input_chunk_lines=$A_INPUT_CHUNK_LINES t_input_chunk_lines=$T_INPUT_CHUNK_LINES" | tee -a "$LOGDIR/status.log"

for i in $(seq 0 $((SHARDS - 1))); do
  question_loop "$i" > "$LOGDIR/question_stream_shard_${i}.log" 2>&1 & echo $! > "$LOGDIR/question_stream_shard_${i}.pid"
  caption_loop "$i" > "$LOGDIR/caption_stream_shard_${i}.log" 2>&1 & echo $! > "$LOGDIR/caption_stream_shard_${i}.pid"
  answer_loop "$i" > "$LOGDIR/answer_stream_shard_${i}.log" 2>&1 & echo $! > "$LOGDIR/answer_stream_shard_${i}.pid"
  thinking_loop "$i" > "$LOGDIR/thinking_stream_shard_${i}.log" 2>&1 & echo $! > "$LOGDIR/thinking_stream_shard_${i}.pid"
done

failed=0
for p in "$LOGDIR"/*_stream_shard_*.pid; do
  pid=$(cat "$p")
  if ! wait "$pid"; then
    echo "STREAM_PROCESS_FAILED $(basename "$p")" | tee -a "$LOGDIR/status.log"
    failed=1
  fi
done
if [ "$failed" != 0 ]; then exit 1; fi

echo "STREAMING_GENERATION_DONE $(date -Is)" | tee -a "$LOGDIR/status.log"
cat "$ROOT"/shards/question_candidates_shard_*.jsonl > "$ROOT/question_candidates.jsonl"
cat "$ROOT"/shards/dense_caption_raw_shard_*.jsonl > "$ROOT/dense_caption_raw.jsonl"
cat "$ROOT"/shards/dense_caption_verified_shard_*.jsonl > "$ROOT/dense_caption_verified.jsonl"
cat "$ROOT"/shards/answers_raw_shard_*.jsonl > "$ROOT/answers_raw.jsonl"
cat "$ROOT"/shards/answers_verified_shard_*.jsonl > "$ROOT/answers_verified.jsonl"
cat "$ROOT"/shards/kimi_thinking_raw_shard_*.jsonl > "$ROOT/kimi_thinking_raw.jsonl"
cat "$ROOT"/shards/kimi_thinking_verified_shard_*.jsonl > "$ROOT/kimi_thinking_verified.jsonl"

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
