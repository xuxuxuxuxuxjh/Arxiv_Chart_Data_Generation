# Edition2 Scripts

Edition2 changes the generation strategy:

1. Filter/classify charts from 2020-2025.
2. Do not quota-sample images before QA generation.
3. For each chart image, try every task type and build a question candidate pool.
4. Generate Gemini answer with thinking and extract final answer.
5. Verify answer with Gemini judger.
6. Generate Kimi thinking from verified answer.
7. Verify Kimi thinking with Gemini judger.
8. Generate Kimi dense caption.
9. Verify caption with Gemini judger.
10. Sample verified QA/thinking records by target task proportions.
11. Merge sampled QA/thinking with verified captions.

## Commands

Prepare all accepted chart inputs from an existing classified manifest:

```bash
python3 scripts_v2/prepare_charts.py
```

If you need to rerun classification with the edition2 chart type enum:

```bash
python3 scripts_v2/classify_charts_v2.py \
  --input work/candidates_2020_2025.filtered.jsonl \
  --out work/edit2/candidates_2020_2025.chart_classified_v2.jsonl

python3 scripts_v2/prepare_charts.py \
  --input work/edit2/candidates_2020_2025.chart_classified_v2.jsonl
```

Generate one question candidate for every task type on every chart:

```bash
python3 scripts_v2/generate_question_candidates.py \
  --input work/edit2/filtered_charts_2020_2025.jsonl \
  --workers 8 \
  --batch-size 16 \
  --group-tasks-per-image
```

`--group-tasks-per-image` keeps the same output contract, but uses one model call per image to request all missing task types. The script is resumable: existing `(candidate_id, task_type)` pairs in `question_candidates.jsonl` are skipped.

Generate Gemini answers and verify them:

```bash
python3 scripts_v2/generate_and_verify_answers.py \
  --input work/edit2/question_candidates.jsonl \
  --extraction-failures work/edit2/logs/answer_extraction_failures.jsonl \
  --workers 8 \
  --batch-size 16 \
  --retry-failed
```

Generate Kimi thinking and verify it:

```bash
python3 scripts_v2/generate_and_verify_thinking.py \
  --input work/edit2/answers_verified.jsonl \
  --workers 24 \
  --batch-size 192 \
  --image-max-pixels 0 \
  --max-tokens 8192 \
  --timeout 300 \
  --retry-failed
```

The default Kimi model is `kimi-k2.6-qianli` through `/v1/chat/completions`
streaming. `--image-max-pixels 0` sends original images without resizing.

Generate captions and verify them:

```bash
python3 scripts_v2/generate_and_verify_captions.py \
  --input work/edit2/filtered_charts_2020_2025.jsonl \
  --workers 8 \
  --batch-size 8 \
  --image-max-pixels 100000 \
  --max-tokens 8192 \
  --retry-failed
```

Sample verified QA/thinking after all task candidates are generated:

```bash
python3 scripts_v2/sample_verified_questions.py \
  --input work/edit2/kimi_thinking_verified.jsonl \
  --target 50000 \
  --max-per-image 3
```

Use `--target 0` to keep all verified QA/thinking records.

Merge:

```bash
python3 scripts_v2/merge_verified_outputs.py \
  --qa work/edit2/qa_thinking_sampled.jsonl \
  --caption work/edit2/dense_caption_verified.jsonl
```

Review:

```bash
python3 scripts_v2/generate_review_v2.py \
  --merged work/edit2/merged.jsonl \
  --limit 200

python3 scripts_v2/export_static_review_v2.py \
  --html work/edit2/review/review.html \
  --out work/edit2/review_static \
  --max-pixels 100000
```

## Small Dry Run

```bash
python3 scripts_v2/generate_question_candidates.py --limit-images 2 --dry-run
python3 scripts_v2/generate_and_verify_answers.py --limit 5 --dry-run
python3 scripts_v2/generate_and_verify_thinking.py --limit 5 --dry-run
python3 scripts_v2/generate_and_verify_captions.py --limit 2 --dry-run
```
