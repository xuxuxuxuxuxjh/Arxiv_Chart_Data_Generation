# Edition2 Full Run - 2026-06-12

Work directory: `work/edit2` (`/mnt/xjh/data/arxiv_chart/work/edit2` through the `work` symlink).

## Inputs

- `filtered_charts_2020_2025.jsonl`: 553 accepted/deduped chart images.
- Years: 2020-2025.
- No paper/month/year sampling quota.
- Each image attempted all 8 task types.

## Question Candidate Pool

- `question_candidates.jsonl`: 4423 records.
- Expected maximum: 553 images x 8 task types = 4424.
- Only missing task: one `cross_element_comparison` for `2006.00008_fig0002_img01`.
- Per task:
  - `approximate_value_estimation`: 553
  - `trend_pattern_analysis`: 553
  - `hypothetical_reasoning`: 553
  - `multi_element_synthesis`: 553
  - `fine_grained_visual_reading`: 553
  - `anomaly_extrema_detection`: 553
  - `complex_calculation`: 553
  - `cross_element_comparison`: 552

## Answer Verification

- `answers_verified.jsonl`: 4347 records.
- `answers_raw.jsonl`: 4362 records.
- `logs/answer_failures.jsonl`: 64 records, mostly Gemini safety/prohibited-content responses.
- `logs/answer_judge_failures.jsonl`: 15 records.
- `logs/answer_extraction_failures.jsonl`: 0 records.
- Verified answers cover all 553 images.

## Kimi Thinking Verification

- Kimi config: `/v1/messages`, `kimi-k2.6-aliyun`, `max_tokens=8192`, image resize `H*W<=100000`.
- `kimi_thinking_verified.jsonl`: 4287 records.
- `kimi_thinking_raw.jsonl`: 4341 records.
- `logs/kimi_thinking_failures.jsonl`: 11 records.
- `logs/kimi_thinking_judge_failures.jsonl`: 54 records.
- Verified thinking covers all 553 images.

## Dense Caption Verification

- Kimi config: `/v1/messages`, `kimi-k2.6-aliyun`, `max_tokens=8192`, image resize `H*W<=100000`.
- `dense_caption_raw.jsonl`: 553 records.
- `dense_caption_verified.jsonl`: 373 records.
- `logs/caption_failures.jsonl`: 0 records.
- `logs/caption_judge_failures.jsonl`: 180 records.
- Caption judge failures are mostly concrete hallucinations: axis labels, units, legends, line styles, panel counts, and caption-only paper context.

## Final Sampling And Merge

- `qa_thinking_sampled.jsonl`: 4287 records, `target=0` kept all verified QA/thinking.
- `merged.jsonl`: 2922 records requiring verified answer, verified thinking, and verified caption.
- `review/review.html`: generated.
- `review_static/review_static.html`: generated with 200 resized images.

Merged distribution:

- Task types:
  - `trend_pattern_analysis`: 369
  - `anomaly_extrema_detection`: 368
  - `multi_element_synthesis`: 368
  - `approximate_value_estimation`: 365
  - `complex_calculation`: 365
  - `fine_grained_visual_reading`: 363
  - `cross_element_comparison`: 362
  - `hypothetical_reasoning`: 362
- Answer types:
  - `numeric_approx`: 1361
  - `short_phrase`: 698
  - `choice`: 693
  - `trend_label`: 82
  - `numeric_exact`: 42
  - `boolean`: 25
  - `ranked_list`: 16
  - `integer`: 5
- Difficulty:
  - `medium`: 1914
  - `hard`: 1008

Quality report: `work/edit2/reports/quality_stats.json`.
