# Edition2 Qianli Thinking Rerun - 2026-06-13

Work directory: `work/edit2` (`/mnt/xjh/data/arxiv_chart/work/edit2` through the `work` symlink).

## Change

Kimi thinking was regenerated from `answers_verified.jsonl` with the updated model and image policy:

- Model: `kimi-k2.6-qianli`
- Endpoint: `/v1/chat/completions`
- Protocol: streaming OpenAI-style chat completions
- Thinking: `extra_kwargs={"thinking": {"type": "enabled", "budget_tokens": 2048}}`
- `max_tokens=8192`
- `temperature=1`, `top_p=0.95`, `top_k=-1`
- Thinking image policy: `image_max_pixels=0`, which sends original images without resizing
- Main full-run concurrency: `workers=24`, `batch_size=192`

The old thinking/downstream outputs were backed up under `work/edit2/backups/thinking_qianli_20260613_020754`.

## Smoke Test

The first original-image qianli smoke test confirmed:

- `model`: `kimi-k2.6-qianli`
- `protocol`: `openai_chat_completions_streaming`
- `image_max_pixels`: `0`
- response contained `<think>...</think>`
- Gemini thinking judge passed after enabling the thinking budget and final-answer format repair

## Final Thinking Outputs

- `answers_verified.jsonl`: 4347 records
- `kimi_thinking_raw.jsonl`: 4347 records
- `kimi_thinking_verified.jsonl`: 4347 records
- API failures after retry: 0
- Gemini thinking judge failures after retry: 0
- All verified records use `kimi-k2.6-qianli`
- All verified records use `image_max_pixels=0`
- All verified records contain `<think>`
- `final_answer_appended=true`: 386 records

`final_answer_appended` means Kimi produced valid reasoning content but did not emit the required `Final answer:` line before stream completion; the pipeline appended the verified answer line before Gemini judging. Gemini still judged the reasoning against the image and answer.

## Downstream Rebuild

- `qa_thinking_sampled.jsonl`: 4347 records, `target=0` kept all verified QA/thinking records
- `dense_caption_verified.jsonl`: 373 records
- `merged.jsonl`: 2945 records requiring verified answer, verified thinking, and verified caption
- `review/review.html`: regenerated
- `review_static/review_static.html`: regenerated with 200 resized review-display images

Merged distribution:

- Task types:
  - `trend_pattern_analysis`: 370
  - `complex_calculation`: 370
  - `anomaly_extrema_detection`: 370
  - `multi_element_synthesis`: 369
  - `approximate_value_estimation`: 367
  - `cross_element_comparison`: 367
  - `hypothetical_reasoning`: 367
  - `fine_grained_visual_reading`: 365
- Answer types:
  - `numeric_approx`: 1370
  - `short_phrase`: 705
  - `choice`: 698
  - `trend_label`: 83
  - `numeric_exact`: 42
  - `boolean`: 26
  - `ranked_list`: 16
  - `integer`: 5
- Difficulty:
  - `medium`: 1924
  - `hard`: 1021

Quality report: `work/edit2/reports/quality_stats.json`.
