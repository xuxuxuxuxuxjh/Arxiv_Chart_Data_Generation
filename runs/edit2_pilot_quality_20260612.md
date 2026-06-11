# Edition2 Pilot Quality Check - 2026-06-12

Input: `work_local/edit2_pilot`, 5 real chart images, every image attempted all 8 task types.

## Question Candidates

- Generated: 40 / 40 candidates.
- Images: 5.
- Task distribution: 5 each for `cross_element_comparison`, `approximate_value_estimation`, `fine_grained_visual_reading`, `multi_element_synthesis`, `trend_pattern_analysis`, `anomaly_extrema_detection`, `complex_calculation`, `hypothetical_reasoning`.
- Answer types: `numeric_approx` 22, `choice` 7, `short_phrase` 8, `numeric_exact` 1, `ranked_list` 1, `trend_label` 1.
- Difficulty: 37 hard, 3 medium.
- Rule/pattern audit: 0 title-only / axis-only / legend-only / generic "what is shown" / false multi-panel layout questions.

Example candidate shape:

- `complex_calculation`: Calculate the approximate average improvement in Robust Accuracy (%) of the `Ours + RO` method over the `RO (Baseline)` method across all perturbation budgets shown.
- `hypothetical_reasoning`: If a minimum robust accuracy of 35% is required, what is the approximate difference in the maximum allowable perturbation budget between two methods?
- `trend_pattern_analysis`: Compare the relative decrease over an x-range and decide which method declines more gradually.

Conclusion: the revised task set and prompt no longer collapse into simple OCR questions in this pilot.

## Judger Behavior

- Answer generation + Gemini answer judger: 40 / 40 verified.
- Kimi thinking + Gemini thinking judger: 37 / 40 verified.
- Caption generation + Gemini caption judger: 3 / 5 verified.
- Final merged records requiring answer, thinking, and caption: 9.

The answer judger is permissive in this small pilot because all generated answers were accepted. Manual spot checks did not show obvious bad accepts, so this is acceptable for continuing, but answer-judge pass rate should be monitored on the larger run.

The thinking and caption judgers are not too loose: rejected cases include wrong legend/color mapping, incorrect line-style identification, and hallucinated or misread numeric values while the final answer still matched. These are the failure modes we want filtered out.

## Follow-Up From Pilot

- Added `--group-tasks-per-image` to `generate_question_candidates.py` to keep every-image/every-task coverage while reducing API calls.
- Added `--retry-failed` to answer, thinking, and caption stages so raw-but-unverified records can be retried without deleting outputs.
- Use `--dedup-similar` during final QA/thinking sampling to reduce repeated near-equivalent questions from the same image.
