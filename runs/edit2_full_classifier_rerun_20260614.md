# Edition2 Full Classifier Rerun - 2026-06-14

Work directory: `work/edit2_full` (`/mnt/xjh/data/arxiv_chart/work/edit2_full` through the `work` symlink).

## Why

The previous edition2 run only classified a small pilot subset:

- `work/candidates_2020_2025.filtered.jsonl`: 3,583,603 records after local filtering.
- `work/candidates_2020_2025.chart_classified.jsonl`: 817 records classified by the old classifier.
- `work/edit2/filtered_charts_2020_2025.jsonl`: 553 records, derived from that small classified subset.

Therefore the 553-image edition2 run was complete only for the pilot-classified subset, not for the full 2020-2025 filtered pool.

## Current Full Classifier Run

The full classifier rerun starts from:

```text
work/candidates_2020_2025.filtered.jsonl
```

Output is sharded under:

```text
work/edit2_full/shards/classified_shard_*.jsonl
work/edit2_full/logs/classifier_failed_shard_*.jsonl
work/edit2_full/reports/classifier_report_shard_*.json
```

An earlier single-process smoke/full attempt wrote:

```text
work/edit2_full/candidates_2020_2025.chart_classified_v2.jsonl
```

The sharded jobs use this file as `--extra-done` so those candidate IDs are skipped.

## Runtime Configuration

Classifier script:

```bash
python3 scripts_v2/classify_charts_v2.py \
  --input work/candidates_2020_2025.filtered.jsonl \
  --out work/edit2_full/shards/classified_shard_${s}.jsonl \
  --failures work/edit2_full/logs/classifier_failed_shard_${s}.jsonl \
  --report work/edit2_full/reports/classifier_report_shard_${s}.json \
  --extra-done work/edit2_full/candidates_2020_2025.chart_classified_v2.jsonl \
  --workers 16 \
  --batch-size 128 \
  --max-pending 128 \
  --image-max-pixels 350000 \
  --status-every 1000 \
  --report-every 5000 \
  --num-shards 8 \
  --shard-index ${s}
```

The eight shard jobs are running inside one mounted `jlaunch` worker.

## Initial Observations

At roughly 10k classified shard records:

- Classified: 9,996
- Accepted: 6,723
- Weak accept: 1
- Failures: 128
- Accepted rate: about 67%
- Aggregate throughput: about 9 records/second

At this throughput, classifying all 3,583,603 locally filtered records is a multi-day run. The run is resumable because each shard skips IDs already present in its output and in the `--extra-done` file.

## Next Steps After Classifier Completes

1. Merge `work/edit2_full/candidates_2020_2025.chart_classified_v2.jsonl` and all `work/edit2_full/shards/classified_shard_*.jsonl`.
2. Run `scripts_v2/prepare_charts.py` on the merged classifier output.
3. Inspect accepted count before launching QA generation. If accepted count is very large, decide whether to run every accepted image or add a stricter local/chart-quality pass before question generation.
