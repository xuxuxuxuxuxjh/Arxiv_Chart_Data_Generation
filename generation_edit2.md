## Edition1 Pipeline

本文档只说明 edition1 真实执行逻辑：如何 filter，如何生成 question/answer/thinking/caption，如何 verify，以及 prompt 和 question 类别如何设计。

代码目录：

```text
/home/i-xujiahao/arxiv_data/scripts
```

主工作目录：

```text
/home/i-xujiahao/arxiv_data/work -> /mnt/xjh/data/arxiv_chart/work
```

当前 pilot 是每组 500 张图：

```text
charxiv_inclusive_50k: 500 images
charxiv_exclusive_50k: 500 images
```

文件名里仍带 `_50k`，但当前已跑的是 pilot。

### 1. 总流程

```text
候选图扫描
  -> local filter
  -> Gemini chart classifier
  -> dedup
  -> inclusive/exclusive sampling
  -> Gemini 生成 question
  -> Gemini 生成 answer 并 consensus verify
  -> Kimi 生成 thinking 并 final-answer verify
  -> Kimi 生成 dense caption 并 JSON/empty/failure verify
  -> merge QA + thinking + caption
  -> review HTML 人工抽查
```

对应脚本：

```text
scan_arxiv_2020_2025_candidates.py
filter_candidates_local.py
classify_chart_candidates.py
dedup_chart_candidates.py
sample_50k_groups.py
generate_questions.py
answer_questions_gemini.py
generate_kimi_thinking_response.py
repair_kimi_pilot_outputs.py
generate_dense_caption.py
merge_qa_caption.py
generate_quality_and_review.py
export_static_review.py
```

### 2. 如何 Filter

filter 分三层：本地规则、Gemini chart classifier、去重采样。

#### 2.1 本地规则 filter

脚本：

```text
scripts/filter_candidates_local.py
```

输入：

```text
work/candidates_2020_2025.jsonl
```

输出：

```text
work/candidates_2020_2025.filtered.jsonl
work/reports/local_filter_report.json
```

命令：

```bash
python3 scripts/filter_candidates_local.py \
  --workers 24 \
  --chunk-size 4096
```

保留条件：

- image 文件必须存在。
- image 后缀必须在 `.png/.jpg/.jpeg/.webp/.bmp/.tif/.tiff` 中。
- 非 merged 图要求 `status=success`。
- caption 长度至少 10。
- short side 至少 256。
- area 至少 80000 pixels。
- aspect ratio 在 `[0.15, 8.0]`。
- 如果同一 figure 有 2-8 panel 的 merged 图，优先保留 merged，丢弃对应 single 图。

本地规则还会根据 caption、label、figure tex、reference paragraph 计算一个 `metadata_chart_score`，但这个分数只作为辅助特征，不直接决定最终接受。

#### 2.2 Gemini chart classifier

脚本：

```text
scripts/classify_chart_candidates.py
```

输入：

```text
work/candidates_2020_2025.filtered.jsonl
```

输出：

```text
work/candidates_2020_2025.chart_classified.jsonl
work/reports/classifier_report.json
work/logs/classifier_failed.jsonl
```

命令：

```bash
python3 scripts/classify_chart_candidates.py \
  --workers 4 \
  --batch-size 50
```

模型配置：

```text
model: gemini-3.5-flash
maxOutputTokens: 4096
temperature: 0
topP: 1
reasoning_effort: low
```

classifier prompt 设计：

```text
你是 arXiv figure filter，要判断图像是否是真实 chart/plot，
是否适合 visual question answering 和 dense captioning。

保留：
- line chart
- bar chart
- scatter plot
- histogram
- heatmap
- confusion matrix
- box/violin plot
- area chart
- ROC/PR/calibration curve
- ablation curve
- multi-panel chart，只要大多数 panel 是 chart

拒绝：
- model architecture diagram
- flowchart / pipeline / framework diagram
- algorithm diagram
- pure formula screenshot
- table screenshot
- code/UI screenshot
- natural/medical/remote-sensing/microscopy image
- qualitative result examples，除非本身是 chart/heatmap/confusion matrix

要求模型返回 strict JSON：
{
  "is_real_chart": true,
  "chart_confidence": 0.93,
  "chart_types": ["line_chart"],
  "non_chart_reason": null,
  "is_diagram": false,
  "is_table_screenshot": false,
  "is_photo_or_qualitative_image": false,
  "is_multi_panel": true,
  "text_readability": "good",
  "has_axes_or_scale": true,
  "has_legend_or_series_labels": true,
  "has_numeric_values": true,
  "suitable_for_vqa": true,
  "suitable_for_dense_caption": true,
  "risk_notes": []
}
```

接受条件：

```text
is_real_chart == true
chart_confidence >= 0.75
suitable_for_vqa == true
suitable_for_dense_caption == true
is_diagram == false
is_table_screenshot == false
is_photo_or_qualitative_image == false
```

`0.60 <= chart_confidence < 0.75` 会标成 `weak_accept`，但默认不进入后续数据。

#### 2.3 Dedup

脚本：

```text
scripts/dedup_chart_candidates.py
```

命令：

```bash
python3 scripts/dedup_chart_candidates.py
```

输出：

```text
work/candidates_2020_2025.deduped.jsonl
work/logs/duplicate_chart_candidates.jsonl
work/reports/dedup_report.json
```

去重规则：

- 默认只保留 `classifier.accepted=true`。
- 同一个 `(paper_id, figure_index, image_kind)` 只保留一个。
- 计算 image pHash，exact pHash duplicate 只保留一个。
- 保留时优先选择 `chart_confidence` 更高的样本。

#### 2.4 Sampling

脚本：

```text
scripts/sample_50k_groups.py
```

pilot 命令：

```bash
python3 scripts/sample_50k_groups.py \
  --target 500 \
  --max-images-per-paper 2 \
  --seed 20260611
```

输出：

```text
work/sample_charxiv_inclusive_50k.jsonl
work/sample_charxiv_exclusive_50k.jsonl
work/reports/sampling_report.md
```

采样逻辑：

- `inclusive`: 允许 CharXiv paper，优先加入 CharXiv paper 图。
- `exclusive`: 排除所有 CharXiv paper。
- 每篇 paper 最多 2 张图。
- 按年份 2020-2025 尽量均衡。
- 按月份设上限，避免某几个月过密。
- 给每条样本加 `sample_meta.chart_type_bucket`。

采样阶段的 chart bucket：

```text
line_chart
bar_chart
scatter_plot
heatmap_confusion_matrix
histogram_density_distribution
box_violin_area
multi_panel_chart
other
```

### 3. 如何生成 Question

脚本：

```text
scripts/generate_questions.py
```

输入：

```text
work/sample_charxiv_inclusive_50k.jsonl
work/sample_charxiv_exclusive_50k.jsonl
```

输出：

```text
work/qa/charxiv_inclusive_50k.questions.jsonl
work/qa/charxiv_exclusive_50k.questions.jsonl
work/logs/question_generation_failures.jsonl
```

命令：

```bash
python3 scripts/generate_questions.py \
  --input work/sample_charxiv_inclusive_50k.jsonl \
  --group inclusive \
  --out work/qa/charxiv_inclusive_50k.questions.jsonl \
  --limit 500 \
  --workers 3

python3 scripts/generate_questions.py \
  --input work/sample_charxiv_exclusive_50k.jsonl \
  --group exclusive \
  --out work/qa/charxiv_exclusive_50k.questions.jsonl \
  --limit 500 \
  --workers 3
```

模型配置：

```text
model: gemini-3.5-flash
maxOutputTokens: 64000
temperature: 1
topP: 0.95
reasoning_effort: high
timeout: 240
retries: 2
```

#### 3.1 Question prompt 设计

核心 prompt：

```text
Generate exactly one image-only question for this chart.

The question must be answerable from the visible chart image alone.
Do not require paper background, the full caption, hidden data, or external knowledge.

Target task type: {task_type}

Allowed task types:
- descriptive_extraction: title, axis label, legend text, colorbar label, subplot label.
- text_in_chart: visible method/category/panel/series label.
- number_in_chart: visible or approximate number only when readable.
- visual_comparison: highest, lowest, larger, smaller, better, worse.
- trend_reasoning: increasing, decreasing, stable, crossing, convergence.
- counting: number of visible bars, curves, panels, clusters, or reliably countable elements.

First version rules:
- evidence_source must be "image_only".
- requires_caption_context must be false.
- question_risk should be "low" only if the answer is visually reliable.
- Do not generate unanswerable or uncertain questions.
- Do not include the answer.

Return strict JSON only:
{
  "question": "...",
  "task_type": "{task_type}",
  "answer_type": "short_text",
  "evidence_source": "image_only",
  "difficulty": "medium",
  "requires_exact_reading": false,
  "requires_caption_context": false,
  "question_risk": "low"
}

Caption LaTeX is auxiliary context for terminology only;
the question itself must be answerable without reading this caption outside the image.
```

输入给 Gemini 的内容包括：

- chart image。
- 上面的 question prompt。
- `classifier` 输出。
- 截断到 5000 字符的 `caption_latex`，只用于术语辅助。

#### 3.2 Question 类别

edition1 共有 6 类 question：

| 类别 | 目标 |
| --- | --- |
| `descriptive_extraction` | 读取 title、axis label、legend text、colorbar label、subplot label |
| `text_in_chart` | 读取图中可见的 method/category/panel/series label |
| `number_in_chart` | 读取或估计可见数值 |
| `visual_comparison` | 比较 highest/lowest/larger/smaller/better/worse |
| `trend_reasoning` | 判断 increasing/decreasing/stable/crossing/convergence |
| `counting` | 数 bar、curve、panel、cluster 等可可靠计数元素 |

当前 task sequence 配比：

```text
descriptive_extraction: 20
text_in_chart:          15
number_in_chart:        15
visual_comparison:      20
trend_reasoning:        15
counting:               10
```

注意：当前模板固定 `answer_type="short_text"`，这是 edition1 question 偏简单、数值题很少的主要问题之一。

### 4. 如何生成 Answer

脚本：

```text
scripts/answer_questions_gemini.py
```

输入：

```text
work/qa/charxiv_inclusive_50k.questions.jsonl
work/qa/charxiv_exclusive_50k.questions.jsonl
```

输出：

```text
work/qa/charxiv_inclusive_50k.consensus.jsonl
work/qa/charxiv_exclusive_50k.consensus.jsonl
work/logs/answer_consensus_failures.jsonl
```

命令：

```bash
python3 scripts/answer_questions_gemini.py \
  --input work/qa/charxiv_inclusive_50k.questions.jsonl \
  --out work/qa/charxiv_inclusive_50k.consensus.jsonl \
  --workers 3 \
  --batch-size 20

python3 scripts/answer_questions_gemini.py \
  --input work/qa/charxiv_exclusive_50k.questions.jsonl \
  --out work/qa/charxiv_exclusive_50k.consensus.jsonl \
  --workers 3 \
  --batch-size 20
```

answer prompt：

```text
Look only at the chart image.
Answer the question with the shortest correct answer.
Do not explain.
If the exact value is unreadable, answer with an approximate value only when the question allows approximation.
If the question cannot be answered from the image alone, answer "Not answerable from the image alone".

Question: {question}
```

模型配置：

```text
model: gemini-3.5-flash
maxOutputTokens: 64000
temperature: 1
topP: 0.95
reasoning_effort: high
timeout: 240
retries: 2
```

每个 question 独立回答 3 次。

### 5. 如何 Verify Answer

answer verify 使用 Gemini 三次回答的 consensus。

逻辑在：

```text
scripts/answer_questions_gemini.py
scripts/pipeline_common.py
```

verify 步骤：

1. 对同一图和同一问题调用 Gemini 3 次。
2. 从输出中提取短答案。
3. 根据 `answer_type` 做 normalize。
4. 如果三次 normalized answer 完全一致，则通过。
5. 如果 `answer_type == numeric_approx`，允许数值容差。
6. 不通过则写入 `work/logs/answer_consensus_failures.jsonl`，不进入 `.consensus.jsonl`。

normalize 规则包括：

- lower case。
- unicode NFKC。
- 去标点。
- 去 `the/a/an`。
- 合并空格。
- numeric 类型会解析第一个数字。
- yes/no 类型归一成 `yes`/`no`。

当前 pilot consensus 结果：

```text
inclusive: 348 / 497
exclusive: 326 / 500
```

### 6. 如何生成 Thinking

thinking 分两类输出：

- `qa_direct`: user 是 `<image>\nquestion`，assistant 是 consensus answer。
- `qa_thinking`: user 是 `<image>\nquestion`，assistant 是 Kimi thinking response。

初版脚本：

```text
scripts/generate_kimi_thinking_response.py
```

但初版路径不再作为正式路径使用，因为它用：

```text
/v1/chat/completions
model: kimi-k2.6-aliyun2kimi
max_tokens: 250000
原始大图
```

这个组合慢、容易 429，并且失败时会 fallback 成只有 answer，没有真正 thinking。

当前正式脚本：

```text
scripts/repair_kimi_pilot_outputs.py
```

当前命令：

```bash
python3 scripts/repair_kimi_pilot_outputs.py \
  --kind thinking \
  --group inclusive \
  --provider messages \
  --workers 8 \
  --batch-size 8 \
  --image-max-pixels 100000 \
  --max-tokens 8192 \
  --retries 1 \
  --timeout 120

python3 scripts/repair_kimi_pilot_outputs.py \
  --kind thinking \
  --group exclusive \
  --provider messages \
  --workers 8 \
  --batch-size 8 \
  --image-max-pixels 100000 \
  --max-tokens 8192 \
  --retries 1 \
  --timeout 120
```

当前 Kimi 配置：

```text
endpoint: /v1/messages
model: kimi-k2.6-aliyun
max_tokens: 8192
image_max_pixels: 100000
workers: 8
batch_size: 8
```

thinking prompt：

```text
Reason from visible chart evidence only, then end with:
Final answer: {answer}

Do not use paper background or caption-only claims.
The final answer must exactly be the given consensus answer.

Question: {question}
Consensus answer: {answer}
Task type: {task_type}
```

输入给 Kimi：

- resize 后的 chart image，保证 `H*W <= 100000`。
- thinking prompt。
- consensus answer。
- task type。

输出字段：

```json
{
  "thinking_response": {
    "model": "kimi-k2.6-aliyun",
    "model_config": {
      "max_tokens": 8192,
      "image_max_pixels": 100000,
      "provider": "messages"
    },
    "response": "... Final answer: ...",
    "final_answer_matches_consensus": true,
    "repair_run": true
  },
  "thinking_response_failed": false
}
```

### 7. 如何 Verify Thinking

thinking verify 不是额外 LLM judge，而是 deterministic check。

逻辑：

1. 从 Kimi response 中解析 `Final answer: ...`。
2. 对 parsed final answer 和 consensus answer 做同样的 `normalize_answer`。
3. 如果一致，则：

```text
thinking_response.final_answer_matches_consensus = true
thinking_response_failed = false
```

4. 如果不一致或 API 失败，则标记失败。
5. `repair_kimi_pilot_outputs.py` 会只重跑缺失或失败项，并重写完整 `.qa_thinking.jsonl`。

当前 pilot 修复后：

```text
work/qa/charxiv_inclusive_50k.qa_thinking.jsonl: 348, failed_after=0
work/qa/charxiv_exclusive_50k.qa_thinking.jsonl: 326, failed_after=0
```

检查命令：

```bash
python3 - <<'PY'
import json
from pathlib import Path

for path in [
    Path("work/qa/charxiv_inclusive_50k.qa_thinking.jsonl"),
    Path("work/qa/charxiv_exclusive_50k.qa_thinking.jsonl"),
]:
    total = failed = empty = 0
    for line in path.open():
        if not line.strip():
            continue
        total += 1
        r = json.loads(line)
        failed += int(bool(r.get("thinking_response_failed")))
        empty += int(not ((r.get("thinking_response") or {}).get("response") or "").strip())
    print(path, "total=", total, "failed=", failed, "empty=", empty)
PY
```

### 8. 如何生成 Caption

caption 是 dense caption，不是 QA answer。

初版脚本：

```text
scripts/generate_dense_caption.py
```

同样，初版脚本使用 `/v1/chat/completions + max_tokens=250000 + 原图`，不再作为最终生成方式。

当前使用：

```text
scripts/repair_kimi_pilot_outputs.py --kind caption
```

命令：

```bash
python3 scripts/repair_kimi_pilot_outputs.py \
  --kind caption \
  --group inclusive \
  --provider messages \
  --workers 8 \
  --batch-size 8 \
  --image-max-pixels 100000 \
  --max-tokens 8192 \
  --retries 1 \
  --timeout 120

python3 scripts/repair_kimi_pilot_outputs.py \
  --kind caption \
  --group exclusive \
  --provider messages \
  --workers 8 \
  --batch-size 8 \
  --image-max-pixels 100000 \
  --max-tokens 8192 \
  --retries 1 \
  --timeout 120
```

caption prompt：

```text
Describe this chart in detail using only visible information.

Requirements:
- Use natural language grounded in the image.
- Include chart type, axes, legend/series, visual encodings, major trends/comparisons, and multi-panel layout when visible.
- Do not invent paper methods, dataset facts, or conclusions that are not visible in the image.
- If text is not legible, say that some labels are not legible.
- The dense_caption must be at least 2 sentences and at most 180 words.

Return strict JSON only:
{
  "dense_caption": "...",
  "visible_elements": {
    "chart_types": [],
    "axes": [],
    "series_or_panels": [],
    "main_trends": []
  },
  "uncertainty": []
}

Caption LaTeX for terminology only:
{caption_latex}
```

输入给 Kimi：

- resize 后的 chart image，保证 `H*W <= 100000`。
- caption prompt。
- 截断到 5000 字符的 `caption_latex`，只用于术语辅助。

输出：

```text
work/dense_caption/charxiv_inclusive_50k.dense_caption.jsonl
work/dense_caption/charxiv_exclusive_50k.dense_caption.jsonl
```

### 9. 如何 Verify Caption

当前 edition1 没有单独的 caption LLM verifier。caption verify 分三层：

#### 9.1 JSON parse verify

Kimi 必须返回 strict JSON，并且能被 `extract_json_object` 解析。

必须包含：

```text
dense_caption
visible_elements
uncertainty
```

如果 JSON parse 失败，会进入 fallback 或 repair target。

#### 9.2 Failure flag verify

脚本检查：

- `quality.generation_failed` 是否为 true。
- `dense_caption` 是否是 fallback 文案。
- caption 是否为空。

失败项会被 `repair_kimi_pilot_outputs.py --kind caption` 重跑。

当前 pilot 修复后：

```text
work/dense_caption/charxiv_inclusive_50k.dense_caption.jsonl: 500, failed_after=0
work/dense_caption/charxiv_exclusive_50k.dense_caption.jsonl: 500, failed_after=0
```

#### 9.3 Review HTML 人工 verify

生成 review：

```bash
python3 scripts/generate_quality_and_review.py \
  --work work \
  --limit 200
```

导出静态 HTML：

```bash
python3 scripts/export_static_review.py \
  --html work/review/pilot_review.html \
  --out /home/i-xujiahao/arxiv_data/review_export \
  --max-side 1600 \
  --max-pixels 100000
```

打开：

```text
http://127.0.0.1:8898/pilot_review_static.html
```

review 页面展示：

- image
- caption LaTeX
- generated question
- Gemini 3-run answers
- Kimi thinking
- dense caption

当前静态 review 只展示同时有 question、answer、thinking、caption 的完整样本。

### 10. 如何 Merge

脚本：

```text
scripts/merge_qa_caption.py
```

命令：

```bash
python3 scripts/merge_qa_caption.py --group both
python3 scripts/merge_qa_caption.py --group both --qa-only
```

输出：

```text
work/merged/charxiv_inclusive_50k.qa_caption.jsonl
work/merged/charxiv_exclusive_50k.qa_caption.jsonl
work/merged/charxiv_inclusive_50k.qa_caption_qa_only.jsonl
work/merged/charxiv_exclusive_50k.qa_caption_qa_only.jsonl
```

merge 逻辑：

- 按 `source.candidate_id` 对齐 sample、qa_direct、qa_thinking、dense_caption。
- `qa_caption` 保留所有 sample 图片；没有 QA 的样本 QA 字段为空。
- `qa_caption_qa_only` 只保留有 consensus QA 和 thinking 的样本。
- 顶层 `messages` 包含三路：

```text
messages.qa_direct
messages.qa_thinking
messages.dense_caption
```

当前 pilot merge 结果：

```text
inclusive qa_caption: 500, missing_qa=152
exclusive qa_caption: 500, missing_qa=174
inclusive qa_caption_qa_only: 348
exclusive qa_caption_qa_only: 326
```

### 11. 当前 Pilot 统计

```text
questions:
  inclusive 497
  exclusive 500

consensus QA:
  inclusive 348
  exclusive 326

qa_direct:
  inclusive 348
  exclusive 326

qa_thinking:
  inclusive 348, failed=0 after repair
  exclusive 326, failed=0 after repair

dense_caption:
  inclusive 500, failed=0 after repair
  exclusive 500, failed=0 after repair

merged:
  inclusive qa_caption 500
  exclusive qa_caption 500
  inclusive qa_caption_qa_only 348
  exclusive qa_caption_qa_only 326
```

### 12. Edition1 已知问题

当前 question 质量偏简单，主要原因：

- `descriptive_extraction` 和 `text_in_chart` 占比较高。
- prompt 强调 image-only、low risk、visually reliable，模型倾向生成读标签/读标题/读图例问题。
- JSON 模板固定 `answer_type="short_text"`，导致数值题和结构化答案很少。
- answer verify 用三次 exact consensus，会进一步保留简单问题，过滤掉 trend/numeric/comparison 里更难的样本。

当前 question 类别里最需要改的是：

- 降低 `descriptive_extraction`。
- 减少单纯 axis/title/legend OCR。
- 增加跨 panel comparison。
- 增加 numeric estimation。
- 增加 multi-step visual reasoning。
- 对不同 task type 设置不同 answer type。

edition2 类别：

```text
cross_panel_comparison:      25%
numeric_estimation:          20%
trend_reasoning:             20%
multi_step_visual_reasoning: 20%
structure_counting:          10%
text_extraction_max:          5%
```

edition2 prompt 应明确禁止：

```text
Do not ask for chart title only.
Do not ask for x-axis/y-axis label only.
Do not ask for legend text only.
Do not ask what a single color represents unless the question also requires comparison or reasoning.
Do not ask for subplot labels such as (a), (b), top, bottom only.
Do not ask generic "what is shown/plotted" questions.
```

## Edition2 scripts_v2 Pipeline

本节记录 `/home/i-xujiahao/arxiv_data/scripts_v2` 的当前实现，不是待办清单。目标是对 2020-2025 年通过 chart filter 的图片生成 QA / thinking / caption，并修复 edition1 的问题：题目偏简单、answer_type 单一、consensus 偏向简单题、Kimi thinking fallback 不透明。

当前代码入口：

```text
scripts_v2/common_v2.py
scripts_v2/classify_charts_v2.py
scripts_v2/prepare_charts.py
scripts_v2/generate_question_candidates.py
scripts_v2/generate_and_verify_answers.py
scripts_v2/generate_and_verify_thinking.py
scripts_v2/generate_and_verify_captions.py
scripts_v2/sample_verified_questions.py
scripts_v2/merge_verified_outputs.py
scripts_v2/generate_review_v2.py
scripts_v2/export_static_review_v2.py
scripts_v2/build_merged_review_static.py
```

默认工作目录由 `common_v2.py` 控制：

```text
ARXIV_CHART_WORK  default: /home/i-xujiahao/arxiv_data/work
ARXIV_CHART_EDIT2 default: /home/i-xujiahao/arxiv_data/work/edit2
GEMINI_MODEL      default: gemini-3.5-flash
KIMI_MODEL        default: kimi-k2.6-qianli
KIMI_ENDPOINT     default: https://models-proxy.stepfun-inc.com/v1/chat/completions
```

### 1. 固定 Chart Type 集合

- [x] 在 classifier prompt 中定义闭集 `chart_types`，要求模型只能从集合中选择，不能自由发挥。
- [x] 把 layout 信息和 chart type 分开：`is_multi_panel`、`panel_count`、`panel_layout` 不作为 chart type。
- [x] 当前 chart type 集合定义在 `common_v2.CHART_TYPES`：

```text
line_chart
bar_chart
scatter_plot
histogram
density_plot
heatmap
confusion_matrix
box_plot
violin_plot
area_chart
roc_curve
pr_curve
calibration_curve
ablation_curve
error_bar_plot
contour_plot
surface_plot
matrix_plot
other_chart
```

- [x] 在 `scripts_v2/classify_charts_v2.py` 中重写 classifier prompt 和 JSON schema。
- [x] classifier 输出中保留：

```json
{
  "chart_types": ["line_chart"],
  "primary_chart_type": "line_chart",
  "is_multi_panel": true,
  "panel_count": 4,
  "text_readability": "good",
  "has_axes_or_scale": true,
  "has_legend_or_series_labels": true,
  "has_numeric_values": true
}
```

### 2. 取消 Sampling 限制

- [x] 第二版先不做 sampling quota。
- [x] 后续处理输入改成全部通过 filter + classifier + dedup 的 2020-2025 chart 图片。
- [x] 不限制每篇 paper 的图片上限。
- [x] 不限制月份上限。
- [x] 不限制年份配比，只要求年份在 2020-2025。
- [x] 保留 `is_charxiv_paper` 标记，但不再优先纳入或排除。
- [x] 新增统一输入：

```text
work/candidates_2020_2025.deduped.jsonl
```

或显式导出：

```text
work/edit2/filtered_charts_2020_2025.jsonl
```

### 3. 重新定义 Question Task Type

- [x] 去掉 edition1 的 `descriptive_extraction`。
- [x] 单纯 title / axis label / legend OCR 题不再作为独立任务。
- [x] 参考 CharXiv Reasoning、ChartQAPro Reasoning、Hypothetical，定义更细的 task type。
- [x] 生成阶段不要先按比例抽 task；每张图都尝试生成每个 task type 的 question，形成 question candidate pool。
- [x] 比例只在全部生成、answer verify、thinking verify 后的最终筛选/采样阶段生效。
- [x] 当前 task type 权重定义在 `common_v2.TASK_SPECS`：

```text
cross_element_comparison        18%
approximate_value_estimation    15%
fine_grained_visual_reading     10%
multi_element_synthesis         15%
trend_pattern_analysis          15%
anomaly_extrema_detection       10%
complex_calculation             10%
hypothetical_reasoning           7%
```

含义：

```text
cross_element_comparison:
  比较不同 curve/bar/panel/category 的大小、排名、差异或相对变化。

approximate_value_estimation:
  从 axis/colorbar/bar/curve/scatter 中估计近似数值，允许合理误差。

fine_grained_visual_reading:
  需要定位特定 panel、series、x-range 或 condition 后读取细节，不允许只是读标题/轴名。

multi_element_synthesis:
  综合 legend、axis、panel、series、颜色、形状等多个视觉元素得到答案。

trend_pattern_analysis:
  判断 increasing/decreasing/stable/non-monotonic/crossing/convergence/divergence/saturation。

anomaly_extrema_detection:
  找 peak、valley、outlier、异常点、最大最小值、突变位置。

complex_calculation:
  做差值、比例、倍数、平均、总和、排序、变化率等计算。

hypothetical_reasoning:
  在给定假设下做推断或估计，例如 if threshold changes / if x increases / if one method is removed。
```

- [x] 定义 `answer_type` 闭集，不能全部默认 `short_text`：

```text
numeric_exact
numeric_approx
integer
choice
ranked_list
trend_label
boolean
short_phrase
```

### 4. 重写 Question Generation Prompt

- [x] 新增 `scripts_v2/generate_question_candidates.py`。
- [x] 对每张 chart image 遍历全部 task type，每个 task type 尝试生成 1 条 question candidate。
- [x] 失败、低价值、不可解析 question 不进入 candidate pool，可按 task type retry。
- [x] prompt 必须强制：

```text
Question must require reasoning over visible chart content.
Do not ask title-only, axis-label-only, legend-only, subplot-label-only questions.
Text reading is allowed only when it is one step in a multi-step reasoning question.
Prefer medium/hard questions.
The answer must be verifiable from the image alone.
```

- [x] 输出 strict JSON schema：

```json
{
  "question": "...",
  "task_type": "complex_calculation",
  "answer_type": "numeric_approx",
  "difficulty": "hard",
  "requires_exact_reading": false,
  "requires_caption_context": false,
  "reasoning_steps_required": 2,
  "visual_elements_required": ["legend", "x_axis", "curve", "panel"],
  "risk_notes": []
}
```

- [x] 增加 rule-based reject：如果 question 命中 title-only / axis-only / legend-only 模板，直接 retry。
- [x] 输出文件应是候选池，例如：

```text
work/edit2/question_candidates.jsonl
```

- [x] 每条候选包含 `candidate_id + task_type`，后续按 verify 结果和目标比例再采样。

### 5. Gemini 生成 Answer：三次独立生成，先 Think 后 Answer

- [x] 新增 `scripts_v2/generate_and_verify_answers.py`。
- [x] 每个 question 调用 Gemini 生成 3 次 answer。
- [x] 每次 answer prompt 都要求 Gemini 先推理再给最终答案。
- [x] 每次当前输出格式：

```text
<think>
Use only visible chart evidence. Locate the relevant panels/series/axes, then reason step by step.
</think>
Final answer: ...
```

- [x] 未采用 JSON 作为主输出；正式实现采用上面的 think/final 格式，extractor 同时兼容 JSON：

```json
{
  "thinking": "...",
  "final_answer": "...",
  "confidence": "high",
  "evidence": ["..."]
}
```

- [x] `answer_generation.samples` 保留 3 次原始输出、thinking、final answer、normalized answer。
- [x] 通过一致性 verify 后，`answer_generation.final_answer` 和顶层 `answer` 使用 consistency judge 选出的 canonical answer。

### 6. 写 Answer Extractor 和 Retry

- [x] 新增通用 extractor，例如：

```text
extract_thinking_and_final_answer(text)
extract_json_with_final_answer(text)
```

- [x] 支持解析：

```text
<think>...</think>
Final answer: ...
```

和 strict JSON：

```json
{"thinking": "...", "final_answer": "..."}
```

- [x] extractor 失败时自动 retry。
- [x] 任意一次 answer generation 的 extractor retry 后仍失败，则整道题放弃，写入：

```text
work/logs/answer_extraction_failures.jsonl
```

- [x] 不允许 extractor 失败的样本进入 verified QA。

### 7. Answer Verify：三次 Answer 一致性 Gemini Judger

- [x] Answer Verify 不判断图像答案是否正确，只判断 3 次 extractor 后的 final answer 是否一致。
- [x] 新增 Gemini consistency judger，对同一 question 的 3 个 extracted final answers 做一致性判断。
- [x] consistency judger 不输入 image。
- [x] judger prompt 输入：

```text
question
task_type
answer_type
extracted final answer 1
extracted final answer 2
extracted final answer 3
```

- [x] judger 输出 strict JSON：

```json
{
  "verdict": "consistent",
  "all_answers_consistent": true,
  "canonical_answer": "...",
  "canonical_answer_index": 1,
  "normalized_answers": ["...", "...", "..."],
  "normalized_answer": "...",
  "reason": "..."
}
```

- [x] 通过条件：

```text
verdict == consistent
all_answers_consistent == true
canonical_answer 非空
```

- [x] 如果 3 次 answer 不一致，直接写入 `work/logs/answer_judge_failures.jsonl`，不进入 verified QA。
- [x] 如果 consistency judger 解析失败，按 `--judge-retries` 重试；仍失败则写入 `work/logs/answer_judge_failures.jsonl`，不进入 verified QA。
- [x] 通过一致性判断的记录写入：

```text
work/edit2/answers_verified.jsonl
```

### 8. Kimi Thinking Generation

- [x] thinking generation 由 `scripts_v2/generate_and_verify_thinking.py` 调用 `common_v2.kimi_messages_generate`。
- [x] 使用 `/v1/chat/completions`，开启 streaming，收集 `reasoning_content` 为 `<think>...</think>`。
- [x] 使用 `kimi-k2.6-qianli`。
- [x] `max_tokens=64000`。
- [x] thinking 阶段不 resize 图片，直接发送原图：

```text
image_max_pixels = 0
```

- [x] Kimi thinking 参数：

```text
temperature=1
top_p=0.95
top_k=-1
extra_kwargs={"thinking": {"type": "enabled"}}
```

- [x] `generate_and_verify_thinking.py` 默认参数：

```text
workers=8
batch_size=8
timeout=300
retries=1
```

Kimi thinking throughput test 可以加 `--skip-judge`，只统计 Kimi 生成耗时，不混入 Gemini thinking judge 耗时。

- [x] thinking prompt 输入 verified answer，而不是未验证 answer。
- [x] prompt 要求：

```text
Reason only from visible chart evidence.
Use the verified answer as the required final answer.
End with exactly:
Final answer: {verified_answer}
```

- [x] 失败样本不能 fallback 成 answer-only messages；必须保留失败标记并进入 retry log。

### 9. Verify Kimi Thinking：Gemini Check

- [x] 新增 Gemini thinking judger。
- [x] 输入：

```text
image
question
verified answer
Kimi thinking response
Kimi final answer
```

- [x] 检查两件事：

```text
1. Kimi final answer 是否和 verified answer 一致。
2. Kimi thinking 是否 grounded in image，是否支持该 answer。
```

- [x] judger 输出 strict JSON：

```json
{
  "verdict": "pass",
  "final_answer_matches": true,
  "reasoning_grounded": true,
  "has_contradiction": false,
  "reason": "..."
}
```

- [x] 通过条件：

```text
verdict == pass
final_answer_matches == true
reasoning_grounded == true
has_contradiction == false
```

- [x] 不通过则 retry Kimi thinking；仍不通过则写入：

```text
work/logs/kimi_thinking_judge_failures.jsonl
```

### 10. Caption Generation，无单独 Verify

- [x] caption 使用 `common_v2.kimi_messages_generate`，实际 transport 是 `/v1/chat/completions` streaming。
- [x] `generate_and_verify_captions.py` 默认 `image_max_pixels=1000000`。
- [x] `max_tokens=8192`。
- [x] caption prompt 输入 image + `caption_latex`。
- [x] `caption_latex` 用于保留领域术语、方法名、变量名、panel 描述和 series 名称。
- [x] 不再调用 Gemini caption judger；Kimi 成功生成非空 caption 后直接进入 `dense_caption_verified.jsonl`。
- [x] `caption_verified=true` 在当前实现中表示 caption 生成成功，不表示通过额外 Gemini verify。
- [x] caption 输出 strict JSON：

```json
{
  "dense_caption": "...",
  "visible_elements": {
    "chart_types": [],
    "axes": [],
    "series_or_panels": [],
    "main_trends": []
  },
  "uncertainty": []
}
```

- [x] 如果 Kimi caption 为空、失败或 retry 后仍无法生成，则写入：

```text
work/logs/caption_failures.jsonl
```

### 11. 输出文件

- [x] 第二版默认输出目录：

```text
work/edit2/
  filtered_charts_2020_2025.jsonl
  question_candidates.jsonl
  answers_raw.jsonl
  answers_verified.jsonl
  kimi_thinking_raw.jsonl
  kimi_thinking_verified.jsonl
  dense_caption_raw.jsonl
  dense_caption_verified.jsonl
  merged.jsonl
  reports/
  logs/
```

- [x] 10 图端到端验证输出目录：

```text
/home/i-xujiahao/arxiv_data/work_local/edit2_10_full/
  filtered_charts_2020_2025.jsonl
  question_candidates.jsonl
  answers_raw.jsonl
  answers_verified.jsonl
  kimi_thinking_raw.jsonl
  kimi_thinking_verified.jsonl
  dense_caption_raw.jsonl
  dense_caption_verified.jsonl
  qa_thinking_sampled.jsonl
  merged.jsonl
  reports/
  logs/
```

- [x] 10 图静态 review 输出目录：

```text
/home/i-xujiahao/arxiv_data/edit2_10_full_review_static/
  index.html
  records.jsonl
  assets/
```

- [x] Jupyter 可访问副本：

```text
/data/jupyter/arxiv_chart_review/
  index.html
  records.jsonl
  assets/
```

- [x] merged 文件只收 verified 样本：

```text
answer_judged == true
thinking_judged == true
caption_generated == true
```

- [x] 原始失败样本全部保留在 logs，不伪装成成功 messages。

### 12. 当前 10 图端到端验证结果

执行脚本：

```text
scripts_v2/run_edit2_10_full_pipeline.sh
scripts_v2/run_edit2_10_thinking_shards.sh
scripts_v2/build_merged_review_static.py
```

最终产物：

```text
/home/i-xujiahao/arxiv_data/work_local/edit2_10_full/merged.jsonl
/home/i-xujiahao/arxiv_data/edit2_10_full_review_static/index.html
/home/i-xujiahao/arxiv_data/edit2_10_full_review_static/records.jsonl
```

计数：

```text
filtered_charts_2020_2025.jsonl: 10
question_candidates.jsonl:       80
answers_verified.jsonl:          76
kimi_thinking_verified.jsonl:    76
dense_caption_verified.jsonl:    10
qa_thinking_sampled.jsonl:       76
merged.jsonl:                    76
unique_images_in_merged:         10
```

merged 中每条记录同时满足：

```text
verified.answer == true
verified.thinking == true
verified.caption == true
```
