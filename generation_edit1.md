# Real arXiv Chart Data Generation Plan

本文档是当前迭代版方案，只处理 **真实 arXiv chart 图**，不处理合成图，不处理普通论文 figure，不直接使用公开 benchmark 的 QA 作为训练数据。

目标：快速生成两组 2020-2025 arXiv chart 数据，用于 chart understanding / reasoning SFT：

1. `arxiv_chart_50k_charxiv_inclusive`
   - 允许包含 CharXiv 涉及到的 paper。
   - 采样时优先纳入能在当前 arXiv 抽图目录中匹配到的 CharXiv paper 图，剩余额度用同分布真实 arXiv chart 补齐。
   - 所有样本显式打 `is_charxiv_paper=true/false`。

2. `arxiv_chart_50k_charxiv_exclusive`
   - 严格排除 CharXiv 涉及到的 paper。
   - 后续作为 contamination-safe 训练版本，避免把 CharXiv benchmark 相关论文图混入训练。

第一版默认解释为：**每组 50k chart images，每张图生成 1 条 QA 和 1 条 dense-caption**。因此每组会输出：

```text
50k QA records
50k image dense-caption records
```

如果成本或吞吐压力太大，可以先把 `50k` 改成 `5k pilot` 跑通全链路，再放大。

## 1. 已确认的数据位置与结构

挂载脚本：

```bash
/home/i-xujiahao/test4.sh
```

目标数据目录：

```text
/mnt/lvhaoran-jfs/cyy/arxiv_fig_extract_0608_addmerge
```

目录结构已确认：

```text
arxiv_fig_extract_0608_addmerge/
  arxiv_2001/
    arXiv_src_2001_001.tar/
      2001.00003/
        2001.00003_extracted_figs/
          extraction_summary.json
          fig_0001_img_01_xxx.json
          fig_0001_img_01_xxx.png
        2001.00003_extracted_figs_merged/
          fig_0002_merged_3x1_03panels.json
          fig_0002_merged_3x1_03panels.png
        2001.00003_total_tex/
          2001.00003_total.tex
```

2020-2025 bucket 已确认存在：

```text
arxiv_2001 ... arxiv_2012
arxiv_2101 ... arxiv_2112
arxiv_2201 ... arxiv_2212
arxiv_2301 ... arxiv_2312
arxiv_2401 ... arxiv_2412
arxiv_2501 ... arxiv_2512
```

已有全量统计文件：

```text
/home/i-xujiahao/arxiv_data/arxiv_fig_extract_stats.json
```

全量粗统计：

```text
paper_count: 1,860,587
papers_with_extracted: 1,860,587
papers_with_merged: 576,424
extracted_image_count: 14,492,184
merged_image_count: 2,160,226
total_image_count: 16,652,410
```

## 2. Figure JSON 可用字段

单图 JSON 典型字段：

```json
{
  "paper_id": "2001.00003",
  "figure_index": 1,
  "figure_env": "figure*",
  "figure_line_start": 604,
  "figure_line_end": 609,
  "figure_tex": "...",
  "labels": ["fig:integrate"],
  "caption_latex": "...",
  "reference_paragraphs_latex": ["..."],
  "image_index_in_figure": 1,
  "status": "success",
  "output_image": ".../fig_0001_img_01_integrate.png",
  "output_json": ".../fig_0001_img_01_integrate.json",
  "source_effective_ext": ".png",
  "was_converted_to_png": false
}
```

merged 图 JSON 典型字段：

```json
{
  "record_type": "merged_figure",
  "is_merged_figure": true,
  "paper_id": "2001.00003",
  "figure_index": 2,
  "caption_latex": "...",
  "reference_paragraphs_latex": ["..."],
  "output_image": ".../fig_0002_merged_3x1_03panels.png",
  "merge_layout": {
    "rows": 3,
    "cols": 1,
    "num_panels": 3
  },
  "merged_panel_count": 3,
  "merged_source_image_count": 3,
  "merged_source_image_indices": [1, 2, 3],
  "merged_from_records": [...]
}
```

这些字段足够构造 manifest、筛选真实 chart、生成 QA、生成 dense-caption、追踪来源和排查错误。

## 3. CharXiv Paper 集合

CharXiv 数据位置：

```text
/mnt/stepeval/datasets/VL_datasets/CharXiv/data/
  image_metadata_val.json
  image_metadata_test.json
  chart_types_val.json
  chart_types_test.json
  descriptive_val.json
  descriptive_test.json
  reasoning_val.json
  reasoning_test.json
```

CharXiv metadata 中可直接取：

```json
{
  "figure_id": 0,
  "paper_id": "2004.10956",
  "category": "cs",
  "year": "20",
  "figure_path": "...",
  "caption": "...",
  "title": "..."
}
```

本轮要先构建：

```text
/home/i-xujiahao/arxiv_data/work/charxiv_paper_ids.json
```

来源：

```text
image_metadata_val.json
image_metadata_test.json
```

字段建议：

```json
{
  "paper_ids": ["2004.10956", "..."],
  "records": [
    {
      "split": "val",
      "figure_id": 0,
      "paper_id": "2004.10956",
      "chart_types": ["Line Chart"],
      "title": "...",
      "caption": "..."
    }
  ]
}
```

注意：

- `charxiv_inclusive` 不是把 CharXiv QA 当训练数据。
- `charxiv_inclusive` 只是允许当前 arXiv 抽图中来自同一批 CharXiv paper 的图进入训练样本，并打标签。
- `charxiv_exclusive` 必须按 `paper_id` 排除这些 paper，后续还应做 image pHash 近邻排重。

## 4. 第一版筛选原则

本轮只做真实 arXiv chart 图。需要排除：

- 模型结构图 / pipeline / framework diagram
- 算法流程图
- 纯公式截图
- 表格截图
- 代码截图 / UI 截图
- 自然图像、医学图像、遥感图像、显微图像等非 chart 图
- 定性可视化图，例如 segmentation examples、generated samples、attention visualization，除非它本身是 heatmap / confusion matrix / chart
- 字体完全不可读或图像分辨率过低的图

保留：

- line chart
- bar chart
- scatter plot
- histogram
- heatmap
- confusion matrix
- box plot / violin plot
- area chart
- calibration / ROC / PR / ablation curve
- table-like but actually plotted matrix / heatmap
- multi-panel chart，只要大部分 panel 是 chart

## 5. Candidate Manifest 构建

先遍历 2020-2025 的 bucket，构建轻量 manifest，不立即调模型。

输入：

```text
/mnt/lvhaoran-jfs/cyy/arxiv_fig_extract_0608_addmerge/arxiv_20*
/mnt/lvhaoran-jfs/cyy/arxiv_fig_extract_0608_addmerge/arxiv_21*
/mnt/lvhaoran-jfs/cyy/arxiv_fig_extract_0608_addmerge/arxiv_22*
/mnt/lvhaoran-jfs/cyy/arxiv_fig_extract_0608_addmerge/arxiv_23*
/mnt/lvhaoran-jfs/cyy/arxiv_fig_extract_0608_addmerge/arxiv_24*
/mnt/lvhaoran-jfs/cyy/arxiv_fig_extract_0608_addmerge/arxiv_25*
```

只接受 `paper_id` 匹配：

```text
^(20|21|22|23|24|25)[0-9]{2}\.[0-9]+
```

候选记录字段：

```json
{
  "candidate_id": "2001.00003_fig0002_merged",
  "paper_id": "2001.00003",
  "year": 2020,
  "month": "2001",
  "source_bucket": "arxiv_2001",
  "figure_index": 2,
  "image_kind": "merged",
  "image_path": "...png",
  "json_path": "...json",
  "caption_latex": "...",
  "reference_paragraphs_latex": ["..."],
  "labels": ["fig:prototypes"],
  "figure_tex": "...",
  "merged_panel_count": 3,
  "merge_layout": {"rows": 3, "cols": 1},
  "is_charxiv_paper": false
}
```

### 5.1 Merged 与 Single 的选择

优先级：

1. 如果同一个 `paper_id + figure_index` 有 merged 图，并且 `merged_panel_count` 在 `[2, 8]`，优先使用 merged 图。
2. 如果没有 merged 图，使用 single 图。
3. 如果 single 图来自同一 figure 的多个 panel，但没有 merged 图：
   - 可以保留 single panel，但标注 `image_kind=single_panel`。
   - 第一版不要对 single panel 生成需要整图上下文的问题。
4. 同一个 `paper_id + figure_index` 不同时进入 merged 和 single，避免重复。

原因：

- arXiv chart 经常是多 panel，原始 caption 解释的是整组 figure。
- 用 merged 图更接近论文真实 figure 语境。
- 同时保留 merged 和 single 会制造重复和答案歧义。

## 6. Chart 筛选方法

筛选分四层，先便宜后昂贵。

### 6.1 Hard Filter

直接丢弃：

```text
status != "success"
image_path 不存在
文件不是可解码图片
caption_latex 为空或长度 < 10
短边 < 256
面积 < 80,000 px
宽高比 < 0.15 或 > 8.0
```

保留长条图的原因：真实论文中有很宽的 multi-panel chart，所以不能一刀切丢掉宽高比大的图。但宽高比大于 5 的图必须通过 VLM chart classifier。

### 6.2 Metadata / Caption Keyword Filter

给每个候选计算 `metadata_chart_score`。

正向关键词：

```text
accuracy, loss, score, performance, comparison, baseline, ablation,
epoch, step, training, validation, test, metric, benchmark,
precision, recall, f1, auc, roc, pr curve, error, rmse, mae,
distribution, histogram, cumulative, density, frequency,
scatter, correlation, heatmap, confusion matrix,
latency, throughput, speed, time, memory, parameter, scaling,
redshift, flux, spectrum, mass, temperature, energy, residual, profile
```

负向关键词：

```text
architecture, framework, pipeline, overview, workflow,
algorithm, pseudocode, screenshot, user interface, ui,
qualitative examples, generated samples, input image, output image,
segmentation result, detection example, reconstruction example,
network structure, computational graph, model diagram
```

规则：

- caption/path/label 命中强正向关键词，加分。
- 命中强负向关键词，降分但不直接丢弃，因为有些 heatmap / confusion matrix 也会出现在 qualitative section。
- `reference_paragraphs_latex` 可辅助判断，但第一版不要把它作为训练输入。

### 6.3 VLM Chart Classifier

对 hard filter 后的候选，用 VLM 做二分类和标签抽取。建议先用较便宜模型或批量策略，输出严格 JSON。

固定模型：

```json
{
  "model": "gemini-3.5-flash",
  "protocol": "gemini_native_generateContent",
  "maxOutputTokens": 4096,
  "temperature": 0,
  "topP": 1,
  "extra_kwargs": {
    "reasoning_effort": "low"
  }
}
```

说明：

- Chart classifier 是低成本、低方差分类任务，不使用 answer 阶段的 `temperature=1` 和 `reasoning_effort=high`。
- Gemini native 省略 `topK`，不传 `-1`。
- 输出必须是严格 JSON；解析失败直接 retry，retry 后仍失败则进入 `classifier_failed` 队列。

输入：

```text
image
caption_latex
```

输出 schema：

```json
{
  "is_real_chart": true,
  "chart_confidence": 0.93,
  "chart_types": ["line_chart", "bar_chart"],
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
suitable_for_dense_caption == true
suitable_for_vqa == true
not is_diagram
not is_table_screenshot
not is_photo_or_qualitative_image
```

弱接受：

```text
0.60 <= chart_confidence < 0.75
```

弱接受样本进入 `review_pool`，不进入第一版 50k。

### 6.4 Dedup

必须做三层去重：

1. `paper_id + figure_index + image_kind` 去重。
2. 图片 pHash 去重。
3. image embedding 近邻去重，优先用于排除 CharXiv near-duplicate。

第一版可以先实现 1 + pHash。embedding near-duplicate 可以作为后续 TODO。

## 7. 50k 采样策略

每组采样单位是 chart image。

### 7.1 全局配额

每组目标：

```text
target_images = 50,000
```

年分布先均衡：

```text
2020: ~8,333
2021: ~8,333
2022: ~8,333
2023: ~8,333
2024: ~8,333
2025: ~8,335
```

如果某年通过筛选的高质量样本不足，则按相邻年份补齐。

### 7.2 Month 与 Paper 限制

限制：

```text
max_images_per_paper = 2
max_images_per_month = ceil(year_quota / 12 * 1.3)
```

原因：

- 防止单篇图很多的论文支配分布。
- 防止某个月份或某个抽图 batch 支配分布。

### 7.3 Chart Type 平衡

VLM classifier 输出 `chart_types` 后，采样时尽量平衡：

```text
line_chart: 25%-35%
bar_chart: 10%-20%
scatter_plot: 10%-20%
heatmap/confusion_matrix: 10%-20%
histogram/density/distribution: 5%-15%
box/violin/area/other: 5%-15%
multi_panel_chart: at least 25%
```

不强求精确，但要避免 50k 几乎全是 line chart 或全是 CS ablation 图。

### 7.4 Inclusive 与 Exclusive

`charxiv_inclusive`：

```text
candidate_pool = all accepted 2020-2025 candidates
sampling:
  1. include all eligible candidates where paper_id in charxiv_paper_ids, subject to max_images_per_paper
  2. fill remaining slots using non-CharXiv candidates with same year/month/type balancing
```

`charxiv_exclusive`：

```text
candidate_pool = accepted 2020-2025 candidates
filter:
  paper_id not in charxiv_paper_ids
  pHash not near CharXiv images
sampling:
  same year/month/type balancing
```

如果用户想要“50k 全部来自 CharXiv 涉及 paper”，需要先统计 CharXiv paper 在当前抽图目录中的可用 chart 数。按目前 CharXiv 规模看，这个要求大概率不现实；本方案默认是“包含 CharXiv paper 的 ablation 版本”。

## 8. QA 构造方式

QA 必须分开构造：

```text
Question generation
  -> Answer generation by Gemini 3.5 Flash, repeated 3 times
  -> answer consistency check
  -> Kimi K2.6 thinking response generation
  -> final validation
```

不要让同一个模型在同一次调用中同时生成 question 和 answer，否则容易形成自洽但不真实的伪 QA。

### 8.0 API 调用配置

API 调用参考：

```text
/home/i-xujiahao/api.py
```

Kimi Base URL：

```text
https://models-proxy.stepfun-inc.com/v1
```

Gemini 原生 endpoint：

```text
https://models-proxy.stepfun-inc.com/gemini/v1beta/models/gemini-3.5-flash:generateContent
```

实现约束：

- Kimi 走 OpenAI-compatible `/v1/chat/completions`。
- Gemini 必须走原生 Gemini `generateContent` 协议，不走 OpenAI-compatible 协议。
- 图片只在请求时临时读取并编码；**JSONL、日志、报告中只保存 image path，不保存 image bytes 或 base64**。

Kimi 固定配置：

```json
{
  "model": "kimi-k2.6-aliyun2kimi",
  "max_tokens": 250000,
  "temperature": 1,
  "top_p": 1,
  "top_k": -1
}
```

Gemini 固定配置：

```json
{
  "model": "gemini-3.5-flash",
  "maxOutputTokens": 64000,
  "temperature": 1,
  "topP": 0.95,
  "extra_kwargs": {
    "reasoning_effort": "high"
  }
}
```

说明：用户口径中的 `Top-k=-1` 在 OpenAI-compatible 调用里可以传 `top_k=-1`，但 Gemini 原生协议要求 `topK` 非负；实测 `topK=-1` 会被上游拒绝。实现时对 Gemini native **省略 `topK`**。

最小调用形态参考：

```python
import requests

url = "https://models-proxy.stepfun-inc.com/gemini/v1beta/models/gemini-3.5-flash:generateContent"
headers = {
    "Content-Type": "application/json",
    "x-google-api-key": OPENAI_API_KEY,
}
payload = {
    "contents": [
        {"role": "user", "parts": [{"text": "..."}]}
    ],
    "generationConfig": {
        "maxOutputTokens": 64000,
        "temperature": 1,
        "topP": 0.95
    },
    "extra_kwargs": {"reasoning_effort": "high"}
}
response = requests.post(url, headers=headers, json=payload, timeout=120)
```

Gemini 原生图片输入格式：

```json
{
  "contents": [
    {
      "role": "user",
      "parts": [
        {
          "inlineData": {
            "mimeType": "image/png",
            "data": "<base64 only in request memory>"
          }
        },
        {
          "text": "Question: ..."
        }
      ]
    }
  ],
  "generationConfig": {
    "maxOutputTokens": 64000,
    "temperature": 1,
    "topP": 0.95
  },
  "extra_kwargs": {
    "reasoning_effort": "high"
  }
}
```

### 8.1 Question Generation

Question generation 固定使用 Gemini native，目标是生成高质量、可由图像回答的问题。这里和 answer generation 一样开启 high reasoning。

固定模型：

```json
{
  "model": "gemini-3.5-flash",
  "protocol": "gemini_native_generateContent",
  "maxOutputTokens": 64000,
  "temperature": 1,
  "topP": 0.95,
  "extra_kwargs": {
    "reasoning_effort": "high"
  }
}
```

输入：

```text
image
caption_latex
chart classifier output
optional OCR / detected visible text
```

输出只包含 question，不包含 answer：

```json
{
  "question": "Which method has the highest final accuracy?",
  "task_type": "visual_comparison",
  "answer_type": "short_text",
  "evidence_source": "image_only",
  "difficulty": "medium",
  "requires_exact_reading": false,
  "requires_caption_context": false,
  "question_risk": "low"
}
```

第一版 QA 只做 `image_only`。如果问题依赖论文 caption 或正文，不进入第一版 QA。

### 8.2 第一版 QA 类型配比

每张图先生成 1 条 QA。配比按全局控制：

```text
descriptive_extraction: 20%
  title / x-axis / y-axis / legend / colorbar / subplot label

text_in_chart: 15%
  answer is a visible label, method name, category, or subplot name

number_in_chart: 15%
  answer is a visible or approximate number

visual_comparison: 20%
  highest / lowest / larger / smaller / better / worse

trend_reasoning: 15%
  increasing / decreasing / stable / crossing / convergence

counting: 10%
  number of bars / curves / panels / clusters / points when visually reliable

unanswerable_or_uncertain: 0%
  first version does not generate unanswerable/uncertain QA
```

第一版先不生成 `unanswerable_or_uncertain`。拒答/不确定性样本放到第二版专项构造。

### 8.3 Gemini Answer 三次一致

Answer 生成模型：

```text
gemini-3.5-flash
native generateContent
maxOutputTokens=64000
temperature=1
topP=0.95
extra_kwargs={"reasoning_effort": "high"}
```

每条 question 连续生成三次：

```text
answer_run_1
answer_run_2
answer_run_3
```

Prompt 要求：

```text
Look only at the chart image.
Answer the question with the shortest correct answer.
Do not explain.
If the exact value is unreadable, answer with an approximate value only when the question allows approximation.
If the question cannot be answered from the image alone, answer "Not answerable from the image alone".
```

一致性判定：

```text
short_text:
  lower-case, strip punctuation/articles/spaces, normalize unicode, exact match

numeric_exact:
  parse number, exact integer or exact decimal-place match

numeric_approx:
  parse number, allow small tolerance:
    absolute tolerance <= max(0.02 * scale, task-specific tolerance)
    or relative tolerance <= 5%

yes_no:
  normalize yes/no

list:
  normalize items, compare unordered unless question asks for order
```

接受条件：

```text
all three normalized answers equal
```

可选弱接受：

```text
2 of 3 equal and third is compatible by numeric tolerance
```

第一版建议只接受 3/3 一致，保证质量。

失败处理：

```text
no consensus -> drop QA candidate
drop_count high for a task_type -> reduce that task_type or improve prompt
```

### 8.4 Kimi K2.6 Thinking Response

Thinking response 模型：

```text
kimi-k2.6-aliyun2kimi
max_tokens=250000
temperature=1
top_p=1
top_k=-1
```

Kimi 不负责确定 GT。GT 只来自 Gemini 三次一致后的 `consensus_answer`。

Kimi 输入：

```text
image
question
consensus_answer
task_type
instruction: reason from visible chart evidence, then end with Final answer: <consensus_answer>
```

输出格式：

```text
<think>
短推理，必须引用可见证据，不要引入 caption 或论文背景。
</think>
Final answer: ...
```

验证：

```text
extract final answer from Kimi response
normalize(final answer) == normalize(consensus_answer)
```

不一致时：

```text
retry Kimi once
still mismatch -> keep QA with direct answer only, mark thinking_response_failed=true
```

训练时可以产出两种版本：

```text
qa_direct.jsonl
  assistant only outputs consensus_answer

qa_thinking.jsonl
  assistant outputs Kimi thinking response + Final answer
```

## 9. Dense Caption 构造方式

除了 VQA，每张图生成 1 条 image dense-caption pair。

第一版 dense-caption **必须使用 verifier**。最终导出的 dense-caption record 必须满足：

```text
generated_by_kimi == true
verified_by_gemini == true
verifier_pass == true
```

不通过 verifier 的 dense-caption 不进入最终 JSONL。

### 9.1 Dense Caption Generator

Generator 固定模型：

```json
{
  "model": "kimi-k2.6-aliyun2kimi",
  "protocol": "openai_compatible_chat_completions",
  "max_tokens": 250000,
  "temperature": 1,
  "top_p": 1,
  "top_k": -1
}
```

Generator 输入：

```text
image
caption_latex
chart classifier output
```

`reference_paragraphs_latex` 第一版不喂给 caption generator。

Generator 输出必须是严格 JSON：

```json
{
  "dense_caption": "...",
  "visible_elements": {
    "chart_types": [],
    "axes": [],
    "legend_or_series": [],
    "panels": [],
    "main_trends": [],
    "notable_comparisons": []
  },
  "uncertainty": []
}
```

最终 caption 必须以图像可见信息为主，可以参考 `caption_latex` 理解术语，但不能把 caption 中不可见的论文背景结论写进 image-only target。

Dense caption 目标：

```text
用自然语言详细描述图中可见 chart 内容。
必须包括 chart type、axes、legend/series、visual encodings、主要趋势/比较、multi-panel layout。
不能编造论文方法、数据集、结论，除非这些文字在图中可见。
```

推荐输出结构：

```json
{
  "dense_caption": "The figure contains three vertically stacked histograms. The x-axis shows numeral quantity on a logarithmic scale, and the y-axis shows counts. The panels compare numerals in Wikipedia 1B with two prototype distributions. The distributions are concentrated at smaller quantities and become sparse toward larger values.",
  "visible_elements": {
    "chart_types": ["histogram"],
    "axes": ["numeral quantity", "number of occurrences"],
    "series_or_panels": ["Numerals in Wikipedia 1B", "Prototypes of SOM-500", "Prototypes of GMM-500-soft"],
    "main_trends": ["counts are higher at smaller quantities", "all panels show long-tailed distributions"]
  },
  "uncertainty": []
}
```

Dense-caption 质量要求：

```text
不少于 2 句
不超过 180 words
不得包含 "the paper proposes" 这类非图像可见结论
必须提到图表类型或主要 visual encoding
如果图中文字不可读，明确说明 "some labels are not legible"
```

### 9.2 Dense Caption Verifier

Verifier 固定模型：

```json
{
  "model": "gemini-3.5-flash",
  "protocol": "gemini_native_generateContent",
  "maxOutputTokens": 4096,
  "temperature": 0,
  "topP": 1,
  "extra_kwargs": {
    "reasoning_effort": "high"
  }
}
```

Verifier 输入：

```text
image
caption_latex
chart classifier output
generated dense_caption JSON
```

Verifier 输出必须是严格 JSON：

```json
{
  "pass": true,
  "grounded_score": 0.93,
  "chart_type_correct": true,
  "visible_elements_supported": true,
  "no_paper_context_hallucination": true,
  "no_unsupported_numeric_claim": true,
  "major_issues": [],
  "minor_issues": []
}
```

Verifier pass 条件：

```text
pass == true
grounded_score >= 0.85
chart_type_correct == true
visible_elements_supported == true
no_paper_context_hallucination == true
no_unsupported_numeric_claim == true
```

失败处理：

```text
1. 第一次 verifier fail：把 verifier major_issues 反馈给 Kimi，retry dense-caption 一次。
2. 第二次 verifier pass：导出该 record，标记 dense_caption_retry_count=1。
3. 第二次仍 fail：丢弃该 dense-caption record，写入 logs/dense_caption_verifier_failures.jsonl。
```

第一版不做三次 caption 一致，因为 dense-caption 是开放文本，三次一致不合适。第一版质量门槛由 Gemini verifier + 人工抽检保证。

## 10. 输出格式

工作目录：

```text
/home/i-xujiahao/arxiv_data/work/
```

当前实际存储（2026-06-11 更新）：

```text
/home/i-xujiahao/arxiv_data/work -> /mnt/xjh/data/arxiv_chart/work
```

说明：`work` 已迁移到 `/mnt/xjh/data/arxiv_chart/work` 并在原路径保留软链接，避免根分区被大规模 manifest / filtered JSONL 填满。

建议输出：

```text
work/
  charxiv_paper_ids.json
  candidates_2020_2025.jsonl
  candidates_2020_2025.filtered.jsonl
  candidates_2020_2025.chart_classified.jsonl
  sample_charxiv_inclusive_50k.jsonl
  sample_charxiv_exclusive_50k.jsonl
  qa/
    charxiv_inclusive_50k.qa_direct.jsonl
    charxiv_inclusive_50k.qa_thinking.jsonl
    charxiv_exclusive_50k.qa_direct.jsonl
    charxiv_exclusive_50k.qa_thinking.jsonl
  dense_caption/
    charxiv_inclusive_50k.dense_caption.jsonl
    charxiv_exclusive_50k.dense_caption.jsonl
  logs/
    answer_consensus_failures.jsonl
    kimi_thinking_failures.jsonl
    dense_caption_verifier_failures.jsonl
  reports/
    sampling_report.md
    quality_stats.json
```

### 10.1 QA Record

```json
{
  "id": "arxiv_chart_qa_2001.00003_fig0002_000001",
  "group": "charxiv_exclusive_50k",
  "image": "/mnt/lvhaoran-jfs/.../fig_0002_merged_3x1_03panels.png",
  "source": {
    "paper_id": "2001.00003",
    "year": 2020,
    "month": "2001",
    "figure_index": 2,
    "image_kind": "merged",
    "is_charxiv_paper": false,
    "json_path": "...json"
  },
  "task_type": "visual_comparison",
  "answer_type": "short_text",
  "evidence_source": "image_only",
  "question": "Which panel has the most concentrated distribution near small values?",
  "answer": "Numerals in Wikipedia 1B",
  "answer_generation": {
    "model": "gemini-3.5-flash",
    "protocol": "gemini_native_generateContent",
    "model_config": {
      "maxOutputTokens": 64000,
      "temperature": 1,
      "topP": 0.95,
      "extra_kwargs": {"reasoning_effort": "high"}
    },
    "runs": ["Numerals in Wikipedia 1B", "Numerals in Wikipedia 1B", "Numerals in Wikipedia 1B"],
    "normalized_runs": ["numerals in wikipedia 1b", "numerals in wikipedia 1b", "numerals in wikipedia 1b"],
    "consensus": true
  },
  "thinking_response": {
    "model": "kimi-k2.6-aliyun2kimi",
    "model_config": {
      "max_tokens": 250000,
      "temperature": 1,
      "top_p": 1,
      "top_k": -1
    },
    "response": "<think>...</think>\nFinal answer: Numerals in Wikipedia 1B",
    "final_answer_matches_consensus": true
  }
}
```

### 10.2 Dense Caption Record

```json
{
  "id": "arxiv_chart_caption_2001.00003_fig0002",
  "group": "charxiv_exclusive_50k",
  "image": "/mnt/lvhaoran-jfs/.../fig_0002_merged_3x1_03panels.png",
  "source": {
    "paper_id": "2001.00003",
    "year": 2020,
    "figure_index": 2,
    "image_kind": "merged",
    "is_charxiv_paper": false,
    "caption_latex": "..."
  },
  "task_type": "dense_caption",
  "evidence_source": "image_only",
  "messages": [
    {
      "role": "user",
      "content": "<image>\nDescribe this chart in detail using only visible information."
    },
    {
      "role": "assistant",
      "content": "..."
    }
  ],
  "quality": {
    "caption_model": "kimi-k2.6-aliyun2kimi",
    "caption_model_config": {
      "max_tokens": 250000,
      "temperature": 1,
      "top_p": 1,
      "top_k": -1
    },
    "verified": true,
    "verifier_model": "gemini-3.5-flash",
    "verifier_protocol": "gemini_native_generateContent",
    "verifier_model_config": {
      "maxOutputTokens": 4096,
      "temperature": 0,
      "topP": 1,
      "extra_kwargs": {"reasoning_effort": "high"}
    },
    "verifier_pass": true,
    "dense_caption_retry_count": 0
  }
}
```

## 11. 质量控制

必须记录：

```text
chart classifier pass rate
sampling distribution by year/month/chart_type
CharXiv paper coverage in inclusive set
CharXiv paper exclusion count in exclusive set
QA question generation count
Gemini answer 3/3 consensus rate
Kimi final answer mismatch rate
dense-caption verifier pass rate
manual spot check pass rate
```

人工抽检：

```text
每组先抽 200 条 QA
每组先抽 100 条 dense-caption
重点检查：
  是否真的是 chart
  是否答案可由图像单独回答
  Gemini 三次一致是否仍然可能错
  Kimi thinking 是否引入不可见信息
  dense-caption 是否把 caption_latex 的论文结论混入 image-only target
```

第一版通过标准：

```text
chart precision >= 90%
QA answer correctness spot-check >= 85%
dense-caption groundedness >= 85%
Kimi final answer mismatch <= 3%
```

如果达不到，先不要扩到 50k，回到筛选 prompt 和 question prompt。

## 12. TODO Tasks

### A. 数据访问与目录扫描

- [x] 进入挂载环境：运行 `/home/i-xujiahao/test4.sh`，确认 `/mnt/lvhaoran-jfs/cyy/arxiv_fig_extract_0608_addmerge` 可读。
- [x] 写 `build_charxiv_paper_ids.py`，从 CharXiv `image_metadata_val.json` 和 `image_metadata_test.json` 抽取 unique `paper_id`。
- [x] 输出 `/home/i-xujiahao/arxiv_data/work/charxiv_paper_ids.json`。
- [x] 写 `scan_arxiv_2020_2025_candidates.py`，遍历 2020-2025 bucket，解析 single / merged JSON。
- [ ] 输出 `/home/i-xujiahao/arxiv_data/work/candidates_2020_2025.jsonl`。（曾生成 6,168,729 条；后因根分区满被删除释放空间，当前文件不存在）
- [x] 统计候选数量：按 year、month、paper、single/merged、caption length、image suffix。

### B. 本地硬筛

- [x] 写 `filter_candidates_local.py`。
- [x] 实现 hard filter：`status`、图片存在、可解码、尺寸、宽高比、caption 长度。
- [x] 实现 merged 优先：同一 `paper_id + figure_index` 优先使用 merged，避免 single/merged 重复。
- [x] 实现 caption/path keyword score，输出 `metadata_chart_score`。
- [x] 输出 `/home/i-xujiahao/arxiv_data/work/candidates_2020_2025.filtered.jsonl`。
- [x] 生成 local filter 报告：保留率、丢弃原因 top-k、每年剩余样本。

### C. VLM Chart Classifier

- [x] 设计 chart classifier prompt，要求输出严格 JSON。
- [x] 写 `classify_chart_candidates.py`，支持断点续跑和并发限速。
- [ ] 对 `filtered.jsonl` 调用 Gemini native `gemini-3.5-flash`，固定参数：`maxOutputTokens=4096, temperature=0, topP=1, topK 省略, extra_kwargs={"reasoning_effort": "low"}`。（当前只跑了 classifier pilot）
- [x] 输出 `is_real_chart/chart_confidence/chart_types/text_readability/suitable_for_vqa`。
- [ ] 输出 `/home/i-xujiahao/arxiv_data/work/candidates_2020_2025.chart_classified.jsonl`。（当前为 pilot 输出：817 条 classified，不是 full filtered 输出）
- [ ] 抽检 200 条 classifier pass 样本，估计 chart precision。
- [ ] 抽检 100 条 classifier reject 样本，确认没有大量误杀 chart。

### D. 去重与两组 50k 采样

- [x] 写 `dedup_chart_candidates.py`，实现 `paper_id + figure_index + image_kind` 去重。
- [x] 加 pHash 去重，记录 duplicate group。
- [x] 写 `sample_50k_groups.py`。
- [ ] 生成 `sample_charxiv_inclusive_50k.jsonl`。（当前是 500 条 pilot，不是 50k）
- [ ] 生成 `sample_charxiv_exclusive_50k.jsonl`。（当前是 500 条 pilot，不是 50k）
- [ ] 输出 sampling report：year/month/chart_type/multi_panel/is_charxiv_paper 分布。（当前只有 pilot report）
- [ ] 检查 exclusive set 中 `paper_id in charxiv_paper_ids` 数量必须为 0。（pilot 已检查为 0；full 50k 未做）
- [ ] 检查 inclusive set 中 CharXiv paper 覆盖数量，并报告可匹配的 CharXiv paper / figure 数。（pilot 已报告；full 50k 未做）

### E. QA Question Generation

- [x] 设计 image-only question generation prompt。
- [x] 写 `generate_questions.py`，输入 50k sample manifest，输出只含 question 的 records。
- [x] question generation 调用 Gemini native `gemini-3.5-flash`，固定参数：`maxOutputTokens=64000, temperature=1, topP=0.95, topK 省略, extra_kwargs={"reasoning_effort": "high"}`。
- [x] 控制任务配比：descriptive、text_in_chart、number_in_chart、visual_comparison、trend、counting。
- [x] question record 必须包含 `task_type/answer_type/evidence_source/difficulty/requires_caption_context=false`。
- [ ] 对每组先生成 500 条 pilot question。（exclusive 500/500；inclusive 497/500，另 3 条失败）
- [ ] 抽检 100 条 question，确认问题能由图像单独回答。

### F. Gemini Answer 三次一致

- [x] 写 `answer_questions_gemini.py`。
- [x] 每条 QA 通过 Gemini 原生 `generateContent` 调用 `gemini-3.5-flash` 三次，固定参数：`maxOutputTokens=64000, temperature=1, topP=0.95, topK 省略, extra_kwargs={"reasoning_effort": "high"}`。
- [x] 实现 answer normalization：short_text、numeric_exact、numeric_approx、yes_no、list。
- [x] 只接受 3/3 一致的答案。
- [ ] 输出 consensus QA。（当前为 pilot consensus：inclusive 348、exclusive 326；full 50k 未完成）
- [x] 记录失败样本到 `logs/answer_consensus_failures.jsonl`。
- [x] pilot 阶段统计 consensus rate，若低于 60%，回调 question prompt。

### G. Kimi Thinking Response

- [x] 写 `generate_kimi_thinking_response.py`。
- [x] 输入 image、question、Gemini consensus answer。
- [x] 调用 `kimi-k2.6-aliyun2kimi` 生成 `<think>...</think>\nFinal answer: ...`，固定参数：`max_tokens=250000, temperature=1, top_p=1, top_k=-1`。
- [x] 抽取 Kimi final answer，必须匹配 consensus answer。
- [x] 不匹配则 retry 一次。
- [x] 仍不匹配则保留 direct QA，标记 `thinking_response_failed=true`。
- [ ] 输出 `qa_direct.jsonl` 和 `qa_thinking.jsonl` 两个版本。（当前为 pilot QA 导出：inclusive 348、exclusive 326；full 50k 未完成）

### H. Dense Caption 生成

- [x] 设计 dense-caption prompt，明确只描述图像可见信息。
- [x] 写 `generate_dense_caption.py`。
- [ ] 每张图调用 `kimi-k2.6-aliyun2kimi` 生成 1 条 dense-caption，固定参数：`max_tokens=250000, temperature=1, top_p=1, top_k=-1`。（pilot 已导出两组各 500；其中 inclusive 357/500、exclusive 438/500 为 fallback 标记，full 50k 未完成）
- [ ] 输出结构化字段：`dense_caption/visible_elements/uncertainty`。（pilot 已完整；full 50k 未完成）
- [ ] 可选调用 Gemini verifier 检查 groundedness。
- [ ] 记录 verifier 失败样本。
- [ ] 抽检每组 100 条 dense-caption。

### I. 数据导出与报告

- [ ] 导出每组 QA direct / QA thinking / dense-caption JSONL。（pilot 已导出；full 50k 未完成）
- [x] 导出检查：JSONL 只保存 image path，不保存 image base64、image bytes 或图片副本。
- [x] 生成 `quality_stats.json`。
- [ ] 生成 `sampling_report.md`，包含样本分布、模型调用成功率、一致率和抽检结论。（当前只有 pilot sampling report，缺完整人工抽检结论）
- [x] 准备 200 条可视化 HTML review 页面，展示 image、caption_latex、question、Gemini 三次 answer、Kimi thinking、dense caption。
- [ ] 用户检查 pilot 后，再扩大到完整 50k。

## 13. 第一版最小闭环

建议先跑最小闭环：

```text
每组 500 images
每张图 1 QA + 1 dense-caption
总计：
  inclusive: 500 QA + 500 caption
  exclusive: 500 QA + 500 caption
```

通过检查后再跑：

```text
每组 5k images
```

最后再跑：

```text
每组 50k images
```

不要直接从 0 跑到 50k。当前最大风险不是样本数，而是筛选精度和 QA 伪一致。

## 14. 当前决策摘要

```text
数据源：真实 arXiv figure extraction，2020-2025
图像类型：真实 chart only
两组数据：CharXiv paper inclusive / exclusive
每组规模：50k chart images
每图输出：1 QA + 1 dense-caption
QA 构造：question 与 answer 分开
Answer：gemini-3.5-flash 走 Gemini 原生 generateContent，连续生成 3 次，3/3 一致才接受
Thinking response：kimi-k2.6-aliyun2kimi 生成，final answer 必须匹配 Gemini consensus
Caption：dense image-only caption，避免混入论文不可见结论
存储：JSONL 只存 image path，不存 image base64 或图片副本
核心质量控制：chart classifier、dedup、paper-level CharXiv 排除、人工抽检
```
