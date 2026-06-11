# Pilot QA 问题记录 edit1

生成文件：

- `work/qa/charxiv_inclusive_50k.questions.jsonl`: 497 条 question
- `work/qa/charxiv_exclusive_50k.questions.jsonl`: 500 条 question
- `work/qa/charxiv_inclusive_50k.consensus.jsonl`: 348 条 consensus QA
- `work/qa/charxiv_exclusive_50k.consensus.jsonl`: 326 条 consensus QA

## 1. Question 明显偏简单

997 条 pilot question 的难度分布：

```text
medium 698
easy   229
low     70
hard     0
```

问题：

- 没有任何 `hard` question。
- 出现了不规范 difficulty 值 `low`，应该统一为 `easy / medium / hard` 或明确新增枚举。
- 大量问题只是读图中文字、坐标轴标签、标题、图例，不足以构成高质量 chart reasoning benchmark。

典型简单问题：

```text
What is the title of the chart?
What is the label for the vertical axis?
What is the label on the y-axis?
What is the label for the horizontal axis?
What is the label for the colorbar on the right side of the chart?
How many distinct curves are plotted in the chart?
```

## 2. Answer type 基本全是 short_text

997 条 question 的 `answer_type` 分布：

```text
short_text 996
number       1
```

问题：

- `number_in_chart` 实际几乎没有生成数值答案。
- `visual_comparison`、`trend_reasoning`、`counting` 也大多被压成短文本答案。
- 当前 prompt 的 JSON 模板固定写了 `"answer_type": "short_text"`，导致模型不会认真区分数值、枚举、布尔、比较结论等答案类型。

需要调整：

- `number_in_chart` 应强制 `answer_type=number` 或 `numeric_range`。
- `counting` 应强制 `answer_type=integer`。
- `visual_comparison` 可用 `answer_type=choice` 或 `short_text`，但问题必须包含明确比较对象。
- `trend_reasoning` 可用 `answer_type=trend_label` 或 `short_text`，但需要避免泛泛描述。

## 3. 低层读取题占比过高

原始 question 中，关键词统计如下：

```text
inclusive:
axis/label       148 / 497 = 29.8%
legend/color     124 / 497 = 24.9%
title             18 / 497 =  3.6%
panel/top/bottom 224 / 497 = 45.1%

exclusive:
axis/label       168 / 500 = 33.6%
legend/color     131 / 500 = 26.2%
title             14 / 500 =  2.8%
panel/top/bottom 230 / 500 = 46.0%
```

启发式判定为“纯读标签/标题/图例、且没有明显推理或计算”的问题：

```text
inclusive 207 / 497 = 41.6%
exclusive 207 / 500 = 41.4%
```

consensus 后最终 QA 中，这个比例更高：

```text
inclusive 168 / 348 = 48.3%
exclusive 158 / 326 = 48.5%
```

问题：

- 最终留下来的 QA 接近一半是简单读取题。
- 这会让 benchmark 更像 chart OCR / metadata extraction，而不是 chart understanding 或 chart reasoning。

## 4. Consensus 过滤进一步偏向简单题

Gemini 三次一致通过率：

```text
inclusive: 348 / 497 = 70.0%
exclusive: 326 / 500 = 65.2%
```

按 task type 看，通过率差异明显：

```text
inclusive:
descriptive_extraction  97 / 119 = 81.5%
text_in_chart           65 /  80 = 81.2%
counting                45 /  52 = 86.5%
number_in_chart         43 /  72 = 59.7%
visual_comparison       70 / 100 = 70.0%
trend_reasoning         28 /  74 = 37.8%

exclusive:
descriptive_extraction  89 / 121 = 73.6%
text_in_chart           66 /  79 = 83.5%
counting                34 /  50 = 68.0%
number_in_chart         38 /  75 = 50.7%
visual_comparison       76 / 100 = 76.0%
trend_reasoning         23 /  75 = 30.7%
```

问题：

- `trend_reasoning` 通过率最低，只有约 30%-38%。
- 三次完全一致会天然保留简单、短答案、标签读取题。
- 难题、数值估计题、趋势题更容易被过滤掉。

需要调整：

- 数值题不要要求字符串完全一致，应使用数值归一化和容差匹配。
- 趋势题不要只做 exact match，应做语义标签归一化，例如 `increasing / decreasing / stable / non-monotonic / crosses`。
- 比较题应归一化选项名称，允许同义表达。

## 5. Prompt 本身鼓励低风险简单题

当前 `scripts/generate_questions.py` 的 prompt 问题：

```text
Allowed task types:
- descriptive_extraction: title, axis label, legend text, colorbar label, subplot label.
...
question_risk should be "low" only if the answer is visually reliable.
Do not generate unanswerable or uncertain questions.
```

问题：

- `descriptive_extraction` 明确允许 title / axis label / legend / colorbar / subplot label。
- prompt 强调低风险和视觉可靠，模型会优先生成最安全的标签读取题。
- `difficulty` 模板固定为 `medium`，但模型仍生成 `easy/low`，说明难度控制不强。
- `answer_type` 模板固定为 `short_text`，直接压制了数值题和计数题。

## 6. 当前 task 配比不适合扩到 50k

当前 `TASK_SEQUENCE`：

```text
descriptive_extraction 20
text_in_chart          15
number_in_chart        15
visual_comparison      20
trend_reasoning        15
counting               10
```

问题：

- `descriptive_extraction` 占比太高，且定义过浅。
- `text_in_chart` 与 `descriptive_extraction` 在实际生成中高度重叠。
- `number_in_chart` 没有真正产出数值型答案。
- `trend_reasoning` 产出后又大量被 consensus 过滤。

建议新配比：

```text
cross_panel_comparison     25%
numeric_estimation         20%
trend_reasoning            20%
multi_step_visual_reasoning 20%
counting_or_structure      10%
text_extraction_max         5%
```

第一版可以先去掉 `descriptive_extraction`，或者仅保留很少比例用于 sanity check。

## 7. 需要禁止的低价值题型

下一版 prompt 应明确禁止：

```text
Do not ask for chart title only.
Do not ask for x-axis/y-axis label only.
Do not ask for legend text only.
Do not ask what a single color represents unless the question also requires comparison or reasoning.
Do not ask for subplot labels such as (a), (b), top, bottom only.
Do not ask generic "what is shown/plotted" questions.
```

允许例外：

- 只有当标签读取是多步问题的一部分时才允许。
- 例如先定位某个 panel / curve，再比较数值或趋势。

## 8. 下一版 question 应加强的类型

建议新增或重定义 task type：

```text
numeric_estimation:
  Read or estimate an approximate numeric value from axes/colorbar/curve/bar.

cross_panel_comparison:
  Compare values/trends/patterns across two or more panels.

trend_reasoning:
  Determine monotonicity, crossing, convergence, divergence, peak, saturation, or phase change.

multi_step_visual_reasoning:
  Need at least two visual operations, e.g. identify series by legend, locate x-range, compare y-values.

structure_counting:
  Count panels/curves/bars/clusters only when not trivial.
```

## 9. Kimi thinking 不是 API 不通，而是批量限流后被 fallback 掩盖

现象：

- review 页面里有不少样本看起来“没有 Kimi thinking”。
- 但 `qa_thinking.jsonl` 并不是完全没有 Kimi response，而是失败样本被脚本 fallback 成了只有 consensus answer 的 assistant message。

实际统计：

```text
work/qa/charxiv_inclusive_50k.qa_thinking.jsonl
total                         348
real_kimi_ok                  310
failed_total                   38
failed_exception               38
failed_final_answer_mismatch    0
assistant_answer_only_messages  38

work/qa/charxiv_exclusive_50k.qa_thinking.jsonl
total                         326
real_kimi_ok                  139
failed_total                  187
failed_exception              187
failed_final_answer_mismatch    0
assistant_answer_only_messages 187
```

失败类型：

```text
inclusive:
429/rate_limit 35
HTTP 400        2
timeout         1

exclusive:
429/rate_limit 183
HTTP 400         3
timeout          1
```

代表性错误：

```text
RuntimeError('RETRYABLE HTTP 424: {"code":424,"msg":"http 429: error, status code: 429, message: Organization Rate limit exceeded, please try again after 1 seconds[...]","data":null}')
```

判断：

- Stepeval 单条 API 测试没问题，不代表批量生成没问题。
- 当前失败主要是批量并发触发 Kimi 组织级限流，网关把上游 `429` 包装成了 `HTTP 424`。
- 不是 final answer 不匹配；统计里 `failed_final_answer_mismatch=0`。
- 当前脚本 `generate_kimi_thinking_response.py` 在异常时会写 fallback：
  - `thinking_response.response = record["answer"]`
  - `thinking_response_failed = true`
  - `messages[-1].content = record["answer"]`
- 所以 review 或下游如果只看 `messages`，就会误以为 Kimi 没有生成 thinking，或者误把 answer-only 当成合法 assistant response。

需要调整：

1. 批量 Kimi generation 降并发：
   - `--workers 1`
   - `--batch-size 1` 或很小
   - 429 后 sleep/backoff，不要立刻重试打满限流。
2. 增加指数退避：
   - 429/424 包含 rate limit 时，等待 2s/5s/10s/30s 后重试。
   - retries 从当前 2 次提高，但必须带 sleep。
3. 不要把异常 fallback 写成正常 answer-only response：
   - 失败样本应单独进入 `logs/kimi_thinking_failures.jsonl`。
   - 或保留在 `qa_thinking.jsonl`，但 `messages` 不应伪装成合法 thinking。
4. Review 页面应显式展示：
   - `thinking_response_failed`
   - `thinking_response.error`
   - fallback/real Kimi response 的区别。
5. 重新补跑失败样本：
   - 输入应只取 `thinking_response_failed=true` 的样本。
   - 成功后替换旧记录或生成新的 `qa_thinking_retry.jsonl`。

## 10. 题目简单但 Kimi 生成仍然慢的原因

直觉上，很多题只是：

```text
What is the label of the y-axis?
What is the title of the chart?
What is the label for the vertical axis?
```

这种题单条回答应该很快。但当前批量 Kimi thinking 的瓶颈不在题目推理难度，而在请求配置和输入体积。

### 10.1 max_tokens 设得过大

当前 `generate_kimi_thinking_response.py` 调用 Kimi 时使用：

```text
max_tokens=250000
temperature=1
top_p=1
top_k=-1
timeout=300
```

问题：

- 对一个只需要一句解释和 final answer 的任务，`max_tokens=250000` 明显过大。
- 模型服务/网关通常会按请求的最大输出预算参与调度和限流，不一定只看最终实际输出长度。
- 674 条 QA 如果每条都声明 250k 输出预算，相当于声明了非常大的批量资源需求，即使实际答案很短也容易触发限流或排队。

建议：

```text
Kimi thinking:
max_tokens 1024 或 2048 即可

dense caption:
max_tokens 4096 或 8192，除非明确需要更长
```

### 10.2 每次请求都上传整张原图 base64

当前 `content_parts()` 会调用 `image_part()`，把原图完整读入并 base64 写进请求：

```python
"url": "data:" + mime + ";base64," + image_base64
```

统计 consensus QA 里的图片体积：

```text
inclusive consensus images:
median image bytes      203,864
p90 image bytes         962,662
max image bytes      16,479,683
max base64 chars     21,972,910
max image area      140,295,000 pixels
largest dims        15000 x 9353

exclusive consensus images:
median image bytes      186,202
p90 image bytes         798,338
max image bytes      17,263,784
max base64 chars     23,018,378
max image area      140,295,000 pixels
largest dims        15000 x 9353
```

问题：

- 有些图片达到 16-17MB，base64 后超过 2200 万字符。
- 最大分辨率达到 15000x9353，PIL 已经出现 `DecompressionBombWarning`。
- 这种请求上传、网关解析、视觉编码都会慢；题目简单也不能绕过图像预处理成本。

建议：

- Kimi thinking 阶段使用压缩后的 review image 或临时缩略图。
- `max_side=1600` 或 `max_side=2048` 足够回答大多数 chart QA。
- 对需要读小字的题可以保留高分辨率，但不要默认全量原图。

### 10.3 批量并发触发组织级 rate limit

当前默认：

```text
--workers 2
--batch-size 40
```

每条失败样本内部又会 retry。遇到 429 时，多条请求会同时重试，容易继续撞限流。

现有失败主要是：

```text
Organization Rate limit exceeded
```

所以问题不是单条 API 不可用，而是批量调度方式不适合 Kimi 这个模型/额度。

建议：

```text
--workers 1
--batch-size 1
```

并在 429/424 rate limit 后做更长 backoff：

```text
2s -> 5s -> 10s -> 30s -> 60s
```

### 10.4 fallback 让问题看起来像“没有 thinking”

当前异常时脚本会写：

```text
thinking_response.response = record["answer"]
messages[-1].content = record["answer"]
thinking_response_failed = true
```

这会让 review 页面看到 answer-only assistant response，误以为 Kimi 没有输出 thinking。

建议：

- 失败样本不要伪装成合法 assistant message。
- review 页面显式显示 `thinking_response_failed` 和 `thinking_response.error`。
- 补跑失败样本时单独输出 `qa_thinking_retry.jsonl`，不要混在旧文件里难以区分。

### 10.5 推荐的 Kimi thinking edit2 配置

```text
max_tokens=2048
temperature=0.2 或 0.7
top_p=0.9
workers=1
batch_size=1
image_max_side=1600
retry_backoff=2/5/10/30/60 seconds
```

如果只是为了构造训练/评测消息，简单题可以不强制 Kimi 生成长 thinking。对于 title/axis/legend-only 题，直接使用 `qa_direct` 更合理；Kimi thinking 应优先用于确实需要比较、趋势、数值估计、多步定位的题。

### 10.6 用户单条 curl 很快，但它和当前批量任务不是同一个请求形态

用户验证过类似下面的单条请求很快：

```text
POST https://models-proxy.stepfun-inc.com/v1/messages
model=kimi-k2.6-aliyun
max_tokens=4096
messages=[{"role":"user","content":"hello"}]
```

这个结果是合理的，但它不能直接说明当前 `qa_thinking` 批量生成也会快，因为当前脚本使用的是另一种请求形态：

```text
endpoint:    /v1/chat/completions
model:       kimi-k2.6-aliyun2kimi
max_tokens: 250000
input:       image as base64 data URL + prompt
batch:       workers=2, batch_size=40
```

关键差异：

1. 文本 hello 没有图片；当前任务每条都上传 chart image。
2. hello 请求 `max_tokens=4096`；当前脚本是 `max_tokens=250000`。
3. hello 是单条；当前是批量并发，多条请求同时撞组织级限流。
4. hello 走 `/v1/messages` + `kimi-k2.6-aliyun`；当前脚本走 `/v1/chat/completions` + `kimi-k2.6-aliyun2kimi`，可能是不同模型别名/网关适配路径。
5. 当前图片最大 16-17MB，base64 后超过 2200 万字符；文本 hello 几乎没有输入负载。

因此更合理的对照测试应该是：

```text
同一个模型/endpoint
同一个 chart image 输入
max_tokens 分别测试 2048 / 4096 / 250000
单条测试后再测 workers=1, batch_size=1
最后才测小并发
```

如果 `/v1/messages` + `kimi-k2.6-aliyun` 对图像输入明显更稳定，可以考虑把脚本从 `/v1/chat/completions` + `kimi-k2.6-aliyun2kimi` 切到用户验证过的 messages endpoint，但需要先确认该 endpoint 的多模态 content 格式和 final answer 抽取格式。

### 10.7 CharXiv 图片明显比当前 arxiv chart 图小

统计路径：

```text
/mnt/stepeval/datasets/VL_datasets/CharXiv/images
```

共 2323 张 `.jpg` 图片。文件大小：

```text
mean    74,398 bytes = 72.7 KB
median  68,944 bytes = 67.3 KB
p75     90,356 bytes = 88.2 KB
p90    115,708 bytes = 113.0 KB
p95    133,084 bytes = 130.0 KB
p99    166,384 bytes = 162.5 KB
max    257,964 bytes = 251.9 KB
```

分辨率：

```text
width mean/median/max   996 / 1024 / 1024
height mean/median/max  701 / 708  / 1024
area mean/median/max    690k / 707k / 1,048,576 pixels
```

对比当前 arxiv chart consensus 图片：

```text
inclusive consensus:
mean image bytes      543,984 = 531.2 KB
median image bytes    203,864 = 199.1 KB
p90 image bytes       962,662 = 940.1 KB
max image bytes    16,479,683 = 15.7 MB
max base64 chars   21,972,910
max area          140,295,000 pixels

exclusive consensus:
mean image bytes      512,144 = 500.1 KB
median image bytes    186,202 = 181.8 KB
p90 image bytes       798,338 = 779.6 KB
max image bytes    17,263,784 = 16.5 MB
max base64 chars   23,018,378
max area          140,295,000 pixels
```

结论：

- CharXiv 平均图片大小约 72.7KB。
- 当前 arxiv chart consensus 图片平均约 500-531KB，是 CharXiv 的约 7 倍。
- 当前 arxiv chart 的最大图片约 16-17MB，是 CharXiv 最大图的约 65 倍。
- CharXiv 图基本被规整到最大边 1024；当前 arxiv chart 里有 15000x9353 这种超大图。

这能解释为什么 CharXiv/Stepeval 场景单条调用感觉很快，而当前批量 Kimi thinking 会慢且容易限流。当前 pipeline 应在送模型前做图片压缩/缩放，建议默认 `max_side=1600`，必要时对小字题保留高分辨率版本。

## 11. 建议的下一步

不要直接扩到 50k。建议先做 edit2 pilot：

1. 修改 `scripts/generate_questions.py` 的 prompt。
2. 调整 task type 和配比，降低简单读取题。
3. 强制每种 task type 的 `answer_type`。
4. 新增本地 question 质量过滤器，过滤 title/axis/legend-only 问题。
5. 对数值题和趋势题改 consensus 归一化逻辑。
6. 重新生成每组 500 条 pilot question。
7. 再统计：
   - `hard` 占比
   - 数值题占比
   - title/axis/legend-only 占比
   - trend/cross-panel/multistep 通过率
   - manual spot check 正确率

建议通过标准：

```text
hard 或 medium-hard >= 30%
纯 title/axis/legend-only <= 10%
numeric/counting answer_type >= 25%
trend/cross-panel/multistep >= 50%
consensus 后简单读取题 <= 20%
人工抽检 answer correctness >= 85%
```
