# Chart / Figure Benchmark Notes

本文档总结 5 个与 chart / scientific figure understanding 相关的 benchmark，重点关注：

- 数据来源与规模
- 问题类型与监督信号
- 标注 / 生成 / 质检流程
- 评测方式
- 对我们从 arXiv 构造 chart VQA / caption / reasoning SFT 数据的启发

在线阅读时间：2026-06-10。

## 0. 总览

| Benchmark | 核心目标 | 数据来源 | 规模 | 主要问题形式 | 对 SFT 数据生成的启发 |
| --- | --- | --- | ---: | --- | --- |
| CharXiv | 真实科研 chart 理解 | arXiv 科研论文图表 | 2,323 charts；每图 4 descriptive + 1 reasoning | open-vocabulary short answer | arXiv chart 数据要拆成 descriptive / reasoning；必须有人审或强 verifier |
| ChartMuseum | 强调视觉推理瓶颈 | 184 个真实网页来源 | 1,162 QA；928 unique images | short-answer chart QA | 区分 textual / visual / visual-text / synthesis reasoning，避免只训练文本抽取 |
| ChartQAPro | 更真实、多样的 chart QA | 真实网页、dashboard、infographic 等 | 1,341 charts；1,948 QA | reasoning、conversational、MCQ、hypothetical、fact-checking、unanswerable | 增加多图、上下文、反事实、不可回答问题，贴近真实用户需求 |
| LAB-Bench | 生物科研任务能力 | 论文、图、表、数据库、序列、protocol | 2,457 MCQ；其中 FigQA 226、TableQA 305 | biology research multiple-choice QA | 科研图表任务可要求 image-only，不给 caption/paper text，逼模型真正读图 |
| FigureQA | 合成图表视觉推理 | 程序合成 scientific-style figures | >100k images；>1M QA | 15 templates；yes/no | 合成数据适合大规模可验证训练，但需要和真实图混合，避免过于模板化 |

## 1. CharXiv

论文：**CharXiv: Charting Gaps in Realistic Chart Understanding in Multimodal LLMs**  
链接：

- Paper: https://arxiv.org/abs/2406.18521
- Project: https://charxiv.github.io/
- GitHub: https://github.com/princeton-nlp/CharXiv
- Dataset: https://huggingface.co/datasets/princeton-nlp/CharXiv

### 核心问题

CharXiv 认为已有 chart benchmark 往往过于简单、同质化、模板化，导致模型在旧 benchmark 上看起来接近饱和，但面对真实科研论文中的复杂 chart 时明显退化。论文报告：已有强开源模型在旧 benchmark 上可能看似超过专有模型，但一个轻微改变 chart 或 question 的 stress test 就可让性能下降最多 34.5%。

### 数据构成

CharXiv 包含：

- 2,323 张真实科研 chart，来自 arXiv preprints。
- 每张图配 4 个 descriptive questions。
- 每张图配 1 个 reasoning question。
- 总体超过 10k 个问题。
- 项目页的 validation set 是 1,000 charts / 5,000 questions。

HuggingFace 数据卡说明每张 chart 的 4 个 descriptive questions 中，3 个 answerable，1 个 intentionally unanswerable；reasoning question 也是 open-vocabulary short answer。注意：数据卡明确写明 CharXiv intended to be used to evaluate models only，不允许用于训练模型。

### 问题类型

CharXiv 分成两大类。

**Descriptive questions**

用于评估模型是否能抽取和聚合基础图表元素。论文设计了 19 个模板，覆盖 5 类能力：

- information extraction：标题、x/y axis label、legend label、tick label 等。
- enumeration：数 tick、数 legend label、数 lines 等。
- pattern recognition：趋势、交叉、数据形态。
- counting：数 chart element。
- compositionality：多 subplot 场景下先定位子图，再抽取对应元素。

这类问题看似简单，但真实 arXiv 图表经常有共享坐标轴、多 panel、图例复杂、字体小、元素密集，所以仍然困难。

**Reasoning questions**

用于评估模型是否能综合复杂视觉元素进行推理。典型需求包括：

- 比较多个 series / method。
- 从视觉位置或趋势判断哪个更大、更高、更稳定。
- 结合多个 subplot 或多个 visual encoding。
- 读近似数值或进行简单计算。
- 答案通常是短文本、label、数字或短语。

### 标注与质检

CharXiv 的核心价值是高质量人工筛选：

- chart 和 question 都由 human experts handpicked / curated / verified。
- Descriptive 使用模板，但不是简单 synthetic chart 模板，而是套在真实复杂科研图上。
- Reasoning 更偏人工构造，强调 chart-specific、short-answer、可验证。
- 每个 chart 的 source arXiv id 被保留，方便追踪来源。

### 评测结果与结论

论文报告：

- GPT-4o reasoning accuracy：47.1%。
- InternVL Chat V1.5 reasoning accuracy：29.2%。
- Human performance：80.5%。

结论是：当前 MLLM 的真实 chart reasoning 能力被旧 benchmark 高估，尤其是复杂科研图、多 subplot、多 series、密集视觉元素场景。

### 对我们数据生成的启发

CharXiv 是最贴近我们“从 arXiv 做 chart SFT 数据”的 benchmark。可复用的设计：

- 每张图生成多个 descriptive QA + 1-2 个 reasoning QA。
- Descriptive 不是低价值任务，它能训练模型可靠识别 axis、legend、tick、subplot。
- Reasoning answer 应保持短、可验证，避免长篇泛化结论。
- 必须保留 `arxiv_id / figure_id / source_path / evidence_source`。
- 应加入 unanswerable 问题，训练模型不要胡编不存在的 title、axis、legend、数值。

不能直接拿 CharXiv 训练，因为其 dataset card 明确限定 evaluation-only。我们可以学习 taxonomy 和流程，但训练数据应从自己的 arXiv 抽图流程重新生成。

## 2. ChartMuseum

论文：**ChartMuseum: Testing Visual Reasoning Capabilities of Large Vision-Language Models**  
链接：

- Paper: https://arxiv.org/abs/2505.13444
- HTML: https://arxiv.org/html/2505.13444v3
- Project / leaderboard: https://chartmuseum-leaderboard.github.io/
- GitHub: https://github.com/Liyan06/ChartMuseum
- Dataset: https://huggingface.co/datasets/lytang/ChartMuseum

### 核心问题

ChartMuseum 认为 chart QA 的关键不是“能否把图中文字 OCR 出来”，而是模型是否具备真正的 visual reasoning。论文明确区分：

- textual reasoning：几乎只靠 chart 中显式文字即可推理。
- visual reasoning：必须看视觉关系，例如位置、轨迹、形状、密度、颜色、相对大小。
- text/visual reasoning：文本或视觉路径都能解。
- synthesis reasoning：必须同时结合文本和视觉推理。

### 数据构成

ChartMuseum 包含：

- 1,162 个 `(image, question, short answer)` tuples。
- 928 unique real-world images。
- 来源覆盖 184 个网站。
- dev / test = 162 / 1000。
- 平均图片尺寸约 1590 x 1243 px。
- 问题平均长度 26.7 tokens，答案平均长度 2.9 tokens。

测试集按 reasoning type 的数量：

- Visual：510
- Synthesis：133
- Visual/Text：234
- Text：123

### 标注流程

ChartMuseum 由 13 名 CS researchers 标注。每条数据经历：

1. 选择高质量、有趣的 chart。
2. 人工创建 question-answer pair。
3. 第一作者做 independent quality review。
4. 与标注者迭代讨论，修正问题清晰度、答案客观性、分类一致性。

论文特别强调：ChartMuseum 的问题不是 LLM 生成后再人工改，而是 researchers without assistance from LLMs 人工策划。这一点是为了避免 LLM 生成问题带来的模板化和语言偏差。

### 视觉任务 taxonomy

论文进一步分析 visual reasoning 失败模式，主要任务包括：

- symbol selection：根据颜色、marker、符号定位对象。
- visual comparison：比较长度、大小、高度、颜色强度、空间距离、variance、pattern。
- trajectory tracking and judgment：追踪曲线走势、波动、交叉、稳定性。
- X/Y value identification：根据坐标、legend、热力图色条等估计值。

这些类别非常适合作为我们 arXiv chart reasoning QA 的二级标签。

### 评测结果与结论

论文报告：

- Human accuracy：93.0%。
- Gemini-2.5-Pro：63.0%。
- Qwen2.5-VL-72B-Instruct：38.5%。
- Visual reasoning 问题比 text-reasoning-heavy 问题低 35%-55%。

结论：更长 CoT 或 reasoning model 只能带来很有限提升；核心瓶颈在视觉编码和视觉 grounding，而不只是语言推理。

### 对我们数据生成的启发

ChartMuseum 对训练数据设计非常关键：

- 不要只做 OCR/文本抽取题，要刻意构造 visual reasoning 题。
- 每条问题标注 `reasoning_type = textual / visual / visual_text / synthesis`。
- 对 visual 题，答案应来自视觉比较，而不是可直接 OCR 的文字。
- 需要训练模型避免错误策略：不要为了比较趋势强行估精确值，再用错误数字做计算。
- 可加入“视觉证据短解释”，例如“the blue curve remains above the orange curve near the right side”，但不要生成冗长 CoT。

## 3. ChartQAPro

论文：**ChartQAPro: A More Diverse and Challenging Benchmark for Chart Question Answering**  
链接：

- Paper: https://arxiv.org/abs/2504.05506
- HTML: https://arxiv.org/html/2504.05506v2
- ACL Anthology: https://aclanthology.org/2025.findings-acl.978/
- GitHub: https://github.com/vis-nlp/ChartQAPro

备注：arXiv v2 写 157 diverse sources；ACL 页面摘要里有版本差异，写 99 diverse sources。本文主要按 arXiv v2 的完整正文和表格记录。

### 核心问题

ChartQAPro 认为 ChartQA 等旧 benchmark 已经饱和，而且真实世界 chart 的形态远比 bar/line/pie + factoid QA 复杂。它重点补充：

- dashboard
- infographic
- multi-chart layout
- accompanying paragraph
- conversational QA
- hypothetical QA
- multiple-choice QA
- fact-checking
- unanswerable questions

### 数据构成

ChartQAPro 包含：

- 1,341 chart images。
- 1,948 human-written / human-verified QA pairs。
- 来源覆盖 157 个 online platforms。
- 包括 Pew Research、Tableau、PPIC、Our World in Data，以及大量 web charts。

Chart type 分布：

| Type | Count |
| --- | ---: |
| Bar | 427 |
| Line | 355 |
| Pie | 29 |
| Area | 30 |
| Scatter | 8 |
| Bubble | 7 |
| Dashboard | 258 |
| Infographic | 190 |
| Other | 37 |

Question type 分布：

| Type | Count |
| --- | ---: |
| Math & Visual Reasoning | 1081 |
| Conversational | 311 |
| Fact Checking | 244 |
| Multiple Choice | 214 |
| Hypothetical | 98 |

### 数据构造流程

ChartQAPro 三阶段：

1. **Chart image collection**
   - 强调视觉多样性和主题多样性。
   - 人工选择多样格式的真实 chart。
   - 包含 dashboard、infographic、multi-series line、stacked/grouped bar 等。

2. **QA annotation**
   - 9 名 team members 协作写 QA。
   - 先由人类写 seed QA。
   - 使用 GPT-4o、Gemini、Claude 进行 VLM-assisted expansion。
   - 人类再过滤过于简单、模糊、不清楚的问题。

3. **QA review**
   - 7 名 annotators 交叉审核。
   - 初始一致率 66.17%，之后通过协商修正所有分歧。
   - 对主观估计类问题，允许小于 1% 的细微差异。

### 评测方式

ChartQAPro 使用增强版 relaxed accuracy：

- numeric answer：通常保留 5% error margin。
- year：要求 exact match，避免 2008/2009 这种年份被 5% 容忍错误放过。
- textual answer：使用 ANLS。
- MCQ / fact-checking：exact match。

论文评测 direct / CoT / PoT 三种 prompting。结果显示：

- Claude Sonnet 3.5 在 ChartQA 可达 90.5%，但在 ChartQAPro 最高只有 55.81%。
- GPT-4o、Gemini、Claude 等闭源模型明显强于开源模型。
- Qwen2-VL-7B 作为较强开源模型也只有约 37.17%。
- Chart-specific models 反而泛化很差，说明它们可能过拟合旧 benchmark 的视觉和问题类型。

### 对我们数据生成的启发

ChartQAPro 对 SFT 数据扩展很有价值：

- arXiv chart 数据不应只做单轮 factoid QA；要加入 conversational、多图、多 panel、hypothetical、fact-checking。
- 应为部分图加入 accompanying paragraph / caption / local context，明确构造 `image+context` QA。
- 应系统加入 unanswerable 问题，例如问图中没有的数据、图无法支持的结论、缺失的实验设定。
- 用多 teacher 生成扩展问题可以提高多样性，但必须人工或强 verifier 过滤。
- 训练时要避免过拟合单一 chart 风格：arXiv 图应按学科、图表类型、布局复杂度做均衡采样。

## 4. LAB-Bench

论文：**LAB-Bench: Measuring Capabilities of Language Models for Biology Research**  
链接：

- Paper: https://arxiv.org/abs/2407.10362
- HTML: https://arxiv.org/html/2407.10362v1
- Project note: https://www.futurehouse.org/research-announcements/lab-bench-measuring-capabilities-of-language-models-for-biology-research
- Dataset: https://huggingface.co/datasets/futurehouse/lab-bench

### 为什么放在这里

LAB-Bench 不是 chart benchmark，而是 biology research capability benchmark。但它包含 **FigQA** 和 **TableQA**，且问题来自真实科研论文图表，非常适合参考“科研图像 QA 数据如何构造”。

### 数据构成

LAB-Bench 总计 2,457 个 multiple-choice questions，覆盖：

| Category | Count |
| --- | ---: |
| LitQA2 | 248 |
| SuppQA | 102 |
| FigQA | 226 |
| TableQA | 305 |
| DbQA | 650 |
| ProtocolQA | 135 |
| SeqQA | 750 |
| CloningScenarios | 41 |

其中与 chart / figure 最相关的是：

- FigQA：226
- TableQA：305

### FigQA 设计

LAB-Bench 的 FigQA 评估模型理解科研论文 figure 的能力：

- 输入只有 figure image。
- 不提供 caption、paper title、paper text。
- 问题要求模型整合 figure 中多个元素。
- 类似视觉版 multi-hop QA。
- 不需要外部 tool，但需要多模态能力。

这点对我们很重要：如果我们想训练 `image-only chart reasoning`，就必须保证答案真的能从图中看出来，不能偷偷依赖 caption 或论文上下文。

### 数据生成方法

FigQA 人工构造：

- 作者或 contracted biology experts 选择 biology paper。
- 截取 figure。
- 基于 figure 内容写 question。
- 要求问题不能仅靠图中文字回答，也不能需要 caption 或 paper text。
- 对外包标注者，先由 LabelBox annotator 标注/隔离 figure image、caption bbox、image-caption bbox 关系、DOI。
- 再通过 Airtable 界面给专家写题。
- GPT-4 Turbo 被用作 brainstorm 工具，但生成内容经常有根本错误或太简单，只能作为草稿灵感。

### 评测方式

LAB-Bench 是 MCQ：

- 所有模型 0-shot CoT prompting。
- 每题加入一个 “Insufficient information to answer the question” 选项，用于评估模型拒答/不确定性。
- 报告 accuracy 和 precision，其中 precision 只统计模型没有选择 insufficient information 的覆盖部分。
- 约 80% public，20% private，用于监控 contamination。

### 对我们数据生成的启发

LAB-Bench 的重要经验：

- 科研图 QA 可以强制 image-only，训练真实读图能力。
- 专家标注比通用众包更可靠，尤其是科研图。
- LLM/VLM 适合作为 brainstorming，不适合直接当 GT。
- 应为模型提供“信息不足”出口，训练其拒答能力。
- 对高难任务，可以保留 private eval split，避免训练污染。

## 5. FigureQA

论文：**FigureQA: An Annotated Figure Dataset for Visual Reasoning**  
链接：

- Paper: https://arxiv.org/abs/1710.07300
- Microsoft Research page: https://www.microsoft.com/en-us/research/project/figureqa-dataset/
- Baseline code: https://github.com/vmichals/FigureQA-baseline

### 核心问题

FigureQA 是早期 chart/figure visual reasoning 数据集，目标是用合成 scientific-style figures 研究模型是否能理解图表元素之间的关系。

### 数据构成

FigureQA 包含：

- 超过 100,000 张 synthetic figure images。
- 超过 1,000,000 个 question-answer pairs。
- 5 类图：
  - line plots
  - dot-line plots
  - vertical bar graphs
  - horizontal bar graphs
  - pie charts
- 15 个 question templates。
- train / validation 问题答案都是 yes/no。

问题关注关系型视觉推理：

- maximum / minimum
- greater / less than
- median
- area under the curve
- smoothness / roughness
- intersection
- one-vs-one / one-vs-all comparison

### 附加监督

FigureQA 的优势是可程序化生成，因此提供强结构化 side data：

- 生成每张图的 numerical source data。
- 每个 plot element 的 bounding box annotation。
- 可用于 auxiliary objectives，例如元素定位、数据重建、结构化 chart parsing。

### 优点与限制

优点：

- 数据规模大。
- GT 完全可控。
- 数值和 bbox 可验证。
- 适合训练视觉关系、比较、计数、趋势判断。

限制：

- 合成图过于干净。
- 问题模板化。
- 答案只有 yes/no，监督信号较窄。
- 和真实 arXiv 科研图的多 panel、复杂 legend、非标准布局差距较大。

### 对我们数据生成的启发

FigureQA 更适合作为合成预训练 / 辅助训练思路，而不是直接模仿真实 benchmark：

- 可以为我们自己的合成 chart 生成 source table + bbox + QA。
- 合成数据可覆盖真实 arXiv 难以平衡的能力点，例如 line intersection、AUC、slope comparison。
- 合成数据应和真实 arXiv 图混合训练，否则模型容易只适应干净模板。
- QA 不应只做 yes/no；应扩展成 short answer、approx numeric、ranking、explanation。

## 6. 横向比较

### 数据来源

| Benchmark | 真实 / 合成 | 来源多样性 | 是否科研图 |
| --- | --- | --- | --- |
| CharXiv | 真实 | arXiv 单一大来源，但学科内多样 | 是 |
| ChartMuseum | 真实 | 184 websites | 部分是 |
| ChartQAPro | 真实 | web / Tableau / Pew / PPIC / OWID 等 | 不限科研 |
| LAB-Bench FigQA | 真实 | biology papers | 是 |
| FigureQA | 合成 | 程序生成 | scientific-style，但非真实论文 |

### 问题设计

| Benchmark | Descriptive | Reasoning | Numeric | Multi-panel / multi-chart | Unanswerable | Context |
| --- | --- | --- | --- | --- | --- | --- |
| CharXiv | 强 | 强 | 有 | 强 | 有 | 无/弱 |
| ChartMuseum | 弱 | 很强 | 有 | 有 | 少 | 无 |
| ChartQAPro | 有 | 强 | 有 | 强 | 有 | 有 |
| LAB-Bench FigQA | 无固定 descriptive | 强 | 有 | 视图而定 | 通过 insufficient option | image-only |
| FigureQA | 模板化 | 强关系推理 | 间接 | 无真实多 panel | 无 | 无 |

### 标注策略

| Benchmark | 生成方式 | 质检强度 |
| --- | --- | --- |
| CharXiv | 人工筛图 + 模板 descriptive + 人工 reasoning | human expert verified |
| ChartMuseum | 研究者人工写题，无 LLM 辅助 | independent review + 迭代讨论 |
| ChartQAPro | 人类 seed + 多 VLM 扩展 + 人工 refine | 交叉审核，解决分歧 |
| LAB-Bench | 专家人工 + 部分程序化；LLM 作 brainstorming | 专家审核 |
| FigureQA | 程序合成 | 规则可验证 |

## 7. 对 arXiv Chart SFT 数据生成的建议

结合这 5 个 benchmark，我建议我们的训练数据按以下 taxonomy 构造。

### 7.1 每张图的样本结构

对高质量 arXiv chart：

```text
1 visual caption
2-3 descriptive QA
2-4 image-only reasoning QA
0-2 paper-aware QA
0-1 unanswerable QA
```

其中：

- descriptive QA 学 CharXiv，覆盖 title / axis / legend / tick / subplot。
- visual reasoning QA 学 ChartMuseum，覆盖 visual comparison / symbol selection / trajectory tracking / value identification。
- diverse task format 学 ChartQAPro，加入 MCQ、fact-checking、hypothetical、conversational。
- image-only scientific figure QA 学 LAB-Bench FigQA，严格禁止依赖 caption。
- synthetic auxiliary 数据学 FigureQA，提供 source table / bbox / exact GT。

### 7.2 必须显式标注 evidence source

每条样本记录：

```text
evidence_source:
  image_only
  image+caption
  image+caption+context
  synthetic_source_table
  not_answerable
```

不要让 `image_only` 样本混入 caption 或 paper context 信息。

### 7.3 必须显式标注 reasoning type

建议字段：

```text
task_type:
  descriptive_extraction
  descriptive_counting
  text_in_chart
  number_in_chart
  approximate_numeric
  visual_comparison
  trajectory_tracking
  symbol_selection
  x_y_value_identification
  synthesis_reasoning
  fact_checking
  hypothetical
  conversational
  unanswerable
```

这个 taxonomy 基本覆盖 CharXiv + ChartMuseum + ChartQAPro 的核心能力。

### 7.4 训练数据和评估数据要分离

CharXiv 明确禁止用其数据训练。对所有 benchmark 都应默认只用于评估或设计参考。我们的训练数据应：

- 从自有 arXiv 抽图 pipeline 生成。
- 按 paper id 切 train / val / test。
- 去掉 benchmark 原图及相似图。
- 保留 private held-out set，参考 LAB-Bench 的 public/private split。

### 7.5 生成器不要单模型裸生成

推荐流程：

```text
chart image + OCR + caption + source metadata
  -> teacher proposal
  -> rule check
  -> second VLM verifier
  -> evidence-source leakage check
  -> human spot check
  -> export SFT JSONL
```

ChartQAPro 说明多 VLM 扩展有助于多样性；LAB-Bench 说明 LLM 生成经常有根本错误；ChartMuseum 说明完全人工策划的问题更自然、更能卡住模型。因此自动生成必须配强 verifier。

### 7.6 要训练拒答和不确定性

必须加入：

- 图中没有该信息。
- exact value 不可读，只能 approximate。
- caption 才能回答，但当前输入没有 caption。
- paper method / dataset / conclusion 无法从图像单独判断。

推荐答案风格：

```text
The chart does not provide enough information to determine the exact value. It appears to be approximately ...
```

### 7.7 合成数据的定位

FigureQA 式合成数据适合补齐能力空洞：

- line intersection
- slope / volatility
- AUC / cumulative comparison
- count / max / min / rank
- color / marker grounding
- value interpolation

但真实 SFT 主体仍应是 arXiv / real-world charts。合成数据应作为可验证辅助数据，避免模型只学会模板视觉。

## 8. 最终建议

如果目标是让 `step-3.7-flash` 这类大模型获得真实 chart understanding / reasoning 能力，不应简单做 “arXiv figure + caption”。更稳的方案是：

1. 学 CharXiv：以 arXiv 真实图为主体，做 descriptive + reasoning，保留 short verifiable answer。
2. 学 ChartMuseum：明确区分 textual vs visual vs synthesis reasoning，增加真正必须看图的视觉推理题。
3. 学 ChartQAPro：增加真实用户问题形态，包括 conversational、hypothetical、fact-checking、unanswerable、多图和 context。
4. 学 LAB-Bench：构造一批严格 image-only scientific figure QA，不给 caption/paper text，防止模型依赖上下文作弊。
5. 学 FigureQA：用合成数据补可验证结构化能力，但不要让它主导训练分布。

最关键的数据原则：

```text
真实图为主
证据来源明确
短答案可验证
视觉推理要占足比例
包含不可回答样本
自动生成必须验证
训练集避开公开 benchmark
按 paper-level 切分
```
