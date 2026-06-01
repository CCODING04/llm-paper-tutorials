# 📖 GPT-2: Language Models are Unsupervised Multitask Learners

> **论文**：Radford et al., 2019 (OpenAI) | ICML 2019
>
> **一句话总结**：足够大的语言模型在足够多、足够杂的数据上训练后，能以 zero-shot 方式完成多种 NLP 任务——不需要微调。

---

# 第一层：鸟瞰

## 🎯 核心贡献

1. **Zero-shot 多任务学习的系统验证**：首次系统证明语言模型可以不做任何微调就完成翻译、问答、摘要、常识推理等多种任务——但效果参差不齐
2. **Scaling 的早期实证**：性能与模型参数量呈近似线性关系——更大模型 = 更强能力，在测试的四个规模（117M-1542M）内没有饱和迹象
3. **WebText 数据集**：通过 Reddit 社区投票筛选高质量网页数据，证明了数据质量和多样性的重要性
4. **Prompt 范式的开创**：用自然语言描述任务的方式（如 "TL;DR:" 做摘要），成为后来 prompt engineering 和 in-context learning 的基础

## 📍 知识网络定位

```
Word2Vec (2013) → 静态词嵌入
ELMo (2018.02) → 上下文词嵌入（LSTM，浅层拼接）
GPT-1 (2018.06) → Transformer 解码器预训练 + 微调
BERT (2018.10) → Transformer 编码器预训练 + 微调（双向）
         ↓
   【GPT-2 (2019.02)】→ Transformer 解码器预训练 + zero-shot prompt
         ↓                    （回到单向解码器但走不同路线）
   GPT-3 (2020.06) → 175B，few-shot prompt（验证 scaling + 弥补 zero-shot 的不足）
   Scaling Laws (2020.09) → 系统化模型大小/数据量/计算量的关系
   Chinchilla (2022.03) → 质疑数据量不够（计算最优需要更多数据）
   InstructGPT (2022.01) → GPT-3 + RLHF 对齐
   ChatGPT (2022.11) → 产品化
```

**一句话给面试官**：GPT-2 的核心论点是"语言模型在足够多的数据上训练后，会隐式学会多种任务"。虽然 zero-shot 效果还有限，但它开创了 prompt 范式和 scaling 思想，直接导致了 GPT-3 和整个大模型时代。

---

# 第二层：精读

## 1. 引言：逐段论证链

GPT-2 的引言是整篇论文**最精彩的部分**——它的论证链值得逐段拆解。

### 第一段：问题定义

> "Machine learning systems now excel (in expectation) at tasks they are trained for...Yet these systems are brittle and sensitive to slight changes in the data distribution...Current systems are better characterized as **narrow experts** rather than **competent generalists**."

**核心论点**：当前 ML 系统是窄专家，对分布偏移敏感。我们想要通才。

> 💡 注意论文用 "(in expectation)" 来限定——即使在训练分布上表现好，也只是**期望**上好，不是每条样本都好。这种措辞暗示了论文对当前范式的根本怀疑。

### 第二段：当前范式的局限

> "The dominant approach to creating ML systems is to collect a dataset...train a system to imitate these behaviors...then test on IID held-out examples."

论文指出这个范式在**输入多样性**面前暴露了问题——image classifier 被对抗样本骗、reading comprehension 系统被稍微改写的题目迷惑。这说明 IID 评估是"虚高"的。

### 第三段：多任务学习的困境

> "Multitask learning is a promising framework...the two most ambitious efforts to date have trained on a total of 10 and 17 (dataset, objective) pairs."

**关键论证**（这是整篇论文最核心的一段）：

> "From a **meta-learning** perspective, each (dataset, objective) pair is a **single training example** sampled from the distribution of datasets and objectives. Current ML systems need hundreds to thousands of examples to induce functions which generalize well. This suggests that multitask training may need just as many effective training pairs to realize its promise."

拆解这段论证：
1. 把多任务学习重新定义为 **meta-learning**——每个 (dataset, objective) 对是一个"训练样本"
2. ML 系统需要成百上千个训练样本才能泛化
3. 所以多任务学习需要成百上千个 (dataset, objective) 对
4. 暴力收集这么多数据集+设计目标函数**不现实**

> ❓ **批判**：这个论证非常巧妙但也有偷梁换柱的嫌疑。"多任务学习 = meta-learning"的映射不一定成立——多任务学习中每个任务有大量样本，而 meta-learning 中每个"任务"只有少量样本。GPT-2 的论点隐含假设了"多任务学习需要大量不同任务"，但也许只需要少量精心设计的任务就能泛化（后来的 T5、mT5 证明了这一点）。

### 第四段：预训练-微调路线的历史

论文梳理了迁移学习在 NLP 中的演化：Word2Vec → ELMo → GPT/BERT。趋势是"越来越灵活的迁移"。

### 第五段：语言建模作为统一框架

> "Since the supervised objective is the same as the unsupervised objective but only evaluated on a subset of the sequence, **the global minimum of the unsupervised objective is also the global minimum of the supervised objective.**"

这是论文最核心的数学论证：

**形式化**：语言模型优化 $\max \sum_{i=1}^{n} \log p(s_i | s_1, ..., s_{i-1})$

监督学习优化的是同一个目标，但只在**子集**上评估。因此：语言建模的全局最优 → 包含了所有监督任务的全局最优。

> ❓ **这个论证成立吗？** 理论上成立。但实践中有两个问题：
> 1. 我们永远无法达到全局最优
> 2. "在子集上评估"意味着训练数据中需要包含足够多的该子集分布。如果训练数据中几乎没有法语，翻译质量必然差——这直接解释了 GPT-2 翻译弱的原因。

### 第六段：核心假设

> "Our speculation is that a language model with sufficient capacity will begin to learn to infer and perform the tasks demonstrated in natural language sequences in order to better predict them."

这是论文的"赌注"——如果语言模型足够大，训练数据足够多足够杂，它就会隐式学会各种任务。

---

## 2. 方法：逐节深入

### 2.1 数据集：WebText

| 特性 | 值 | 设计理由 |
|------|-----|---------|
| 来源 | Reddit 外链（≥3 karma） | 社区投票 = 天然质量筛选 |
| 文档数 | ~800 万 | 足够大 |
| 总大小 | 40GB (~80 亿 token) | 比 GPT-1 的 BooksCorpus (5GB) 大 8 倍 |
| 文本提取 | Dragnet + Newspaper3k | 从 HTML 中提取正文 |
| 去重 | 已去重 + 启发式清洗 | 减少记忆 |
| 特殊处理 | **去除所有 Wikipedia** | 避免 benchmark 数据污染 |

#### 和同期数据集的对比

| 数据集 | 大小 | 来源 | 特点 |
|--------|------|------|------|
| BooksCorpus (GPT-1) | 5GB | 7,000 本书 | 长文本、连贯性好，但领域窄 |
| Wikipedia (BERT) | 2.5B 词 | 百科全书 | 高质量、结构化，但风格单一 |
| Common Crawl | ~PB 级 | 全网爬取 | 量大但质量极差 |
| **WebText (GPT-2)** | **40GB** | **Reddit 精选外链** | **量质平衡** |

> ❓ **为什么不用 Common Crawl？** 论文明确说数据质量太差。WebText 的设计哲学是：**不假设要做什么任务，尽可能收集多样化、高质量的自然文本**。

> ❓ **WebText 有什么偏差？** Reddit 用户偏向年轻、英语、科技、男性。小语种覆盖不足（法语只有 ~10MB），非技术领域覆盖不足。这些偏差直接导致了 GPT-2 的弱点——翻译差、摘要差。

### 2.2 输入表示：Byte-level BPE

这是 GPT-2 一个**被低估的创新**。

#### 算法详解

1. **基础词表 = 256**（所有可能的字节值 0x00-0xFF）
2. 在训练语料上统计相邻字节对频率
3. 贪心合并最高频的字节对，生成新 token
4. 重复直到词表达到目标大小（50,257）

#### 防止跨类别合并

论文限制 BPE 不跨字符类别合并（字母不跟标点合并）。为什么？否则 "dog"、"dog!"、"dog?" 会变成三个不同 token，浪费词表空间。

#### 和 BERT WordPiece 的全面对比

| 维度 | WordPiece (BERT) | Byte-level BPE (GPT-2) |
|------|-----------------|----------------------|
| 起点 | 词级别 | **字节级别** |
| 词表大小 | 30,000 | 50,257 |
| OOV | `##` 拼接 | **永远不会 OOV** |
| 多语言 | 每种语言建词表 | 天然支持所有语言 |
| 代码/URL/emoji | 处理差 | 天然支持 |

> 💡 这个设计决策经受住了时间考验——GPT-3/4 和几乎所有现代大模型都用 byte-level BPE 或其变体（如 SentencePiece）。

### 2.3 模型架构

GPT-2 基于 GPT-1，做了**三个关键改动**。

#### 改动 1：Pre-LN（Layer Norm 前置）

**Post-LN（GPT-1）**：`output = LayerNorm(x + sublayer(x))`
**Pre-LN（GPT-2）**：`output = x + sublayer(LayerNorm(x))`

**为什么 Pre-LN 更稳定？**

在 Post-LN 中，残差连接被 LayerNorm "包裹"了——梯度必须通过 LayerNorm 才能回传。在 Pre-LN 中，残差连接是"干净的"——梯度可以直接走残差路径回传，形成"梯度高速公路"。

> 💡 Pre-LN 成为了大模型的事实标准。GPT-3、LLaMA、Mistral、DeepSeek 全部使用 Pre-LN 或其变体。

#### 改动 2：残差初始化缩放

每层残差初始化为 $1/\sqrt{N}$（N = 层数）。确保深层和浅层的输出量级在训练开始时一致。

#### 改动 3：上下文长度翻倍（512 → 1024）

网页文本比书籍片段更长，需要更长的上下文。

#### 四种规模

| 模型 | 参数量 | 层数 | d_model | 注意力头 | FFN | d_head |
|------|--------|------|---------|---------|-----|--------|
| Small | 117M | 12 | 768 | 12 | 3072 | 64 |
| Medium | 345M | 24 | 1024 | 16 | 4096 | 64 |
| Large | 762M | 36 | 1280 | 20 | 5120 | 64 |
| **XL** | **1542M** | **48** | **1600** | **25** | **6400** | **64** |

> 💡 Small 和 GPT-1（117M）/ BERT_BASE（110M）参数量一样——方便对比。d_head 始终 = 64，和原始 Transformer 一样。

---

## 3. 实验：每个实验验证了什么？

### 3.1 语言建模（Table 3）

| 数据集 | 之前 SOTA | GPT-2 XL | 提升 |
|--------|----------|----------|------|
| **LAMBADA** | 99.8 PPL | **8.6 PPL** | **-91.4** |
| WikiText-2 | 39.14 PPL | 18.34 PPL | -53% |
| PTB | 46.54 PPL | 35.76 PPL | -23% |
| 1BW | 23.7 PPL | 42.16 PPL | **退步** |

**LAMBADA 为什么降幅最大？** LAMBADA 测试"预测段落最后一个词"——和自回归 LM 的训练目标完美对齐。

**1BW 为什么没赢？** 1BW 打乱了句子，破坏了长距离结构——而长距离依赖正是 GPT-2 的强项。

### 3.2 Children's Book Test（Figure 2）

- Common Nouns：~93%（Human ~96%）
- Named Entities：~89%（Human ~92%）

Named Entities 的提升更持续（1542M 还在上升）——实体区分需要更精确的上下文推理。

### 3.3 阅读理解：CoQA

GPT-2 zero-shot 55 F1，匹配 3/4 baseline（它们用了 127K+ 训练数据）。

**论文自己的分析**（最诚实的部分）：GPT-2 经常用简单启发式（如用人名回答 "who" 问题），不是真正理解。超过弱 baseline 不能证明真正的理解能力。

### 3.4 翻译：WMT-14——最弱的结果

| 方向 | GPT-2 XL | 无监督 SOTA | 有监督 SOTA |
|------|----------|------------|------------|
| En→Fr | 5 BLEU | 33.5 BLEU | 40+ BLEU |
| Fr→En | 11.5 BLEU | 33.5 BLEU | 40+ BLEU |

**Fr→En 远好于 En→Fr**——因为 GPT-2 是英语语言模型，英语生成是强项。法语数据只有 ~10MB，比无监督翻译研究用的少 **500 倍**。直接验证了"能力取决于数据覆盖度"的论点。

### 3.5 摘要：CNN/DM——几乎等于随机

加 "TL;DR:" 提示后（21.40 ROUGE）只比随机选 3 句（20.98）好 0.42——**几乎可以忽略**。但 prompt 确实激活了任务行为（无提示只有 15.03，差距 6.4）。

> ❓ **为什么摘要这么弱？** WebText 中摘要的"自然演示"很少；摘要需要**压缩改写**能力，而语言模型训练的是**续写**；CNN/DM 是新闻风格，WebText 偏科技博客。

### 3.6 常识推理：Winograd Schema——最令人惊讶

~71%（Partial Scoring），超过 SOTA（63%）约 8%。但数据集只有 273 个样本，统计波动大。

### 3.7 规模效应（Figure 1）——论文最核心的图

![Figure 1](./images/123b2e3711e5e6107c8589d946c274291aeb2e46dbc6aa5001a21170070bbf69.jpg)

四面板折线图，四个任务都呈近似线性上升。关键观察：
- **Translation**：117M 几乎为 0，说明翻译能力需要一定规模才能"涌现"
- **Summarization**：增长最慢，1542M 也只有 22 ROUGE——摘要需要更复杂的能力
- **所有面板都没有饱和**——暗示继续增大规模会有更多提升

这张图同时证明了两个论点：
1. ✅ Scaling 有效——性能随规模单调增长
2. ⚠️ Zero-shot 还不够好——需要更多规模（GPT-3 的动机）

### 3.8 训练/测试 PPL（Figure 4）——还在 underfitting

| 参数量 | 训练 PPL | 测试 PPL | 差距 |
|--------|---------|---------|------|
| 117M | ~16 | ~16.5 | 0.5 |
| 345M | ~12.2 | ~12.8 | 0.6 |
| 762M | ~10.2 | ~11.0 | 0.8 |
| 1542M | ~8.8 | ~10.2 | 1.4 |

两条曲线近似平行下降——**没有过拟合**。但差距在缓慢扩大。即使 1542M 还在 underfitting——这直接推动了 GPT-3（175B）的诞生。

### 3.9 数据污染分析（Section 4 + Figure 5）

8-gram 重叠的 CDF 分析：GPT-2 生成文本的重叠**远低于**真实文本——说明模型在生成而非记忆。

排除重叠样本后 LAMBADA PPL 只从 8.6 变到 8.7——**性能主要来自泛化**。

---

# 第三层：批判性思考

## 🤔 设计决策分析

### 为什么用解码器而不是编码器？

论文的立场：如果目标是**不做微调**就完成任务，生成式模型更自然——只需要在输入后面接续文本。

> ❓ **但为什么后来大模型几乎都选了解码器？** 三个原因：
> 1. **生成能力隐含理解能力**——能流利续写文本的模型必然理解了上下文
> 2. **理解能力不隐含生成能力**——BERT 的编码器无法自然生成文本
> 3. **Prompt 范式的统一性**——所有任务都统一为"文本续写"，不需要为每个任务设计输出层

### 为什么不用微调？

论文：微调需要标注数据，限制了通用性。

> ❓ **批判**：GPT-2 的 zero-shot 在多数任务上不如微调方法。后来的发展证明"不用微调"的路线需要**极大的模型**（GPT-3 175B）才能接近微调效果，而 RLHF（InstructGPT/ChatGPT）实际上是一种"微调"——只不过是用人类反馈而非标注数据。

### Byte-level BPE 的局限性

> ❓ **批判**：byte-level BPE 在处理中文等语言时效率很低——一个汉字可能需要 3 个 UTF-8 字节编码，而 WordPiece 可以把常用字直接编码为一个 token。这也是为什么后来有专门为中文优化的分词器。

## ⚠️ 局限性

### 论文承认的
1. Zero-shot 在摘要、翻译上还很弱
2. 模型仍然 underfitting WebText
3. 生成文本有时重复、缺乏全局一致性

### 自己发现的
1. **数据偏差**：WebText 偏向英文、科技、Reddit 社区
2. **评估不够严格**：很多 zero-shot 方式很"粗糙"
3. **数据污染**：虽然分析了 8-gram 重叠，但更细粒度的污染（如语义级）无法检测
4. **没有和微调方法全面对比**：论文主要和 zero-shot baseline 比
5. **生成质量的评估缺失**：论文展示了生成样本但没用系统化方法评估质量

## 🎯 面试视角

**Q1: GPT-2 和 BERT 的核心区别？**

> **标准回答**：三个维度——架构（解码器 vs 编码器）、预训练目标（自回归 LM vs MLM）、下游方式（zero-shot prompt vs 微调）。
>
> **追问：为什么 GPT 路线赢了？** 因为生成能力隐含理解能力，反之不然。解码器可以通过文本续写统一所有任务，而编码器需要为每个任务设计输出层。

**Q2: 论文的核心论证是什么？你同意吗？**

> **标准回答**："监督目标 = 无监督目标的子集，因此语言建模的全局最优包含了所有任务的最优"。理论上成立但实践有限制——需要训练数据覆盖任务分布。GPT-2 翻译弱就是因为法语数据太少。
>
> **追问：这个论点在 ChatGPT 时代还成立吗？** 部分成立。ChatGPT 的能力仍然受限于训练数据分布。RLHF 改变了优化目标（不再只是语言建模），但基础能力仍来自预训练。

**Q3: Pre-LN 为什么比 Post-LN 好？**

> **标准回答**：Pre-LN 让残差连接保持"干净"——梯度可以直接走残差路径回传，形成"梯度高速公路"。Post-LN 中残差连接被 LayerNorm "包裹"，深层梯度容易被稀释。
>
> **追问：有什么代价吗？** Pre-LN 的输出分布可能不如 Post-LN 标准化——因为最后一层的输出没有经过 LayerNorm。实践中通常在最后加一个额外的 LayerNorm 来弥补。

**Q4: GPT-2 的 scaling 实验发现了什么？**

> **标准回答**：性能与参数量呈近似线性关系，没有饱和。训练/测试 PPL 曲线平行下降，说明 1.5B 还在 underfitting。这直接推动了 GPT-3（175B）的诞生。
>
> **追问：后来有什么补充？** Scaling Laws（Kaplan et al., 2020）系统化了这个观察，发现 PPL 和参数量/数据量呈**幂律关系**。Chinchilla（2022）进一步证明 GPT-2/3 的数据量不够——计算最优需要 20 倍数据。

**Q5: GPT-2 的 zero-shot 方式有什么局限？**

> **标准回答**：
> 1. 任务指定不精确——用 "TL;DR:" 指定摘要太粗糙
> 2. 没有反馈机制——模型无法纠正自己的错误
> 3. 效果差距大——阅读理解和常识推理还行，翻译和摘要很弱
>
> **GPT-3 怎么解决的？** Few-shot prompt（给出几个示例）——更精确地指定任务。RLHF 进一步让模型学会遵循指令。

---

# 第四层：知识网络

## 📅 时间线

```
Word2Vec (2013) → 静态词嵌入
ELMo (2018.02) → 上下文词嵌入（LSTM）
GPT-1 (2018.06) → 预训练+微调（解码器）
BERT (2018.10) → 预训练+微调（编码器）
    【GPT-2 (2019.02)】→ 预训练+zero-shot prompt（解码器）
T5 (2019.10) → 编码器-解码器，统一 text-to-text
XLNet (2019.06) → 排列语言模型（试图结合双向和自回归）
GPT-3 (2020.06) → 175B，few-shot（直接延续 GPT-2）
Scaling Laws (2020.09) → 系统化 scaling 观察
Chinchilla (2022.03) → 计算最优 scaling
InstructGPT (2022.01) → GPT-3 + RLHF
ChatGPT (2022.11) → 产品化
```

## ↔️ 同期对比（2018-2019 的大模型竞赛）

| 维度 | GPT-2 (2019.02) | BERT (2018.10) | T5 (2019.10) | XLNet (2019.06) |
|------|----------------|---------------|-------------|-----------------|
| 架构 | 解码器 | 编码器 | 编码器+解码器 | 排列+解码器 |
| 方向 | 单向 | 双向 | 编码双向+解码单向 | 排列（部分双向） |
| 预训练 | 自回归 LM | MLM+NSP | Span corruption | 排列 LM |
| 下游 | Zero-shot | Fine-tuning | Fine-tuning | Fine-tuning |
| 参数量 | 1.5B | 340M | 11B | 340M |
| 数据 | 40GB WebText | 16GB Books+Wiki | 800GB C4 | 16GB Books+Wiki |

## 🔗 知识关联

- **llm-math-foundations Ch02**：GPT-2 的训练目标 = 最大似然估计 = 链式法则分解 $p(x) = \prod p(s_i | s_{<i})$
- **llm-math-foundations Ch03**：对数线性 scaling 关系 → 后来被 Scaling Laws 系统化为幂律
- **本系列 02-BERT**：GPT-2 的"对手"——编码器 vs 解码器路线分歧的起点
- **本系列 04-GPT-3**：GPT-2 的直接延续——175B，从 zero-shot 到 few-shot
- **本系列 05-Chinchilla**：质疑 GPT-2/GPT-3 的数据量不够

## 📊 GPT-2 的遗产

| GPT-2 的创新 | 被谁继承 | 被谁抛弃/修正 |
|-------------|---------|-------------|
| Zero-shot prompt | GPT-3 (few-shot) | Zero-shot 被 few-shot 取代 |
| Scaling 思想 | 所有后续大模型 | 没有被抛弃 |
| Byte-level BPE | GPT-3/4, LLaMA | 没有被抛弃 |
| Pre-LN | 所有后续大模型 | Post-LN 被抛弃 |
| WebText 数据策略 | Common Crawl + 过滤 | Reddit-only 被更大数据源取代 |
| "不做微调"的立场 | GPT-3 (部分) | 被 RLHF (InstructGPT) 修正 |

---

## ❓ 深度思考题

1. **概念题**：论文说"监督目标 = 无监督目标的子集"——这个论点的成立条件是什么？如果训练数据完全不含某种任务的分布（如数学推理），这个论点还能成立吗？

2. **设计题**：如果你来设计 GPT-2 的预训练数据，你会做哪些不同于 WebText 的选择？如何平衡数据多样性和数据质量？你会主动注入某些"弱监督"信号吗？

3. **批判题**：GPT-2 在 CoQA 上 55 F1 但分析发现用了简单启发式——这说明 zero-shot 评估有什么根本性问题？设计 prompt 让模型"做任务"和真正"理解任务"之间有什么差距？

4. **面试题**：面试官问"如果 BERT 和 GPT-2 同时面试，你会选谁做你的 NLP 系统？"你怎么回答？分别从 2019 年和 2026 年的角度回答。

5. **拓展题**：GPT-2 选择解码器（单向）而 BERT 选择编码器（双向）。后来的大模型几乎都选了解码器。从信息论角度，单向注意力比双向注意力**少了一半信息**。为什么"少信息"的路线反而赢了？这说明评估"信息量"不是预测最终胜出的正确维度——那什么是？

6. **实现题**：Byte-level BPE 的"防止跨类别合并"规则具体怎么实现？如果你要自己实现，会怎么定义"字符类别"？对中文、日文等非拉丁文字有什么影响？

7. **哲学题**：论文标题说 "Language Models are **Unsupervised** Multitask Learners"——但训练数据的分布决定了模型能学什么。互联网文本本身就是人类"监督"的产物（有人写了翻译对照、有人写了问答）。这真的是"无监督"吗？还是只是把监督信号隐藏在了数据分布中？如果真是如此，"无监督多任务学习"和"隐式监督学习"有什么区别？

8. **对比题**：BERT 的消融实验证明**双向性**是关键，GPT-2 的实验证明**规模**是关键。这两个结论矛盾吗？如何统一理解？（提示：考虑规模和架构是不是正交的维度——大模型必须选单向还是双向吗？如果 GPT-2 用了双向注意力，会不会更好？)

---

## 📚 延伸阅读

| 论文 | 年份 | 和 GPT-2 的关系 |
|------|------|----------------|
| **GPT-3** (Brown et al.) | 2020 | 直接延续——175B，few-shot 验证 scaling |
| **Scaling Laws** (Kaplan et al.) | 2020 | 系统化 GPT-2 的 scaling 观察，发现幂律关系 |
| **Chinchilla** (Hoffmann et al.) | 2022 | 质疑数据量不够——计算最优需要 20x 数据 |
| **GPT-1** (Radford et al.) | 2018 | GPT-2 的前身——预训练+微调范式 |
| **BERT** (Devlin et al.) | 2018 | 竞争对手——编码器+双向+微调 |
| **T5** (Raffel et al.) | 2019 | 同期的编码器-解码器方案，统一 text-to-text |
| **InstructGPT** (Ouyang et al.) | 2022 | 解决 GPT-2 开始的"不对齐"问题 |
| **WebText 的遗产** | — | 后来 OpenAI 用更大数据（Pile, SlimPajama 等）验证了"更多数据"的价值 |