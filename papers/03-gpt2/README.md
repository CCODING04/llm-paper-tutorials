# 📖 GPT-2: Language Models are Unsupervised Multitask Learners

> **论文**：Radford et al., 2019 | OpenAI
> 
> **一句话总结**：足够大的语言模型在足够多的数据上训练后，能以 zero-shot 方式完成多种 NLP 任务——不需要微调。

---

## 1 🎯 这篇论文在解决什么问题？

2018 年的 NLP 世界是这样的：
- **BERT**（你刚读完的）开创了"预训练 + 微调"范式
- 但每个下游任务仍然需要**有标注的数据**来微调
- 多任务学习虽然在理论上很好，但实践中很难扩展到几十上百个任务

GPT-2 的核心论点是：**如果你训练一个足够大的语言模型在足够多、足够杂的数据上，它会自己学会做各种任务**——完全不需要专门训练。

> 💡 **类比**：就像一个人读了整个互联网的文本，自然而然就学会了翻译、问答、摘要……因为你见过太多"英法对照文本""问答对"之类的自然演示。

---

## 2 🏗️ 核心思想：语言建模就是多任务学习

### 2.1 为什么语言模型能做各种任务？

论文的核心观察：

$$p(x) = \prod_{i=1}^{n} p(s_n | s_1, ..., s_{n-1})$$

语言模型的目标是预测下一个 token。如果你训练数据里包含：
- 英法对照文本 → 模型学会了**翻译**
- 问答对 → 模型学会了**问答**
- 文章 + 摘要 → 模型学会了**摘要**

关键洞察：**监督学习的目标（预测 output）和无监督语言建模的目标（预测下一个 token）在本质上是一样的**——都是条件概率估计。

> ❓ **和 BERT 的本质区别是什么？**
> 
> BERT：预训练 → 加分类头 → 微调（需要标注数据）
> GPT-2：预训练 → 直接用（zero-shot，不需要任何标注数据）
> 
> GPT-2 用的是 **prompt** 的方式：把任务描述和输入拼成文本，让模型续写。比如翻译就是：
> ```
> Translate to French: "Hello world" = "Bonjour le monde"
> ```
> 模型自然接续后面的翻译。

### 2.2 从 BERT 到 GPT-2 的范式转变

| 维度 | BERT | GPT-2 |
|------|------|-------|
| 架构 | Transformer **编码器** | Transformer **解码器** |
| 预训练目标 | MLM（双向） | 自回归 LM（单向） |
| 下游任务方式 | 微调（加分类头） | **Zero-shot**（直接 prompt） |
| 核心论点 | 双向预训练更好 | 更大的模型 + 更多数据 = 涌现能力 |
| 代表能力 | 理解 | **生成** |

---

## 3 📊 数据集：WebText

### 3.1 数据收集

GPT-2 使用了一个新数据集 **WebText**：
- 从 Reddit 上获得高质量外链（至少 3 个 upvote）
- 抓取网页内容，清洗后得到约 **800 万个文档**
- 总计约 **40GB 文本**（约 80 亿 token）
- 去除了 Wikipedia（因为很多基准测试用了维基数据）

### 3.2 为什么不用 Common Crawl？

论文解释了为什么不用现成的 Common Crawl：
- 数据质量太差，很多文档"基本不可读"
- WebText 通过 Reddit 社区投票做了天然的质量筛选

---

## 4 🧱 模型架构

GPT-2 的架构基本和 GPT-1 一样（Transformer 解码器），主要改动：

| 改动 | GPT-1 | GPT-2 |
|------|-------|-------|
| Layer Norm 位置 | 残差之后 | **残差之前**（Pre-LN） |
| 初始化 | 常规 | 残差层按 $1/\sqrt{N}$ 缩放 |
| 上下文长度 | 512 | **1024** |
| 词表大小 | BPE 37,000 | **BPE 50,257** |

### 4.1 模型规模

| 模型 | 层数 | 隐藏维度 | 注意力头 | 参数量 |
|------|------|---------|---------|--------|
| Small | 12 | 768 | 12 | 117M |
| Medium | 24 | 1024 | 16 | 345M |
| Large | 36 | 1280 | 20 | 762M |
| **XL** | **48** | **1600** | **25** | **1542M** |

> 💡 **GPT-2 XL 的 1.5B 参数**在当时是最大的语言模型。和 BERT_LARGE（340M）对比一下——大了 4.5 倍。

### 4.2 代码验证：GPT-2 模型结构

```python
from transformers import GPT2Model
import torch

model = GPT2Model.from_pretrained('gpt2')
total_params = sum(p.numel() for p in model.parameters())
print(f"GPT-2 参数量: {total_params / 1e6:.1f}M")

# 查看模型结构
print(f"\n层数: {model.config.n_layer}")
print(f"隐藏维度: {model.config.n_embd}")
print(f"注意力头: {model.config.n_head}")
print(f"上下文长度: {model.config.n_ctx}")
print(f"词表大小: {model.config.vocab_size}")
```

```
GPT-2 参数量: 124.4M

层数: 12
隐藏维度: 768
注意力头: 12
上下文长度: 1024
词表大小: 50257
```

---

## 5 📈 核心实验结果

### 5.1 语言建模（Table 2）

GPT-2 在 **7/8** 个语言建模基准上取得 zero-shot SOTA：

| 数据集 | 之前 SOTA | GPT-2 (zero-shot) |
|--------|----------|-------------------|
| WikiText-103 | 37.5 PPL | **37.5 PPL** |
| LAMBADA | 99.8 PPL | **8.6 PPL** |
| Penn Tree Bank | 62.4 PPL | **35.7 PPL** |
| ... | ... | ... |

> 💡 **LAMBADA 的突破特别惊人**：从 99.8 PPL 降到 8.6 PPL。LAMBADA 测试的是预测段落的最后一个词，需要长距离上下文理解。

### 5.2 阅读理解：CoQA

| 模型 | F1 | 训练数据 |
|------|-----|---------|
| Baseline 1 | 55.0 | 127K+ |
| Baseline 2 | 55.3 | 127K+ |
| Baseline 3 | 62.1 | 127K+ |
| Baseline 4 | 65.1 | 127K+ |
| **GPT-2 (zero-shot)** | **55.0** | **0** |

GPT-2 不用任何训练数据就匹配了 3/4 个 baseline！

### 5.3 规模效应：性能随模型大小对数线性增长

![Figure 1](./images/123b2e3711e5e6107c8589d946c274291aeb2e46dbc6aa5001a21170070bbf69.jpg)

**关键发现**：
- 性能与模型参数量呈**对数线性关系**
- 没有饱和迹象——模型越大越好
- 这为后来的 scaling laws（Chinchilla 论文）埋下了伏笔

### 5.4 文本生成

GPT-2 能生成长而连贯的文本段落（当时最令人震惊的演示之一）。

---

## 6 🔑 关键贡献与影响

### 6.1 核心贡献

1. **Zero-shot 学习**：首次系统证明了语言模型可以不做任何微调就完成多种任务
2. **Scaling 思想**：更大模型 + 更多数据 → 更好性能（对数线性）
3. **数据的重要性**：WebText 的构建方式表明**数据质量**至关重要
4. **Prompt 范式**：用自然语言描述任务的方式成为后来 prompt engineering 的基础

### 6.2 对后续工作的影响

- **GPT-3** 直接继承了 GPT-2 的路线，把模型扩大 100 倍（175B），从 zero-shot 扩展到 few-shot
- **Prompt engineering** 成为新范式
- **Scaling laws** 成为研究方向
- "更大 = 更好"的信念推动了整个大模型浪潮

### 6.3 局限性

- Zero-shot 性能仍然不如微调方法
- 翻译等任务效果较弱（BLEU 5 vs 监督方法 30+）
- 模型仍然 underfit WebText
- 没有证明 fine-tuning 后的效果

---

## 7 🔗 与其他知识的关联

### 7.1 与 BERT 的路线分歧

BERT 和 GPT-2 代表了两条路线的起点：
- **BERT 路线**：编码器 + 微调 → 适用于理解任务
- **GPT-2 路线**：解码器 + prompt → 适用于生成任务

后来，GPT 路线胜出了——因为**生成能力隐含理解能力**。能正确续写答案的模型，必然理解了问题。

### 7.2 与 llm-math-foundations 的关联

- GPT-2 的训练目标就是**最大似然估计**（llm-math-foundations Ch02）
- 自回归分解 $p(x) = \prod p(s_n|s_1,...,s_{n-1})$ 就是**链式法则**
- 规模效应的对数线性关系是 **scaling laws** 的早期实证

---

## 8 ❓ 思考题

1. **为什么 GPT-2 用解码器（单向注意力）而不是编码器（双向）？生成任务和单向有什么必然联系？**

2. **GPT-2 的 zero-shot 和后来 GPT-3 的 few-shot 有什么本质区别？为什么 few-shot 效果好这么多？**

3. **WebText 通过 Reddit upvote 筛选数据质量，这种做法有什么潜在的偏差？**

4. **论文中说"监督学习的目标是无监督目标的子集"，这个论点成立吗？有什么隐含假设？**

5. **GPT-2 的 Pre-LN（Layer Norm 在残差之前）为什么比 Post-LN 更稳定？**

---

## 9 📚 延伸阅读

1. **GPT-3** (Brown et al., 2020) — 本系列下一篇，175B 参数，few-shot learning
2. **GPT-1** (Radford et al., 2018) — 预训练+微调范式，GPT 系列的起点
3. **Scaling Laws** (Kaplan et al., 2020) — 系统研究模型大小、数据量、计算量的关系
4. **Chinchilla** (Hoffmann et al., 2022) — 本系列第 5 篇，计算最优的 scaling
