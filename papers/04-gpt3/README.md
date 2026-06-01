# 📖 GPT-3: Language Models are Few-Shot Learners

> **论文**：Brown et al., 2020 | NeurIPS 2020
> 
> **一句话总结**：175B 参数的语言模型通过 few-shot in-context learning，不需要任何梯度更新就能在多种 NLP 任务上接近甚至超越微调方法。

---

## 1 🎯 这篇论文在解决什么问题？

GPT-2 证明了 zero-shot 的可能性，但性能距离监督方法还很远。GPT-3 的核心问题是：

> **如果模型足够大（175B 参数），能否仅通过给几个示例（few-shot），不更新任何参数，就达到甚至超越微调方法？**

### 1.1 三种评估方式

| 方式 | 输入 | 参数更新 |
|------|------|---------|
| **Zero-shot** | 任务描述 | ❌ |
| **One-shot** | 任务描述 + 1 个示例 | ❌ |
| **Few-shot** | 任务描述 + 多个示例 | ❌ |

> 💡 **关键**：所有方式都**不更新模型参数**。示例只是拼在输入文本里，模型通过条件概率直接生成答案。这就是 **In-Context Learning**。

### 1.2 和 BERT/GPT-2 的范式对比

| 维度 | BERT | GPT-2 | **GPT-3** |
|------|------|-------|-----------|
| 使用方式 | 微调 | Zero-shot prompt | **Few-shot prompt** |
| 参数更新 | ✅ 全量微调 | ❌ | ❌ |
| 需要标注数据 | ✅ | ❌ | ❌（示例只在输入里） |
| 参数量 | 340M | 1.5B | **175B** |

---

## 2 🧱 模型与训练

### 2.1 模型架构

GPT-3 架构和 GPT-2 基本一样（Transformer 解码器），主要改动：

| 改动 | 说明 |
|------|------|
| 交替注意力密度 | Dense attention 和稀疏 attention 交替（每 4 层有一次稀疏） |
| 上下文长度 | 1024 → **2048** |
| 词表 | 和 GPT-2 一样的 Byte-level BPE，50,257 |
| Pre-LN | 和 GPT-2 一样 |

### 2.2 八种规模

| 模型 | 参数量 | 层数 | 隐藏维度 | 注意力头 | 上下文 |
|------|--------|------|---------|---------|--------|
| Small | 125M | 12 | 768 | 12 | 2048 |
| Medium | 350M | 24 | 1024 | 16 | 2048 |
| Large | 760M | 24 | 1536 | 16 | 2048 |
| XL | 1.3B | 24 | 2048 | 24 | 2048 |
| ... | ... | ... | ... | ... | ... |
| **175B** | **175B** | **96** | **12288** | **96** | **2048** |

> 💡 GPT-3 175B 用了 96 层，12288 维隐藏层。训练用了 **3.14 × 10²³ FLOPS**（约 355 V100 GPU-years）。

### 2.3 训练数据

| 数据集 | 比例 | Token 数 |
|--------|------|---------|
| Filtered Common Crawl | 60% | 410B |
| WebText2 | 20% | 19B |
| Books1 | 6% | 12B |
| Books2 | 3% | 55B |
| Wikipedia | 3% | 10B |

总计约 **500B tokens**（~570GB 文本）。Common Crawl 做了质量过滤（类似 WebText 的思路）。

### 2.4 训练细节

- 训练 batch size: 3.2M tokens
- 学习率: 0.6e-4（175B）
- 训练 300B tokens（部分数据训练了不到 1 个 epoch）
- 集群: V100 GPU，带宽约 800 Gbps

---

## 3 📈 核心实验结果

### 3.1 语言建模

GPT-3 在 Penn Tree Bank 和 LAMBADA 上：
- PTB: **20.50 PPL**（SOTA）
- LAMBADA: **1.92 PPL**（SOTA，远超之前最好结果）

### 3.2 SuperGLUE（Table 3.1）

| 任务 | Fine-tuned BERT-Large | Fine-tuned RoBERTa | **GPT-3 (few-shot)** |
|------|----------------------|--------------------|---------------------|
| BoolQ | 76.4 | 83.4 | **81.0** |
| CB | 87.3 | 95.0 | **87.5** |
| Copa | 74.0 | 92.4 | **91.6** |
| MultiRC | 74.2 | 84.0 | **78.9** |
| RTE | 73.1 | 86.6 | **69.0** |
| WiC | 69.2 | 72.0 | **49.4** |

> 💡 GPT-3 在 SuperGLUE 上**接近但不如**微调后的 RoBERTa。但关键是 GPT-3 **没有更新参数**。

### 3.3 问答（Table 3.4）

| 数据集 | GPT-3 (few-shot) | SOTA (fine-tuned) |
|--------|-----------------|-------------------|
| TriviaQA | **71.2** | 68.0 |
| Natural Questions | **29.9** | 36.6 |
| WebQuestions | **22.0** | 20.1 |

GPT-3 在 TriviaQA 上**超越了**微调 SOTA！

### 3.4 翻译

| 方向 | GPT-3 (few-shot) | 之前 SOTA (unsupervised) |
|------|-----------------|-------------------------|
| En→Fr | **25.2 BLEU** | 33.5 BLEU |
| Fr→En | **28.6 BLEU** | 33.5 BLEU |

仍然不如监督方法，但比 GPT-2 的 zero-shot（5 BLEU）进步巨大。

### 3.5 规模效应（核心发现）

- **几乎每个任务都随模型规模平滑提升**
- Few-shot 的提升幅度远大于 zero-shot
- 某些任务在特定规模出现"涌现"（如算术运算在大模型上突然变好）

### 3.6 代码验证：Few-shot Prompting

```python
from transformers import GPT2Tokenizer
import json

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Few-shot prompt 示例（翻译任务）
prompt = """Translate English to French:

English: Hello, how are you?
French: Bonjour, comment allez-vous?

English: The cat is on the table.
French: Le chat est sur la table.

English: I love machine learning.
French:"""

# 计算这个 prompt 的 token 数
tokens = tokenizer.encode(prompt)
print(f"Prompt tokens: {len(tokens)}")
print(f"Prompt preview:\n{prompt[:200]}...")
```

```
Prompt tokens: 67
Prompt preview:
Translate English to French:

English: Hello, how are you?
French: Bonjour, comment allez-vous?

English: The cat is on the table.
French: Le chat est sur la table.

English: I love machine learning.
French:...
```

---

## 4 🔑 核心贡献与局限性

### 4.1 贡献

1. **In-Context Learning**：证明了模型可以通过输入中的示例学习新任务，不需要参数更新
2. **Scaling 的力量**：175B 参数在 few-shot 下可以接近甚至超越微调方法
3. **涌现能力**：某些能力（如算术、新词使用）在大模型上突然出现
4. **统一接口**：所有任务通过同一种方式（文本到文本）解决

### 4.2 局限性（论文自己承认的）

1. **文本生成仍有弱点**：长文本一致性、重复问题
2. **一些任务差距大**：如阅读理解（SQuAD 2.0）和 WiC
3. **偏见和公平性**：训练数据包含互联网偏见
4. **训练成本**：175B 模型训练极其昂贵
5. **数据污染**：训练数据和测试集可能重叠

---

## 5 🔗 关联与延伸

### 5.1 与 BERT/GPT-2 的演进

```
BERT (2018) → 预训练+微调 → 理解能力强
GPT-2 (2019) → 预训练+zero-shot → 生成能力强但效果弱
GPT-3 (2020) → 预训练+few-shot → 生成强+效果接近微调
```

后来的路线：所有现代大模型（ChatGPT、LLaMA、Claude）都走 GPT 路线（解码器），但在 GPT-3 的 few-shot 基础上加上了 **RLHF 对齐**（InstructGPT → ChatGPT）。

### 5.2 与 llm-math-foundations 的关联

- Scaling 的 power-law 关系对应 llm-math-foundations 中的 **幂律分布**
- In-context learning 的机制至今仍是研究热点

---

## 6 ❓ 思考题

1. **In-context learning 和 fine-tuning 的本质区别是什么？为什么不给模型更新参数也能学？**

2. **GPT-3 的 few-shot 示例为什么能起作用？信息是如何从示例"流入"预测的？**

3. **如果 GPT-3 用了 300B tokens 训练但数据集有 500B tokens，这意味着什么？**

4. **论文中哪些任务 GPT-3 表现特别差？为什么？**

5. **数据污染问题在大模型时代如何解决？**

---

## 7 📚 延伸阅读

1. **InstructGPT** (Ouyang et al., 2022) — 本系列第 6 篇，GPT-3 + RLHF
2. **Scaling Laws** (Kaplan et al., 2020) — OpenAI 的 scaling laws 研究
3. **Chinchilla** (Hoffmann et al., 2022) — 本系列第 5 篇，质疑 GPT-3 的数据量不够
4. **PaLM** (Chowdhery et al., 2022) — Google 的 540B 模型，进一步验证 scaling
