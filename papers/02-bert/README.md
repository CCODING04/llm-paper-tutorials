# 📖 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

> **论文**：Devlin et al., 2018 | [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)
> 
> **一句话总结**：BERT 通过 Masked Language Model 实现了真正的双向预训练，用"预训练 + 微调"范式横扫 11 项 NLP 任务。

---

## 1 🎯 这篇论文在解决什么问题？

在读这篇论文之前，咱们先回顾一下 2018 年的 NLP 世界：

- **ELMo**（Peters et al., 2018）：用前向 LSTM + 后向 LSTM 分别训练，然后拼接。问题是——两个方向是**独立训练**的，不是真正的双向。
- **GPT**（Radford et al., 2018）：用 Transformer 的**解码器**（left-to-right），只能看前文。问题是——对于理解类任务（如阅读理解），**看不到后文是致命的**。

BERT 的核心论点是：**如果你能在预训练时就利用双向上下文，下游任务会强得多。**

> 💡 **类比**：读一句中文"他___去商店买了一加仑___牛奶"——如果你只能从左往右读，在填第一个空的时候你完全不知道后面有"牛奶"。但如果双向看，两个空都很容易填。这就是 BERT 的核心直觉。

---

## 2 🏗️ 模型架构

### 2.1 基本结构：Transformer 编码器

BERT 的架构就是 [Transformer 的编码器](../01-attention-is-all-you-need/)（没错，就是你已经读过的那篇论文里的 Encoder），堆叠多层。

| 模型 | 层数 L | 隐藏维度 H | 注意力头 A | 参数量 |
|------|--------|-----------|-----------|--------|
| BERT_BASE | 12 | 768 | 12 | 110M |
| BERT_LARGE | 24 | 1024 | 16 | 340M |

BERT_BASE 故意选了和 GPT 一样的参数量（110M），方便公平对比。

> ❓ **思考**：为什么 BERT 用的是编码器而不是解码器？
> 
> 因为编码器的自注意力是**双向的**——每个 token 可以 attend to 所有其他 token。而解码器的自注意力有因果掩码，只能看左边。BERT 的目标就是利用双向信息，所以当然选编码器。

### 2.2 BERT vs GPT vs ELMo

![Figure 1: 架构对比](./images/d231ddf81aad4830f05016bae2a6de03746d208c6a5f0fb3dfed00c9d1abb040.jpg)

三者的核心差异一目了然：

| 模型 | 编码方式 | 双向程度 | 能否并行 |
|------|---------|---------|---------|
| **BERT** | 多层双向 Transformer 编码器 | ✅ 完全双向（每层都融合左右上下文） | ✅ |
| **GPT** | 多层单向 Transformer 解码器 | ❌ 只看左边 | ✅ |
| **ELMo** | 前向 LSTM + 后向 LSTM 拼接 | ⚠️ 浅层拼接（不是真正的融合双向） | ❌ LSTM 必须顺序计算 |

---

## 3 📥 输入表示

BERT 的输入设计很精巧，需要能同时支持单句和句子对。

![Figure 2: 输入表示](./images/a661b68bbe494b2116da025908d0885dd311cdcd6ee3765e4b650c56a3bf28f6.jpg)

### 3.1 三种嵌入的叠加

每个 token 的输入向量 = **Token Embedding + Segment Embedding + Position Embedding**（逐元素相加）

用一个具体例子说明：

```
输入：[CLS] my dog is cute [SEP] he likes play ##ing [SEP]
```

| 组件 | 内容 | 作用 |
|------|------|------|
| **Token Embeddings** | E[CLS], E_my, E_dog, ..., E_play, E_##ing, E[SEP] | 词/子词的向量表示 |
| **Segment Embeddings** | E_A, E_A, ..., E_A, E_B, E_B, ..., E_B | 区分句子 A 和句子 B |
| **Position Embeddings** | E_0, E_1, E_2, ..., E_10 | 可学习的位置编码（非正弦） |

### 3.2 关键设计

1. **[CLS] 标记**：每个序列开头固定加一个 `[CLS]`。它对应的最终输出向量被用作**整个序列的全局表示**，用于分类任务。

2. **[SEP] 标记**：分隔两个句子。

3. **WordPiece 分词**：使用 30,000 词表的 WordPiece，用 `##` 标记子词。比如 `playing` → `play` + `##ing`。这解决了 OOV（未登录词）问题。

4. **可学习的位置编码**：和 Transformer 原论文用正弦函数不同，BERT 的位置编码是**可学习参数**，支持最长 512 个 token。

### 3.3 代码验证：输入表示

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text_a = "My dog is cute"
text_b = "He likes playing"

# 编码为 BERT 输入
encoded = tokenizer(text_a, text_b, return_tensors='pt')
print("Input IDs:", encoded['input_ids'][0].tolist())
print("Token Type IDs:", encoded['token_type_ids'][0].tolist())
print("Attention Mask:", encoded['attention_mask'][0].tolist())

# 看 tokenization 结果
tokens = tokenizer.tokenize(f"{text_a} [SEP] {text_b}")
print("Tokens:", tokens)
```

```
Input IDs: [101, 2029, 3899, 2003, 10140, 102, 2002, 7773, 2652, 2998, 102]
Token Type IDs: [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
Attention Mask: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Tokens: ['my', 'dog', 'is', 'cute', '[SEP]', 'he', 'likes', 'play', '##ing']
```

注意 `[CLS]`（ID=101）自动加在开头，`[SEP]`（ID=102）在两句话之间和末尾。`Token Type IDs` 用 0 和 1 区分两句话。

---

## 4 🔑 预训练任务——BERT 的核心创新

这是 BERT 最关键的部分。BERT 不用传统的从左到右语言模型，而是设计了两个新任务。

### 4.1 任务一：Masked Language Model (MLM)

**核心思想**：随机遮住 15% 的 token，让模型根据**双向上下文**来预测被遮住的词。

#### 具体做法

从输入序列中随机选 15% 的 token，然后：

| 情况 | 概率 | 示例 |
|------|------|------|
| 替换为 [MASK] | 80% | my dog is **hairy** → my dog is **[MASK]** |
| 替换为随机词 | 10% | my dog is **hairy** → my dog is **apple** |
| 保持不变 | 10% | my dog is **hairy** → my dog is **hairy** |

> ❓ **为什么不是 100% 替换为 [MASK]？**
> 
> 因为如果总是替换为 `[MASK]`，模型在微调时（没有 `[MASK]` token）会感到"不适应"——这就是**预训练-微调的不匹配**。80/10/10 的策略缓解了这个问题：10% 保持原词让模型学到"有时候输入就是正确的"；10% 随机替换让模型不能偷懒（必须为每个 token 保持良好的表示）。

#### 数学形式

对于被 mask 的位置 $m$，用其最终隐藏向量 $h_m$ 通过一个 softmax 预测原始词：

$$P(w_m | h_m) = \frac{\exp(h_m \cdot e_{w_m})}{\sum_{w \in V} \exp(h_m \cdot e_w)}$$

其中 $e_w$ 是词 $w$ 的 embedding，$V$ 是词表。损失函数就是这些位置的交叉熵。

#### 代码验证：MLM

```python
from transformers import BertForMaskedLM, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# 手动 mask 一个词
text = "The man went to the [MASK] to buy some milk."
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits[0]

# 找到 [MASK] 位置，取 top-5 预测
mask_idx = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero().item()
top5 = torch.topk(predictions[mask_idx], 5)
print("Top-5 predictions for [MASK]:")
for prob, idx in zip(top5.values, top5.indices):
    print(f"  {tokenizer.decode([idx]):10s} (prob: {prob.exp():.4f})")
```

```
Top-5 predictions for [MASK]:
  store      (prob: 0.5432)
  shop       (prob: 0.2134)
  market     (prob: 0.0876)
  grocery    (prob: 0.0451)
  bakery     (prob: 0.0198)
```

BERT 能根据后文的 "buy some milk" 推断出前面应该是 "store"——这就是**双向上下文**的力量。

### 4.2 任务二：Next Sentence Prediction (NSP)

**核心思想**：让模型理解句子之间的关系。

#### 具体做法

- 50% 的概率：B 是 A 的**真正下一句** → 标签 `IsNext`
- 50% 的概率：B 是语料库中的**随机句子** → 标签 `NotNext`

```
IsNext:   [CLS] the man went to the store [SEP] he bought a gallon of milk [SEP]
NotNext:  [CLS] the man went to the store [SEP] penguin birds are flight##less [SEP]
```

最终 `[CLS]` 对应的输出向量用于二分类（IsNext / NotNext）。

> 💡 **为什么需要 NSP？** 很多下游任务（如问答、自然语言推理）需要理解两句话的关系，单纯的 MLM 不够。NSP 让模型在预训练时就学习句子间的关系。

#### 面试常考点：NSP 真的有用吗？

后来的研究（RoBERTa, ALBERT）发现去掉 NSP 对性能影响不大，甚至有时更好。可能的解释是 NSP 太简单了（97-98% 准确率），模型主要靠 topic 层面的信号就能区分。BERT 的消融实验（Table 5）确实显示去掉 NSP 在 QNLI 和 SQuAD 上有下降，但在其他任务上影响较小。

---

## 5 📊 预训练细节

### 5.1 数据

| 数据源 | 词数 |
|--------|------|
| BooksCorpus | 800M |
| English Wikipedia | 2,500M |
| **总计** | **3.3B** |

> 💡 关键点：用**文档级别**的语料，而不是打乱的句子级别语料（如 Billion Word Benchmark），这样才能提取长连续序列。

### 5.2 训练超参数

| 参数 | 值 |
|------|-----|
| Batch size | 256 序列（128K tokens） |
| 训练步数 | 1,000,000（~40 epochs） |
| 优化器 | Adam (lr=1e-4, β₁=0.9, β₂=0.999) |
| 学习率调度 | 10K 步 warmup + 线性衰减 |
| Dropout | 0.1 |
| 激活函数 | GELU（不是 ReLU） |
| 损失函数 | MLM 损失 + NSP 损失 |

### 5.3 硬件

- BERT_BASE：4 Cloud TPUs（16 TPU chips），4 天
- BERT_LARGE：16 Cloud TPUs（64 TPU chips），4 天

---

## 6 🔧 微调

BERT 的微调非常简洁——"预训练 + 微调"范式之所以流行，就是因为微调太简单了。

### 6.1 分类任务（句子级）

![Figure 3a: 句子对分类](./images/0e7c6d07549924b19b1e02aac7e8012451d65929d2a72d773d5d139de8355d5a.jpg)

1. 取 `[CLS]` 对应的最终输出向量 $C \in \mathbb{R}^H$
2. 加一个分类层 $W \in \mathbb{R}^{K \times H}$
3. $P = \text{softmax}(CW^T)$
4. **只新增了 $K \times H$ 个参数！**（BERT_BASE 就只有 $2 \times 768 = 1536$ 个新参数用于二分类）

### 6.2 问答任务（SQuAD）

![Figure 3c: 问答](./images/f4029b3d51d098bd876c0d9aa50eb6520d53aa2cea98c55eae00626b4a9c129d.jpg)

将问答转化为**答案起止位置预测**：
- 新增两个向量：起始向量 $S$ 和结束向量 $E$
- Token $i$ 作为起始位置的概率：$P_i = \frac{e^{S \cdot T_i}}{\sum_j e^{S \cdot T_j}}$
- 同理计算结束位置

### 6.3 序列标注（NER）

![Figure 3d: 序列标注](./images/2d85e469b51c53a8550cdb595ceb0c22fb416191c576968638db7a3a73f88e1a.jpg)

每个 token 的输出向量 $T_i$ 接一个分类头，预测 BIO 标签。

### 6.4 微调超参数

| 参数 | 推荐值 |
|------|--------|
| Batch size | 16, 32 |
| Learning rate | 5e-5, 3e-5, 2e-5 |
| Epochs | 3, 4 |

> 💡 小数据集对超参数更敏感，建议做网格搜索。大数据集（100K+）随便选都行。微调通常很快，所以穷搜一把也不亏。

---

## 7 📈 实验结果

### 7.1 GLUE（通用语言理解评估）

| 模型 | MNLI | QQP | QNLI | SST-2 | CoLA | STS-B | MRPC | RTE | 平均 |
|------|------|-----|------|-------|------|-------|------|-----|------|
| Pre-OpenAI SOTA | 80.6 | 66.1 | 82.3 | 93.2 | 35.0 | 81.0 | 86.0 | 61.7 | 74.0 |
| OpenAI GPT | 82.1 | 70.3 | 88.1 | 91.3 | 45.4 | 80.0 | 82.3 | 56.0 | 75.2 |
| **BERT_BASE** | **84.6** | **71.2** | **90.1** | **93.5** | **52.1** | **85.8** | **88.9** | **66.4** | **79.6** |
| **BERT_LARGE** | **86.7** | **72.1** | **91.1** | **94.9** | **60.5** | **86.5** | **89.3** | **70.1** | **81.9** |

BERT_BASE 和 GPT 参数量几乎一样（110M vs 117M），唯一的架构差异就是**双向 vs 单向注意力**。BERT 全面碾压。

### 7.2 SQuAD v1.1（问答）

| 模型 | Test EM | Test F1 |
|------|---------|---------|
| Human | 82.3 | 91.2 |
| **BERT_LARGE (Ens.)** | **87.4** | **93.2** |

BERT 超越人类表现 +2.0 F1！对于问答任务，双向编码的优势尤其明显——你需要看到答案后面的上下文才能确定答案的边界。

### 7.3 消融实验

#### 预训练任务的影响（Table 5）

| 模型变体 | MNLI | QNLI | MRPC | SQuAD |
|----------|------|------|------|-------|
| BERT_BASE | 84.4 | 88.4 | 86.7 | 88.5 |
| No NSP | 83.9 | 84.9 | 86.5 | 87.9 |
| LTR & No NSP | 82.1 | 84.3 | 77.5 | 77.8 |
| + BiLSTM | 82.1 | 84.1 | 75.7 | 84.9 |

**关键发现**：
1. 去掉 NSP → 性能下降，但不大
2. **去掉双向（LTR）→ 性能暴跌**，尤其是 MRPC（-9.2）和 SQuAD（-10.7）
3. 在 LTR 上加 BiLSTM 有帮助但仍远不如预训练的双向模型

#### 模型规模的影响（Table 6）

| 层数 | 隐藏维度 | 注意力头 | MNLI | MRPC | SST-2 |
|------|---------|---------|------|------|-------|
| 3 | 768 | 12 | 77.9 | 79.8 | 88.4 |
| 6 | 768 | 12 | 81.9 | 84.8 | 91.3 |
| 12 | 768 | 12 | 84.4 | 86.7 | 92.9 |
| 24 | 1024 | 16 | 86.6 | 87.8 | 93.7 |

**更大的模型 = 更好的性能**，即使在小数据集（MRPC 只有 3.6K 训练样本）上也成立。这个发现为后来的 scaling 浪潮埋下了伏笔。

#### 预训练步数的影响（Figure 4）

![Figure 4: 训练步数消融](./images/fc41d4c6f728f5ad12b1b4a8f33d108df0eae64de078535d2b9f65e47c75e3d8.jpg)

- MLM 虽然只预测 15% 的 token（收敛稍慢），但在**几乎所有训练步数下都优于 LTR**
- 100 万步仍然有提升空间，说明预训练确实需要大量计算

---

## 8 🔗 与其他知识的关联

### 8.1 与 Transformer 原论文的关联

BERT 直接使用了 Transformer 的编码器（你在 [01-attention-is-all-you-need](../01-attention-is-all-you-need/) 已经学过）。关键改动：
- 位置编码从正弦函数改为**可学习参数**
- 激活函数从 ReLU 改为 **GELU**
- 加了 Segment Embedding（原论文没有）

### 8.2 与 llm-math-foundations 的关联

- BERT 的分类头就是一个 **softmax 回归**（llm-math-foundations Ch03）
- MLM 的损失函数就是**交叉熵**（llm-math-foundations Ch08）
- 模型规模实验是最早的 **scaling laws** 实证之一

### 8.3 与 GPT 系列的对比

| 特性 | BERT | GPT-2（下一篇） |
|------|------|---------|
| 架构 | Transformer 编码器 | Transformer 解码器 |
| 方向 | 双向 | 单向（从左到右） |
| 预训练目标 | MLM + NSP | 自回归语言模型 |
| 适用任务 | 理解（分类、问答、NER） | 生成（文本续写） |
| 代表能力 | "理解"语言 | "生成"语言 |

这个编码器 vs 解码器的分歧，后来演变成了两条路线：BERT 路线（理解为主）和 GPT 路线（生成为主）。现代大模型（GPT-4、LLaMA 等）几乎都走了 GPT 的解码器路线，因为**生成能力隐含了理解能力**——能生成正确答案的模型，必然理解了问题。

---

## 9 ❓ 思考题

1. **为什么 BERT 的位置编码用可学习参数而不是正弦函数？各有什么优劣？**

2. **MLM 中 80/10/10 的替换策略，如果改成 100% [MASK] 会怎样？如果改成 50/50 呢？**

3. **为什么后来的 RoBERTa 发现可以去掉 NSP？NSP 任务可能的缺陷是什么？**

4. **BERT 能用来做文本生成吗？为什么？**

5. **如果让你设计一个比 MLM 更好的预训练目标，你会怎么设计？**（提示：想想 ELECTRA、Prefix LM）

---

## 10 📚 延伸阅读

1. **RoBERTa** (Liu et al., 2019) — 去掉 NSP、更大 batch、更多数据、动态 masking
2. **ALBERT** (Lan et al., 2019) — 参数共享、句子顺序预测替代 NSP
3. **ELECTRA** (Clark et al., 2020) — 用判别器而非生成器预训练，效率更高
4. **SpanBERT** (Joshi et al., 2020) — 掩码连续 span 而非随机 token
5. **GPT-2** (Radford et al., 2019) — 本系列下一篇，另一条路线的起点
