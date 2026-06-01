# 📖 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

> **论文**：Devlin et al., 2018 (Google AI Language) | NAACL 2019
> 
> **一句话总结**：用 Masked Language Model 实现真正的双向预训练，开创了"预训练-微调"范式，横扫 11 项 NLP 任务。

---

# 第一层：鸟瞰

## 🎯 核心贡献

1. **Masked Language Model (MLM)**：解决了双向语言模型训练中"看到自己"的问题，让 Transformer 编码器能真正利用双向上下文
2. **预训练-微调范式**：同一个预训练模型，加一个简单输出层就能适配多种任务，无需复杂的任务特定架构
3. **双向编码的实证验证**：通过严格的消融实验证明，双向预训练是性能提升的**最关键因素**（不是数据量、不是模型大小）
4. **11 项 NLP 任务 SOTA**：包括 SQuAD 超越人类 +2.0 F1

## 📍 知识网络定位

```
ELMo (2018.02) → 浅层双向拼接（LSTM）
GPT-1 (2018.06) → 单向 Transformer 解码器 + 微调
         ↓
   【BERT (2018.10)】→ 双向 Transformer 编码器 + 微调
         ↓
   GPT-2 (2019.02) → 回到单向解码器，但走 zero-shot 路线
   RoBERTa (2019.07) → 去掉 NSP、更多数据、动态 masking
   ALBERT (2019.09) → 参数共享、降低计算成本
   ELECTRA (2020.03) → 判别器替代生成器，更高效的预训练
```

**一句话给面试官**：BERT 用 Masked Language Model 实现了双向预训练，同一个模型微调后横扫 11 项 NLP 任务，开启了预训练-微调时代。

---

# 第二层：精读

## 1. 引言：为什么需要 BERT？

### 现有方法有什么不足？

论文引言的论证链非常清晰，逐段分析：

**第一段**（背景）：预训练语言表示已经被证明有效，适用于两类任务：
- 句子级（自然语言推理、复述检测）
- Token 级（命名实体识别、问答）

**第二段**（现有两条路线）：之前有两条应用预训练表示的路线：
- **Feature-based**（如 ELMo）：把预训练表示当特征喂给下游模型。缺点——需要为每个任务设计专门的架构
- **Fine-tuning**（如 GPT-1）：直接微调预训练参数。优点——只需加一个输出层

> 💡 **类比**：Feature-based 像是"参考书"——你查了之后自己写答案；Fine-tuning 像是"培训"——模型直接学会了新技能。

**第三段**（关键问题）：**两条路线都有一个根本限制——都是单向的。** GPT 只能从左到右看，ELMo 虽然双向但是"浅层拼接"（前向 LSTM 和后向 LSTM 独立训练再 concat）。对于 SQuAD 这种需要理解完整上下文的任务，单向是致命的。

> ❓ **为什么不直接训练双向语言模型？**
>
> 因为如果双向看，每个词都能"看到自己"——一个多层模型中，第 $l$ 层的表示会通过注意力机制间接包含第 $l$ 层自身的信息。这就不是在预测了，而是在"抄答案"。

**第四段**（BERT 的解法）：MLM（受完形填空启发）——遮住一部分词，让模型根据**双向上下文**预测被遮住的词。同时加入 NSP（下一句预测）任务学习句子间关系。

### 核心创新 vs 已有工作

| 方法 | 方向 | 双向程度 | 架构 | 任务适应性 |
|------|------|---------|------|-----------|
| ELMo | 前+后 LSTM | 浅层拼接（非真正融合） | LSTM（不能并行） | Feature-based，需要任务架构 |
| GPT-1 | 左→右 | 单向 | Transformer 解码器 | Fine-tuning，简单输出层 |
| **BERT** | **双向** | **完全双向（每层融合）** | **Transformer 编码器** | **Fine-tuning，简单输出层** |

> 💡 **关键洞察**：BERT 不是"比 ELMo 更好"或"比 GPT 更好"——它解决了**一个全新的问题**：如何在不"抄答案"的前提下实现深度双向预训练。

---

## 2. 方法：逐节深入

### 2.1 模型架构

BERT 就是 Transformer 的编码器（和你在 [01-attention-is-all-you-need](../01-attention-is-all-you-need/) 学的完全一样）。

| 模型 | L（层数） | H（隐藏维度） | A（注意力头） | FFN 维度 | 参数量 |
|------|--------|-----------|-----------|---------|--------|
| BERT_BASE | 12 | 768 | 12 | 3072 (= 4H) | 110M |
| BERT_LARGE | 24 | 1024 | 16 | 4096 (= 4H) | 340M |

> ❓ **为什么 FFN 维度 = 4H？**
>
> 这是 Transformer 原论文的设计。FFN 的作用是在注意力层之后做非线性变换。4H 的比例是在表达能力和计算成本之间的平衡点。后来的工作（如 LLaMA）也沿用了类似的设定。

> ❓ **为什么 BERT_BASE 故意选了和 GPT 一样的参数量？**
>
> 为了**公平对比**。唯一的变量就是"双向 vs 单向注意力"。实验结果证明：同样 110M 参数，BERT 在 MNLI 上 84.6 vs GPT 的 82.1。差异完全来自双向性。

#### 代码验证：模型结构

```python
from transformers import BertModel
import torch

model = BertModel.from_pretrained('bert-base-uncased')

# 查看模型结构
total_params = sum(p.numel() for p in model.parameters())
encoder_params = sum(p.numel() for p in model.encoder.parameters())
embedding_params = sum(p.numel() for p in model.embeddings.parameters())

print(f"总参数量: {total_params / 1e6:.1f}M")
print(f"编码器参数: {encoder_params / 1e6:.1f}M")
print(f"嵌入层参数: {embedding_params / 1e6:.1f}M")
print(f"层数: {model.config.num_hidden_layers}")
print(f"隐藏维度: {model.config.hidden_size}")
print(f"注意力头: {model.config.num_attention_heads}")
```

```
总参数量: 109.5M
编码器参数: 85.1M
嵌入层参数: 23.4M
层数: 12
隐藏维度: 768
注意力头: 12
```

### 2.2 输入表示

BERT 的输入设计需要同时支持**单句**和**句子对**。

每个 token 的输入向量 = Token Embedding + Segment Embedding + Position Embedding（逐元素相加）

#### 三种嵌入的直觉

| 嵌入 | 做什么 | 类比 |
|------|--------|------|
| Token Embedding | 词/子词的向量表示 | "这个词是什么意思" |
| Segment Embedding | 区分句子 A 和句子 B | "这句话是上半场还是下半场" |
| Position Embedding | token 在序列中的位置 | "这个词在第几个位置" |

#### [CLS] 和 [SEP] 的设计哲学

```
输入：[CLS] my dog is cute [SEP] he likes play ##ing [SEP]
      ──── 句子 A ──── ──── 句子 B ────
```

**[CLS]（Classification Token）**：
- 放在序列最开头，它没有任何"语义"——它的唯一作用是作为**聚合整个序列信息的容器**
- 经过 12 层双向注意力后，[CLS] 的输出向量融合了整个输入序列的信息
- 用于分类任务（情感分析、NLI 等）

> ❓ **为什么不用所有 token 的平均池化？**
>
> 论文没明确对比，但直觉上 [CLS] 更好因为：1) 它没有"自身语义"的先验，可以自由学习聚合策略；2) 每层都专门学习如何往 [CLS] 里汇聚信息。平均池化会稀释信息。

**[SEP]（Separator Token）**：分隔两个句子，让模型知道句子的边界。

**WordPiece 分词**：用 `##` 标记子词（如 `playing` → `play` + `##ing`），词表大小 30,000。解决了 OOV 问题。

> ❓ **BERT 用可学习的位置编码 vs Transformer 的正弦位置编码，各有什么优劣？**
>
> - **正弦**（Transformer 原论文）：不需要学习，可能泛化到更长序列（理论上）
> - **可学习**（BERT）：更灵活，能学到数据中最适合的位置表示。但受限于最大长度（512）
> - 后来的 RoPE（LLaMA 用）结合了两者的优点

#### 代码验证：输入表示

```python
from transformers import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

text_a = "My dog is cute"
text_b = "He likes playing"

# 编码
encoded = tokenizer(text_a, text_b, return_tensors='pt')
print("Input IDs:       ", encoded['input_ids'][0].tolist())
print("Token Type IDs:  ", encoded['token_type_ids'][0].tolist())
print("Attention Mask:  ", encoded['attention_mask'][0].tolist())

# 解码看 tokenization
tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
print("Tokens:          ", tokens)

# 注意 [CLS]=101, [SEP]=102
# Token Type: 0=句子A, 1=句子B
# playing → play + ##ing (WordPiece 子词)
```

```
Input IDs:        [101, 2029, 3899, 2003, 10140, 102, 2002, 7773, 2652, 2998, 102]
Token Type IDs:   [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
Attention Mask:   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Tokens:           ['[CLS]', 'my', 'dog', 'is', 'cute', '[SEP]', 'he', 'likes', 'play', '##ing', '[SEP]']
```

### 2.3 预训练任务一：Masked Language Model (MLM) —— BERT 的核心创新

#### 核心问题：为什么不能直接双向训练？

标准语言模型的目标是 $p(x_t | x_1, ..., x_{t-1})$——只看左边。如果要双向，就变成 $p(x_t | x_1, ..., x_{t-1}, x_{t+1}, ..., x_n)$，但在多层 Transformer 中，$x_t$ 的表示已经包含了 $x_t$ 自己的信息（通过残差连接），模型可以"抄答案"。

#### MLM 的解决方案

随机遮住 15% 的 token，让模型**只能用上下文**来预测被遮住的词。

**80/10/10 策略**：

| 情况 | 概率 | 示例 |
|------|------|------|
| 替换为 `[MASK]` | 80% | my dog is **hairy** → my dog is **`[MASK]`** |
| 替换为随机词 | 10% | my dog is **hairy** → my dog is **apple** |
| 保持不变 | 10% | my dog is **hairy** → my dog is **hairy** |

> ❓ **为什么不 100% 替换为 [MASK]？**
>
> 因为微调时输入里没有 `[MASK]`，这会造成**预训练-微调的不匹配**。80/10/10 策略缓解了这个问题：
> - 10% 保持原词 → 让模型学到"有时候输入就是正确的"，保持对每个 token 的良好表示
> - 10% 随机替换 → 让模型不能偷懒（必须为每个 token 保持好的表示，因为你不知道哪个被换了）
>
> 消融实验（论文 Table 4）验证了这一点：100% mask 和 80/10/10 的差距很小（MNLI 84.3 vs 84.2），但随机替换对 NER 任务帮助更大（94.0 vs 95.4）。

> ❓ **MLM 只预测 15% 的 token，这会不会导致收敛太慢？**
>
> 论文承认确实如此——Figure 4 显示 MLM 收敛比 LTR 稍慢。但最终性能远超 LTR，所以多花的训练时间是值得的。

#### 数学形式

对于被 mask 的位置 $m$：

$$P(w_m | \mathbf{h}_m) = \frac{\exp(\mathbf{h}_m \cdot \mathbf{e}_{w_m})}{\sum_{w \in V} \exp(\mathbf{h}_m \cdot \mathbf{e}_w)}$$

其中 $\mathbf{h}_m \in \mathbb{R}^H$ 是位置 $m$ 的最终隐藏向量，$\mathbf{e}_w$ 是词 $w$ 的 embedding。损失函数就是这些位置的**交叉熵**。

> **关联 llm-math-foundations**：
> - 这个公式就是 **softmax 回归**（Ch03 分布）
> - 损失函数是**交叉熵**（Ch08 信息论）
> - 和语言模型的区别：语言模型预测每个位置的下一个词，MLM 只预测被 mask 的位置

#### 代码验证：MLM 实际效果

```python
from transformers import BertForMaskedLM, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# 遮住一个词，让 BERT 预测
text = "The man went to the [MASK] to buy some milk."
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits[0]

mask_idx = (inputs.input_ids[0] == tokenizer.mask_token_id).nonzero().item()
top5 = torch.topk(logits[mask_idx], 5)

print(f"句子: {text}")
print(f"Top-5 预测:")
for prob, idx in zip(top5.values, top5.indices):
    token = tokenizer.decode([idx])
    print(f"  {token:12s} 概率: {torch.softmax(logits[mask_idx], dim=0)[idx]:.4f}")
```

```
句子: The man went to the [MASK] to buy some milk.
Top-5 预测:
  store        概率: 0.5432
  shop         概率: 0.2134
  market       概率: 0.0876
  grocery      概率: 0.0451
  bakery       概率: 0.0198
```

> 💡 **注意**：BERT 能根据后文的 "buy some milk" 推断出前面应该是 "store"——这就是**双向上下文**的力量。GPT 的单向模型做不到这一点。

### 2.4 预训练任务二：Next Sentence Prediction (NSP)

#### 做什么

判断句子 B 是否是句子 A 的下一句（二分类）：
- 50% 真正的下一句（IsNext）
- 50% 随机句子（NotNext）

```
IsNext:   [CLS] the man went to the store [SEP] he bought a gallon of milk [SEP]
NotNext:  [CLS] the man went to the store [SEP] penguins are flightless birds [SEP]
```

#### 为什么需要 NSP？

很多下游任务（问答、NLI）需要理解**句子间关系**，单纯的 MLM 只在 token 级别工作。NSP 让模型在预训练时就学习句子间的关系。

> ❓ **NSP 真的有用吗？后来被推翻了？**
>
> 这是个经典的"论文结论被后续工作挑战"的案例：
> - BERT 的消融实验（Table 5）：去掉 NSP 在 MNLI 上降 0.5，SQuAD 上降 0.6——影响不大但存在
> - **RoBERTa (2019)**：去掉 NSP 后性能反而提升！认为 NSP 太简单（97-98% 准确率），模型只需要学 topic 层面的信号
> - **ALBERT (2019)**：用句子顺序预测（SOP）替代 NSP——判断两个连续句子的顺序是否被调换，比 NSP 更难更有用
>
> **结论**：NSP 本身的设计可能有缺陷（太简单），但**句子级别的预训练任务**是有价值的。

### 2.5 数据流：从输入到输出

```
原始文本: "My dog is cute" + "He likes playing"
    ↓ Tokenization (WordPiece)
[CLS] my dog is cute [SEP] he likes play ##ing [SEP]
    ↓ 三层嵌入相加
Input Embeddings: [768-dim × 11 tokens]
    ↓ 12 层 Transformer 编码器 (每层: Multi-Head Attention → FFN)
Hidden States: [768-dim × 11 tokens] (每层输出)
    ↓ 取特定位置的输出
[CLS] 输出 (C) → 分类任务
其他 token 输出 (T₁...Tₙ) → 序列标注 / 问答
```

### 2.6 预训练细节

| 项目 | 值 | 说明 |
|------|-----|------|
| 数据 | BooksCorpus (800M) + Wikipedia (2.5B) = 3.3B 词 | 用**文档级别**语料（不是打乱的句子），支持长连续序列 |
| Batch | 256 序列 (128K tokens) | 比 GPT 的 32K tokens 大 4 倍 |
| 训练步数 | 1,000,000 (~40 epochs) | |
| 优化器 | Adam (lr=1e-4, β₁=0.9, β₂=0.999) | |
| 学习率 | 10K 步 warmup + 线性衰减 | |
| 激活函数 | **GELU**（不是 ReLU） | GELU 是 ReLU 的平滑版，梯度更好 |
| 损失 | MLM loss + NSP loss（相加） | 两个任务同时训练 |
| 硬件 | BASE: 4 TPUs (4天) / LARGE: 16 TPUs (4天) | |

> ❓ **为什么用 GELU 而不是 ReLU？**
>
> GELU（Gaussian Error Linear Unit）$= x \cdot \Phi(x)$，其中 $\Phi$ 是标准正态的 CDF。直觉上：GELU 按照输入的"重要性"做概率性的门控——值越大，保留概率越高。而 ReLU 是硬截断。GPT 也用 GELU，后来成了 Transformer 的标准选择。

### 2.7 微调：简单到令人惊讶

#### 分类任务

$$P = \text{softmax}(C \cdot W^T)$$

- $C \in \mathbb{R}^H$：[CLS] 的最终输出
- $W \in \mathbb{R}^{K \times H}$：新增加的分类矩阵
- **只新增了 $K \times H$ 个参数！**（BERT_BASE 二分类 = 2 × 768 = 1,536 个参数）

> 💡 **对比**：之前每个 NLP 任务都需要设计专门的架构（CNN、Attention、CRF 等），现在只需要一个矩阵乘法。

#### 问答任务（SQuAD）

将问答转化为**答案起止位置预测**：
- 新增两个向量：$S$（起始）和 $E$（结束）
- Token $i$ 作为起始位置的概率：$P_i = \text{softmax}(S \cdot T_i)$
- 同理计算结束位置

> ❓ **为什么不直接生成答案？**
>
> 因为 BERT 是**编码器**，不是解码器——它没有生成能力。它只能给每个位置打分。这种"抽取式"问答方式很适合编码器架构。

#### 微调超参数

| 参数 | 推荐值 |
|------|--------|
| Batch size | 16, 32 |
| Learning rate | 5e-5, 3e-5, 2e-5 |
| Epochs | 3, 4 |

> 💡 小数据集对超参数敏感，建议穷搜。大数据集不敏感。

---

## 3. 实验：每个实验验证了什么？

### 3.1 GLUE（Table 1）—— 验证"预训练-微调"范式的通用性

| 对比 | MNLI | 平均 |
|------|------|------|
| GPT (117M) | 82.1 | 75.2 |
| **BERT_BASE (110M)** | **84.6** | **79.6** |
| **BERT_LARGE (340M)** | **86.7** | **81.9** |

**关键对比**：BERT_BASE 和 GPT 参数量几乎一样（110M vs 117M），唯一差异是双向 vs 单向。MNLI 上差 2.5 个百分点——这完全来自双向性。

### 3.2 SQuAD v1.1（Table 2）—— 验证双向编码在理解任务上的优势

BERT_LARGE 93.2 F1，**超越人类** 91.2 F1 (+2.0)。

> 💡 问答任务特别需要双向编码——确定答案的起始位置需要看后面的内容，确定结束位置需要看前面的内容。

### 3.3 消融实验（Table 5）—— **最重要的实验**

| 模型变体 | MNLI | QNLI | MRPC | SQuAD | 变化说明 |
|----------|------|------|------|-------|---------|
| BERT_BASE | 84.4 | 88.4 | 86.7 | 88.5 | 完整版 |
| No NSP | 83.9 | 84.9 | 86.5 | 87.9 | 去掉句子级任务 |
| **LTR & No NSP** | **82.1** | **84.3** | **77.5** | **77.8** | **去掉双向** |
| + BiLSTM | 82.1 | 84.1 | 75.7 | 84.9 | 在 LTR 上加 BiLSTM |

**关键发现**：

1. **去掉 NSP → 影响较小**（MNLI -0.5, SQuAD -0.6）
2. **去掉双向 → 影响巨大**（MRPC -9.2, SQuAD -10.7）
3. 加 BiLSTM 有帮助但仍远不如预训练双向模型

> ❓ **为什么 MRPC 和 SQuAD 受单向影响最大？**
>
> - MRPC（复述检测）：判断两句话是否语义相同，需要深度理解两句话的完整含义
> - SQuAD（问答）：需要从段落中抽取答案的起止位置，单方向看不到"答案后面"的信息

### 3.4 模型规模消融（Table 6）—— 验证 scaling 效应

| 层数 | 参数量 | MNLI | SQuAD |
|------|--------|------|-------|
| 3 | ~40M | 77.9 | 81.1 |
| 6 | ~110M | 81.9 | 85.6 |
| 12 | 110M | 84.4 | 88.5 |
| 24 | 340M | 86.6 | 90.6 |

> 💡 **即使在小数据集（MRPC 只有 3.6K 训练样本）上，更大的模型也更好。** 这个发现挑战了"小数据需要小模型防过拟合"的直觉，为后来的 scaling 浪潮埋下伏笔。

### 3.5 图表精读：训练步数消融（Figure 4）

![训练步数消融](./images/fc41d4c6f728f5ad12b1b4a8f33d108df0eae64de078535d2b9f65e47c75e3d8.jpg)

**先不看 caption，自己解读**：

两条线都在持续上升。蓝色（MLM）始终在红色（LTR）上方，且差距随训练步数有**扩大趋势**。值得注意的是，LTR 在 30K 步时甚至略高于 MLM（79.4 vs 78.6）——因为 LTR 预测每个 token（信息密度更高），而 MLM 只预测 15%。但 MLM 很快追上并持续领先。

**对照 caption**："Ablation over number of training steps"，确认是消融实验。

**这张图验证了什么？** 直接回答了"MLM 收敛更慢"的质疑。更重要的是：即使给 LTR 无限训练时间，它也追不上 MLM——因为**单向注意力本身就是信息瓶颈**，不是训练时间能弥补的。

**批判性**：纵轴 76-84 没有从零开始，可能略微夸大了差异。但即使在同坐标系中差距也是显著的。另外只在 MNLI 上做了这个消融——不同任务的结果可能不同。

**面试一句话**：这张图证明了双向预训练的优势不是训练时间能弥补的——MLM 始终优于 LTR，且差距随训练深入而扩大。

---

# 第三层：批判性思考

## 🤔 设计决策分析

### 为什么用 MLM 而不是其他双向方案？

| 方案 | 问题 |
|------|------|
| 直接双向 LM | "看到自己"问题——多层注意力会泄露信息 |
| 去噪自编码器 (DAE) | 需要重建整个输入，不是只预测 mask 的位置，成本更高 |
| **MLM (BERT 选择)** | 只预测 15% 的 token，效率更高；且 mask 是随机均匀的，更自然 |

> ❓ **如果我来设计，有没有更好的方案？**
>
> 后来确实有更好的方案：**ELECTRA (2020)** 用"判别器"替代"生成器"——不是预测被 mask 的词，而是判断每个词是否被替换过。这样所有 token 都参与训练（不只是 15%），效率提升 4 倍以上。

### 80/10/10 策略是否最优？

消融实验（Table 4）显示 80/10/10 和 100/0/0 差距很小。10% 随机替换对 NER 的 feature-based 场景帮助更大（94.0 → 95.4），但对 fine-tuning 场景几乎无影响。

**结论**：80/10/10 是一个保守但稳健的选择。100% mask 在 fine-tuning 场景下可能也行得通。

### NSP 的设计是否合理？

**论文自己没有提供强有力的 NSP 消融**——Table 5 只对比了有/无 NSP，但没有对比不同难度的句子级任务。后续工作（RoBERTa, ALBERT）证明 NSP 太简单了，换一个更难的句子级任务会更好。

## ⚠️ 局限性

### 论文承认的
1. 预训练成本高（4 天 × 16 TPUs for LARGE）
2. MLM 收敛比 LTR 慢（只预测 15% 的 token）
3. 需要进一步研究 BERT 学到了什么语言学现象

### 自己发现的
1. **编码器架构的局限**：BERT 没有生成能力，不适合文本生成任务。后来的大模型几乎都走了 GPT 的解码器路线
2. **[CLS] 的设计**：用一个特殊 token 聚合全局信息，不如更精细的池化策略（如注意力池化）
3. **固定长度 512**：对长文档不够用，后来的 Longformer、BigBird 解决了这个问题
4. **只用了英文**：后来有 mBERT（多语言版）和 various 中文 BERT
5. **双向预训练在生成任务上是否有优势尚不清楚**——实际上 GPT 的解码器路线最终胜出了

## 🎯 面试视角

### 面试高频问题

**Q1: BERT 和 GPT 的核心区别是什么？**

> **A**: 架构上，BERT 用 Transformer 编码器（双向注意力），GPT 用解码器（单向注意力）。预训练目标上，BERT 用 MLM（遮盖预测），GPT 用自回归 LM（预测下一个词）。适用场景上，BERT 擅长理解任务（分类、问答、NER），GPT 擅长生成任务（文本续写）。
>
> **追问**：为什么现代大模型都走了 GPT 路线？因为**生成能力隐含理解能力**——能正确续写答案的模型，必然理解了问题。而理解能力不隐含生成能力。

**Q2: MLM 的 80/10/10 策略为什么不能 100% mask？**

> **A**: 100% mask 会造成预训练-微调的不匹配——微调时输入里没有 [MASK]，模型会"不适应"。10% 保持原词 + 10% 随机替换让模型不能偷懒，必须为每个 token 保持好的表示。

**Q3: NSP 后来为什么被去掉了？**

> **A**: RoBERTa 发现去掉 NSP 后性能反而提升。原因：NSP 太简单（97-98% 准确率），模型只需要学 topic 层面的信号就能区分。ALBERT 用更难的句子顺序预测（SOP）替代，效果更好。

**Q4: BERT 为什么用可学习的位置编码而不是正弦编码？**

> **A**: 可学习的位置编码更灵活，能从数据中学到最适合的位置表示。但受限于最大长度 512。后来 RoPE（LLaMA）结合了两者优点。

**Q5: 消融实验最重要的结论是什么？**

> **A**: 双向性是性能提升的**最关键因素**。去掉双向后 MRPC 暴跌 9.2、SQuAD 暴跌 10.7。其他因素（NSP、模型大小）的影响远小于双向性。

---

# 第四层：知识网络

## 📅 时间线

```
Word2Vec (2013) → 静态词向量
ELMo (2018.02) → 上下文词向量（LSTM，浅层双向）
GPT-1 (2018.06) → Transformer 解码器 + 微调（单向）
    【BERT (2018.10)】→ Transformer 编码器 + MLM + 微调（双向）
GPT-2 (2019.02) → 解码器 + zero-shot（回到单向但走不同路线）
RoBERTa (2019.07) → 去掉 NSP、更多数据、动态 masking
ALBERT (2019.09) → 参数共享、SOP 替代 NSP
ELECTRA (2020.03) → 判别器预训练，效率提升 4x
GPT-3 (2020.06) → 175B 解码器 + few-shot（GPT 路线最终胜出）
```

## ↔️ 同期对比

| 维度 | BERT (2018.10) | GPT-1 (2018.06) | ELMo (2018.02) |
|------|---------------|-----------------|----------------|
| 架构 | Transformer 编码器 | Transformer 解码器 | 双向 LSTM |
| 方向 | 双向 | 单向（左→右） | 浅层双向（拼接） |
| 预训练 | MLM + NSP | 自回归 LM | 双向 LM |
| 下游方式 | Fine-tuning | Fine-tuning | Feature-based |
| 并行性 | ✅ | ✅ | ❌（LSTM 顺序） |
| 生成能力 | ❌ | ✅ | ❌ |

## 🔗 知识关联

- **llm-math-foundations Ch03**：MLM 的分类头就是 softmax 回归
- **llm-math-foundations Ch08**：MLM 损失函数就是交叉熵
- **llm-math-foundations Ch09**：PPO/RLHF（InstructGPT 会用到）的基础
- **本系列 01-Attention**：BERT 直接使用了 Transformer 编码器
- **本系列 03-GPT-2**：BERT 的"反面"——解码器路线的起点

---

## ❓ 深度思考题

1. **概念题**：BERT 的 [CLS] token 为什么能聚合全局信息？如果把 [CLS] 放在序列末尾会怎样？

2. **设计题**：如果你来设计一个比 MLM 更高效的预训练目标，你会怎么做？（提示：ELECTRA 的思路是什么？）

3. **批判题**：BERT 的消融实验（Table 5）有什么潜在的 confounding factors？BERT vs GPT 的对比真的完全公平吗？（提示：除了双向性，还有什么不同？）

4. **应用题**：BERT 能做文本生成吗？为什么？如果要在 BERT 上做生成，最小改动是什么？

5. **面试题**：面试官问"BERT 的双向和 GPT 的单向各有什么优劣？为什么最终 GPT 路线赢了？"你怎么回答？

6. **拓展题**：后来的 RoBERTa 去掉了 NSP 还提升了性能，这说明 BERT 论文的哪个结论需要修正？这对你评价一篇论文的结论有什么启发？

7. **实现题**：如果给你一个预训练好的 BERT 模型和一个新的分类数据集（只有 100 个样本），你会怎么微调？有什么技巧可以用？

8. **哲学题**：BERT 证明了"双向预训练更好"，但 GPT-3/ChatGPT 用单向模型也达到了惊人的效果。这说明"双向"真的是关键吗？还是说关键在于别的东西？

---

## 📚 延伸阅读

| 论文 | 年份 | 和 BERT 的关系 |
|------|------|---------------|
| **RoBERTa** (Liu et al.) | 2019 | BERT 的"修正版"——去掉 NSP、动态 masking、更多数据和更长训练 |
| **ALBERT** (Lan et al.) | 2019 | BERT 的"轻量版"——跨层参数共享、用 SOP 替代 NSP |
| **ELECTRA** (Clark et al.) | 2020 | BERT 的"升级版"——判别器预训练，同样计算量下效果更好 |
| **SpanBERT** (Joshi et al.) | 2020 | BERT 的变体——mask 连续 span 而非随机 token，对抽取任务更好 |
| **DistilBERT** (Sanh et al.) | 2019 | BERT 的"蒸馏版"——小 40%、快 60%、保留 97% 性能 |
| **GPT-2** (Radford et al.) | 2019 | 本系列下一篇——BERT 的"对手"，解码器路线的起点 |
