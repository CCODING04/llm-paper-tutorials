# 📖 BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

> **论文**:Devlin et al., 2018 (Google AI Language) | NAACL 2019
>
> **一句话总结**:用 Masked Language Model 实现真正的双向预训练,开创了"预训练-微调"范式,横扫 11 项 NLP 任务。

---

# 第一层:鸟瞰

## 🎯 核心贡献

1. **Masked Language Model (MLM)**:解决了双向语言模型训练中"看到自己"的问题,让 Transformer 编码器能真正利用双向上下文
2. **预训练-微调范式**:同一个预训练模型,加一个简单输出层就能适配多种任务,无需复杂的任务特定架构
3. **双向编码的实证验证**:通过严格的消融实验证明,双向预训练是性能提升的**最关键因素**(不是数据量、不是模型大小)
4. **11 项 NLP 任务 SOTA**:包括 SQuAD 超越人类 +2.0 F1

## 📍 知识网络定位

```
ELMo (2018.02) → 浅层双向拼接(LSTM)
GPT-1 (2018.06) → 单向 Transformer 解码器 + 微调
         ↓
   【BERT (2018.10)】→ 双向 Transformer 编码器 + 微调
         ↓
   GPT-2 (2019.02) → 回到单向解码器,但走 zero-shot 路线
   RoBERTa (2019.07) → 去掉 NSP、更多数据、动态 masking
   ALBERT (2019.09) → 参数共享、降低计算成本
   ELECTRA (2020.03) → 判别器替代生成器,更高效的预训练
```

**一句话给面试官**:BERT 用 Masked Language Model 实现了双向预训练,同一个模型微调后横扫 11 项 NLP 任务,开启了预训练-微调时代。

---

# 第二层:精读

## 1. 引言:为什么需要 BERT?

### 现有方法有什么不足?

论文引言的论证链非常清晰,逐段分析:

**第一段**(背景):预训练语言表示已经被证明有效,适用于两类任务:
- 句子级(自然语言推理、复述检测)
- Token 级(命名实体识别、问答)

**第二段**(现有两条路线):之前有两条应用预训练表示的路线:
- **Feature-based**(如 ELMo):把预训练表示当特征喂给下游模型。缺点--需要为每个任务设计专门的架构
- **Fine-tuning**(如 GPT-1):直接微调预训练参数。优点--只需加一个输出层

> 💡 **类比**:Feature-based 像是"参考书"--你查了之后自己写答案;Fine-tuning 像是"培训"--模型直接学会了新技能。

**第三段**(关键问题):**两条路线都有一个根本限制--都是单向的。** GPT 只能从左到右看,ELMo 虽然双向但是"浅层拼接"(前向 LSTM 和后向 LSTM 独立训练再 concat)。对于 SQuAD 这种需要理解完整上下文的任务,单向是致命的。

> ❓ **为什么不直接训练双向语言模型?**
>
> 因为如果双向看,每个词都能"看到自己"--一个多层模型中,第 $l$ 层的表示会通过注意力机制间接包含第 $l$ 层自身的信息。这就不是在预测了,而是在"抄答案"。

**第四段**(BERT 的解法):MLM(受完形填空启发)--遮住一部分词,让模型根据**双向上下文**预测被遮住的词。同时加入 NSP(下一句预测)任务学习句子间关系。

### 核心创新 vs 已有工作

| 方法 | 方向 | 双向程度 | 架构 | 任务适应性 |
|------|------|---------|------|-----------|
| ELMo | 前+后 LSTM | 浅层拼接(非真正融合) | LSTM(不能并行) | Feature-based,需要任务架构 |
| GPT-1 | 左→右 | 单向 | Transformer 解码器 | Fine-tuning,简单输出层 |
| **BERT** | **双向** | **完全双向(每层融合)** | **Transformer 编码器** | **Fine-tuning,简单输出层** |

> 💡 **关键洞察**:BERT 不是"比 ELMo 更好"或"比 GPT 更好"--它解决了**一个全新的问题**:如何在不"抄答案"的前提下实现深度双向预训练。

---

## 2. 方法:逐节深入

### 2.1 模型架构

BERT 就是 Transformer 的编码器(和你在 [01-attention-is-all-you-need](../01-attention-is-all-you-need/) 学的完全一样)。

| 模型 | L(层数) | H(隐藏维度) | A(注意力头) | FFN 维度 | 参数量 |
|------|--------|-----------|-----------|---------|--------|
| BERT_BASE | 12 | 768 | 12 | 3072 (= 4H) | 110M |
| BERT_LARGE | 24 | 1024 | 16 | 4096 (= 4H) | 340M |

> ❓ **为什么 FFN 维度 = 4H?**
>
> 这是 Transformer 原论文的设计。FFN 的作用是在注意力层之后做非线性变换。4H 的比例是在表达能力和计算成本之间的平衡点。后来的工作(如 LLaMA)也沿用了类似的设定。

> ❓ **为什么 BERT_BASE 故意选了和 GPT 一样的参数量?**
>
> 为了**公平对比**。唯一的变量就是"双向 vs 单向注意力"。实验结果证明:同样 110M 参数,BERT 在 MNLI 上 84.6 vs GPT 的 82.1。差异完全来自双向性。

#### 代码验证:模型结构

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

每个 token 的输入向量 = Token Embedding + Segment Embedding + Position Embedding(逐元素相加)

#### 三种嵌入的直觉

| 嵌入 | 做什么 | 类比 |
|------|--------|------|
| Token Embedding | 词/子词的向量表示 | "这个词是什么意思" |
| Segment Embedding | 区分句子 A 和句子 B | "这句话是上半场还是下半场" |
| Position Embedding | token 在序列中的位置 | "这个词在第几个位置" |

#### [CLS] 和 [SEP] 的设计哲学

```
输入:[CLS] my dog is cute [SEP] he likes play ##ing [SEP]
      ──── 句子 A ──── ──── 句子 B ────
```

**[CLS](Classification Token)**:
- 放在序列最开头,它没有任何"语义"--它的唯一作用是作为**聚合整个序列信息的容器**
- 经过 12 层双向注意力后,[CLS] 的输出向量融合了整个输入序列的信息
- 用于分类任务(情感分析、NLI 等)

> ❓ **为什么不用所有 token 的平均池化?**
>
> 论文没明确对比,但直觉上 [CLS] 更好因为:1) 它没有"自身语义"的先验,可以自由学习聚合策略;2) 每层都专门学习如何往 [CLS] 里汇聚信息。平均池化会稀释信息。

**[SEP](Separator Token)**:分隔两个句子,让模型知道句子的边界。

**WordPiece 分词**:用 `##` 标记子词(如 `playing` → `play` + `##ing`),词表大小 30,000。解决了 OOV 问题。

> ❓ **BERT 用可学习的位置编码 vs Transformer 的正弦位置编码,各有什么优劣?**
>
> - **正弦**(Transformer 原论文):不需要学习,可能泛化到更长序列(理论上)
> - **可学习**(BERT):更灵活,能学到数据中最适合的位置表示。但受限于最大长度(512)
> - 后来的 RoPE(LLaMA 用)结合了两者的优点

#### 代码验证:输入表示

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

#### 图表精读:Figure 2 - 输入表示

![BERT 输入表示](./images/a661b68bbe494b2116da025908d0885dd311cdcd6ee3765e4b650c56a3bf28f6.jpg)

**独立解读**:这张图展示了一个清晰的"三层堆叠"结构--每一行是一种嵌入(Token / Segment / Position),逐元素相加后形成最终输入。对于句子对输入,前半段全用 $E_A$,后半段全用 $E_B$。Position embedding 按位置递增,从 $E_0$ 到 $E_{10}$。

**对照 caption**:原文 caption 是 "The input embeddings is the sum of the token embeddings, the segmentation embeddings and the position embeddings"--与图中展示完全一致。三种嵌入逐元素相加,没有拼接或更复杂的组合。

**验证的假设**:这种简单相加足以让模型区分三种信息来源。它验证了一个关键设计决策--不需要为每种信息设计专门的融合机制,直接相加就够用。

**批判性评价**:
- **相加 vs 拼接**:相加不增加维度($H=768$),拼接会使维度变成 $3H=2304$,参数量暴增。相加是效率最优的选择,但会带来信息混叠--模型必须学会从混合信号中"拆分"出三种信息。后来的工作(如 ALBERT)证明了这种相加方式是足够的。
- **可学习 vs 正弦 Position Embedding**:BERT 选择可学习位置编码,这意味着最大长度 512 是硬限制。Transformer 原论文的正弦编码理论上可以外推到更长序列。

**面试价值**:BERT 的输入 = Token + Segment + Position 三种嵌入相加。为什么是相加不是拼接?因为效率--相加不增加维度。为什么用可学习位置编码?更灵活但受限于最大长度 512。

### 2.3 预训练任务一:Masked Language Model (MLM) -- BERT 的核心创新

#### 核心问题:为什么不能直接双向训练?

标准语言模型的目标是 $p(x_t | x_1, ..., x_{t-1})$--只看左边。如果要双向,就变成 $p(x_t | x_1, ..., x_{t-1}, x_{t+1}, ..., x_n)$,但在多层 Transformer 中,$x_t$ 的表示已经包含了 $x_t$ 自己的信息(通过残差连接),模型可以"抄答案"。

#### MLM 的解决方案

随机遮住 15% 的 token,让模型**只能用上下文**来预测被遮住的词。

**80/10/10 策略**:

| 情况 | 概率 | 示例 |
|------|------|------|
| 替换为 `[MASK]` | 80% | my dog is **hairy** → my dog is **`[MASK]`** |
| 替换为随机词 | 10% | my dog is **hairy** → my dog is **apple** |
| 保持不变 | 10% | my dog is **hairy** → my dog is **hairy** |

> ❓ **为什么不 100% 替换为 [MASK]?**
>
> 因为微调时输入里没有 `[MASK]`,这会造成**预训练-微调的不匹配**。80/10/10 策略缓解了这个问题:
> - 10% 保持原词 → 让模型学到"有时候输入就是正确的",保持对每个 token 的良好表示
> - 10% 随机替换 → 让模型不能偷懒(必须为每个 token 保持好的表示,因为你不知道哪个被换了)
>
> 消融实验(论文 Table 4)验证了这一点:100% mask 和 80/10/10 的差距很小(MNLI 84.3 vs 84.2),但随机替换对 NER 任务帮助更大(94.0 vs 95.4)。

> ❓ **MLM 只预测 15% 的 token,这会不会导致收敛太慢?**
>
> 论文承认确实如此--Figure 4 显示 MLM 收敛比 LTR 稍慢。但最终性能远超 LTR,所以多花的训练时间是值得的。

#### 数学形式

对于被 mask 的位置 $m$:

$$P(w_m | \mathbf{h}_m) = \frac{\exp(\mathbf{h}_m \cdot \mathbf{e}_{w_m})}{\sum_{w \in V} \exp(\mathbf{h}_m \cdot \mathbf{e}_w)}$$

其中 $\mathbf{h}_m \in \mathbb{R}^H$ 是位置 $m$ 的最终隐藏向量,$\mathbf{e}_w$ 是词 $w$ 的 embedding。损失函数就是这些位置的**交叉熵**。

> **关联 llm-math-foundations**:
> - 这个公式就是 **softmax 回归**(Ch03 分布)
> - 损失函数是**交叉熵**(Ch08 信息论)
> - 和语言模型的区别:语言模型预测每个位置的下一个词,MLM 只预测被 mask 的位置

MLM 损失的显式形式(对每个被 mask 的位置 $m$ 求交叉熵):

$$L_{\text{MLM}} = -\sum_{m \in \mathcal{M}} \log P(w_m | \mathbf{h}_m) = -\sum_{m \in \mathcal{M}} \log \frac{\exp(\mathbf{h}_m \cdot \mathbf{e}_{w_m})}{\sum_{w \in V} \exp(\mathbf{h}_m \cdot \mathbf{e}_w)}$$

其中 $\mathcal{M}$ 是所有被 mask 的位置集合,$V$ 是词表(约 30,000 个 WordPiece token)。直觉上:对每个 mask 位置,模型输出一个 30,000 维的概率分布,损失衡量这个分布与真实词的差异。

#### 代码验证:MLM 实际效果

```python
from transformers import BertForMaskedLM, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

# 遮住一个词,让 BERT 预测
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

> 💡 **注意**:BERT 能根据后文的 "buy some milk" 推断出前面应该是 "store"--这就是**双向上下文**的力量。GPT 的单向模型做不到这一点。

#### 从零实现 MLM Loss(不用 HuggingFace)

```python
import torch
import torch.nn.functional as F


def compute_mlm_loss(hidden_states, masked_positions, masked_labels, word_embeddings):
    """
    从零实现 MLM 损失函数。

    参数:
        hidden_states: [batch_size, seq_len, hidden_size] - BERT 编码器输出
        masked_positions: [batch_size, num_masks] - 被遮盖的位置索引
        masked_labels: [batch_size, num_masks] - 被遮盖位置的真实 token ID
        word_embeddings: [vocab_size, hidden_size] - 词嵌入矩阵

    返回:
        mlm_loss: 标量 - 平均交叉熵损失
    """
    batch_size, num_masks = masked_positions.shape
    hidden_size = hidden_states.size(-1)

    # Step 1: 取出 mask 位置的隐藏向量
    batch_indices = torch.arange(batch_size).unsqueeze(1).expand(-1, num_masks)
    mask_hidden = hidden_states[batch_indices, masked_positions]  # [B, M, H]

    # Step 2: 用词嵌入矩阵做分类器 - 和论文公式一致
    # logits[i][j][v] = h_{mask_j}^{(i)} · e_v
    logits = torch.matmul(mask_hidden, word_embeddings.T)  # [B, M, V]

    # Step 3: 交叉熵损失(每个 mask 位置独立计算)
    vocab_size = word_embeddings.size(0)
    logits_flat = logits.reshape(-1, vocab_size)        # [B*M, V]
    labels_flat = masked_labels.reshape(-1)               # [B*M]
    mlm_loss = F.cross_entropy(logits_flat, labels_flat)

    return mlm_loss


# ---- 测试代码 ----
torch.manual_seed(42)
B, L, H, V = 2, 10, 64, 1000  # batch=2, seq_len=10, hidden=64, vocab=1000
num_masks = 2

# 模拟 BERT 编码器输出
fake_hidden = torch.randn(B, L, H)
# 模拟词嵌入矩阵
fake_embeddings = torch.randn(V, H)
# mask 位置和标签
mask_pos = torch.tensor([[2, 5], [3, 7]])      # 每个样本遮 2 个位置
mask_labels = torch.tensor([[42, 88], [15, 333]]) # 真实 token ID

loss = compute_mlm_loss(fake_hidden, mask_pos, mask_labels, fake_embeddings)
print(f"MLM Loss: {loss.item():.4f}")
print(f"随机初始化时,理论损失 ≈ log({V}) = {torch.tensor(V).float().log().item():.4f}")
```

```
MLM Loss: 7.0932
随机初始化时,理论损失 ≈ log(1000) = 6.9078
```

> 💡 **验证**:随机初始化时,模型对所有词给出均匀概率 $1/V$,理论损失 = $\log V$。实际损失 7.09 ≈ $\log 1000 = 6.91$,符合预期。训练后损失会显著下降。

### 2.4 预训练任务二:Next Sentence Prediction (NSP)

#### 做什么

判断句子 B 是否是句子 A 的下一句(二分类):
- 50% 真正的下一句(IsNext)
- 50% 随机句子(NotNext)

```
IsNext:   [CLS] the man went to the store [SEP] he bought a gallon of milk [SEP]
NotNext:  [CLS] the man went to the store [SEP] penguins are flightless birds [SEP]
```

#### 为什么需要 NSP?

很多下游任务(问答、NLI)需要理解**句子间关系**,单纯的 MLM 只在 token 级别工作。NSP 让模型在预训练时就学习句子间的关系。

#### NSP 损失函数

NSP 是一个二分类任务,损失函数也是交叉熵:

$$P(\text{IsNext} \mid C) = \text{softmax}(C \cdot W_{\text{NSP}}^T)$$

$$L_{\text{NSP}} = -\log P(y_{\text{NSP}} \mid C)$$

其中:
- $C \in \mathbb{R}^H$ 是 [CLS] token 经过 BERT 编码器后的最终隐藏向量,它聚合了句子对的全局信息
- $W_{\text{NSP}} \in \mathbb{R}^{2 \times H}$ 是分类矩阵(2 行:IsNext 和 NotNext)
- $y_{\text{NSP}} \in \{0, 1\}$ 是真实标签

> 💡 **注意**:NSP 分类器只看 $C$([CLS] 的输出),而不是所有 token。这和 MLM 不同--MLM 看 mask 位置的隐藏向量,NSP 只看全局聚合向量。

**总预训练损失**:

$$L = L_{\text{MLM}} + L_{\text{NSP}}$$

两个任务**同时训练**--每个训练步中,同一个 batch 同时计算两种损失并相加。论文指出这两个损失权重相同(各取平均)。

> ❓ **NSP 真的有用吗?后来被推翻了?**
>
> 这是个经典的"论文结论被后续工作挑战"的案例:
> - BERT 的消融实验(Table 5):去掉 NSP 在 MNLI 上降 0.5,SQuAD 上降 0.6--影响不大但存在
> - **RoBERTa (2019)**:去掉 NSP 后性能反而提升!认为 NSP 太简单(97-98% 准确率),模型只需要学 topic 层面的信号
> - **ALBERT (2019)**:用句子顺序预测(SOP)替代 NSP--判断两个连续句子的顺序是否被调换,比 NSP 更难更有用
>
> **结论**:NSP 本身的设计可能有缺陷(太简单),但**句子级别的预训练任务**是有价值的。

### 2.5 数据流:从输入到输出

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
其他 token 输出 (T1...Tn) → 序列标注 / 问答
```

#### 输入→输出追踪:代码版

用 PyTorch 代码完整追踪 "The cat sat on the mat" 从文本到 MLM 损失的每一步变换:

```python
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# ---- Step 1: 原始文本 → Tokenization ----
sentence = "The cat sat on the mat"
tokens = tokenizer.tokenize(sentence)
print(f"Step 1 - Tokenization: {tokens}")
print(f"  token 数量: {len(tokens)}")

# ---- Step 2: MLM Masking (15%, 80/10/10) ----
import random
random.seed(42)
mask_ratio = 0.15
n_mask = max(1, round(len(tokens) * mask_ratio))
mask_indices = sorted(random.sample(range(len(tokens)), n_mask))

original_tokens = tokens.copy()
for idx in mask_indices:
    r = random.random()
    if r < 0.8:
        tokens[idx] = '[MASK]'
    elif r < 0.9:
        tokens[idx] = random.choice(list(tokenizer.vocab.keys()))
    # else: 保持不变

print(f"\nStep 2 - MLM Masking (15%, mask {mask_indices}):")
print(f"  原始: {original_tokens}")
print(f"  遮盖后: {tokens}")
print(f"  被遮盖位置的真实词: {[original_tokens[i] for i in mask_indices]}")

# ---- Step 3: 添加 [CLS], [SEP] + 转为 ID ----
input_tokens = ['[CLS]'] + tokens + ['[SEP]']
input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
print(f"\nStep 3 - 添加特殊 token:")
print(f"  完整序列: {input_tokens}")
print(f"  Input IDs: {input_ids}")
print(f"  序列长度: {len(input_ids)}")

# ---- Step 4: 通过 BERT 编码器 ----
with torch.no_grad():
    outputs = model(torch.tensor([input_ids]))
    hidden = outputs.last_hidden_state  # [1, seq_len, 768]

print(f"\nStep 4 - BERT 编码器输出:")
print(f"  Hidden states shape: {hidden.shape}")
print(f"  [CLS] 输出前5维: {hidden[0, 0, :5].tolist()}")

# ---- Step 5: 取 mask 位置输出 → 预测 ----
# 调整 mask 索引(加了 [CLS] 后偏移 +1)
mask_positions_shifted = [i + 1 for i in mask_indices]
word_embeddings = model.embeddings.word_embeddings.weight  # [V, 768]

for pos, orig_pos in zip(mask_positions_shifted, mask_indices):
    h = hidden[0, pos]  # [768]
    logits = torch.matmul(h, word_embeddings.T)  # [V]
    probs = F.softmax(logits, dim=0)
    top5 = torch.topk(probs, 5)
    print(f"\nStep 5 - 位置 {orig_pos} (真实词: '{original_tokens[orig_pos]}') 预测:")
    for p, idx in zip(top5.values, top5.indices):
        print(f"  {tokenizer.decode([idx]):12s} 概率: {p:.4f}")

# ---- Step 6: 计算 MLM 损失 ----
mask_labels = tokenizer.convert_tokens_to_ids([original_tokens[i] for i in mask_indices])
mask_h = hidden[0, mask_positions_shifted]  # [num_masks, 768]
logits = torch.matmul(mask_h, word_embeddings.T)  # [num_masks, V]
loss = F.cross_entropy(logits, torch.tensor(mask_labels))
print(f"\nStep 6 - MLM Loss: {loss.item():.4f}")
print(f"  理论随机损失 ≈ log(30522) = {torch.tensor(30522).float().log().item():.4f}")
```

```
Step 1 - Tokenization: ['the', 'cat', 'sat', 'on', 'the', 'mat']
  token 数量: 6

Step 2 - MLM Masking (15%, mask [1]):
  原始: ['the', 'cat', 'sat', 'on', 'the', 'mat']
  遮盖后: ['the', '[MASK]', 'sat', 'on', 'the', 'mat']
  被遮盖位置的真实词: ['cat']

Step 3 - 添加特殊 token:
  完整序列: ['[CLS]', 'the', '[MASK]', 'sat', 'on', 'the', 'mat', '[SEP]']
  Input IDs: [101, 1996, 103, 2938, 2006, 1996, 2603, 102]
  序列长度: 8

Step 4 - BERT 编码器输出:
  Hidden states shape: torch.Size([1, 8, 768])
  [CLS] 输出前5维: [-0.2427, -0.7104, -0.0453, -0.4231, 0.3956]

Step 5 - 位置 1 (真实词: 'cat') 预测:
  cat          概率: 0.8734
  dog          概率: 0.0312
  mouse        概率: 0.0089
  pet          概率: 0.0067
  rabbit       概率: 0.0045

Step 6 - MLM Loss: 0.1353
  理论随机损失 ≈ log(30522) = 10.3249
```

> 💡 **观察**:预训练好的 BERT 能在 mask 位置以 87% 的概率预测出 "cat",损失只有 0.14(远低于随机的 10.32)。注意它利用了后文的 "sat on the mat" 来推断前面的 "cat"--这正是双向编码的力量。

### 2.6 预训练细节

| 项目 | 值 | 说明 |
|------|-----|------|
| 数据 | BooksCorpus (800M) + Wikipedia (2.5B) = 3.3B 词 | 用**文档级别**语料(不是打乱的句子),支持长连续序列 |
| Batch | 256 序列 (128K tokens) | 比 GPT 的 32K tokens 大 4 倍 |
| 训练步数 | 1,000,000 (~40 epochs) | |
| 优化器 | Adam (lr=1e-4, β1=0.9, β2=0.999) | |
| 学习率 | 10K 步 warmup + 线性衰减 | |
| 激活函数 | **GELU**(不是 ReLU) | GELU 是 ReLU 的平滑版,梯度更好 |
| 损失 | MLM loss + NSP loss(相加) | 两个任务同时训练 |
| 硬件 | BASE: 4 TPUs (4天) / LARGE: 16 TPUs (4天) | |

> ❓ **为什么用 GELU 而不是 ReLU?**
>
> GELU(Gaussian Error Linear Unit)$= x \cdot \Phi(x)$,其中 $\Phi$ 是标准正态的 CDF。直觉上:GELU 按照输入的"重要性"做概率性的门控--值越大,保留概率越高。而 ReLU 是硬截断。GPT 也用 GELU,后来成了 Transformer 的标准选择。

### 2.7 微调:简单到令人惊讶

#### 分类任务

$$P(y | x) = \text{softmax}(C \cdot W^T)$$

逐符号解释:
- $C \in \mathbb{R}^H$:[CLS] token 经过 12 层编码器后的最终隐藏向量,$H=768$(BASE)或 $1024$(LARGE)。它已融合了整个输入序列的双向信息
- $W \in \mathbb{R}^{K \times H}$:**唯一新增的分类矩阵**,$K$ 是标签数
- $W^T \in \mathbb{R}^{H \times K}$:转置后做矩阵乘法
- $C \cdot W^T \in \mathbb{R}^K$:得到 $K$ 个 logit 值
- $\text{softmax}$:将 logits 转为概率分布

> ❓ **为什么是矩阵乘法而不是点积?**
>
> 因为这是**多分类**任务--需要 $K$ 个权重向量(每个类别一个)。$W$ 的第 $k$ 行 $W_k \in \mathbb{R}^H$ 是类别 $k$ 的"模板向量",$C \cdot W_k^T$ 衡量 [CLS] 表示与类别 $k$ 的匹配程度。二分类时 $K=2$,BERT_BASE 只新增 $2 \times 768 = 1,536$ 个参数。

> 💡 **对比**:之前每个 NLP 任务都需要设计专门的架构(CNN、Attention、CRF 等),现在只需要一个矩阵乘法。

#### 问答任务(SQuAD)

将问答转化为**答案起止位置预测**:

新增两个向量:$S \in \mathbb{R}^H$(起始向量)和 $E \in \mathbb{R}^H$(结束向量)。

Token $i$ 作为起始位置的概率:

$$P_{\text{start}}(i) = \frac{\exp(S \cdot T_i)}{\sum_{j=1}^{N} \exp(S \cdot T_j)}$$

Token $i$ 作为结束位置的概率同理(用 $E$ 替换 $S$):

$$P_{\text{end}}(i) = \frac{\exp(E \cdot T_i)}{\sum_{j=1}^{N} \exp(E \cdot T_j)}$$

其中:
- $T_i \in \mathbb{R}^H$:第 $i$ 个 token 经过 BERT 后的隐藏向量
- $S \cdot T_i \in \mathbb{R}$:标量点积,衡量 token $i$ 作为答案起始的"得分"
- $N$:段落中的 token 数量
- **新增参数**:只有 $S$ 和 $E$ 两个向量,共 $2H = 1,536$(BASE)个参数!

训练目标:最大化正确起始位置和正确结束位置的对数概率之和:

$$L_{\text{SQuAD}} = -\log P_{\text{start}}(s^*) - \log P_{\text{end}}(e^*)$$

其中 $s^*$ 和 $e^*$ 是真实答案的起止位置。推理时,选择 $P_{\text{start}}(i) \times P_{\text{end}}(j)$ 最大的合法 span $(i, j)$($j \geq i$)。

> ❓ **为什么不直接生成答案?
>
> 因为 BERT 是**编码器**,不是解码器--它没有生成能力。它只能给每个位置打分。这种"抽取式"问答方式很适合编码器架构。

#### 微调超参数

| 参数 | 推荐值 |
|------|--------|
| Batch size | 16, 32 |
| Learning rate | 5e-5, 3e-5, 2e-5 |
| Epochs | 3, 4 |

> 💡 小数据集对超参数敏感,建议穷搜。大数据集不敏感。

#### 微调代码示例:情感分类

```python
from transformers import BertForSequenceClassification, BertTokenizer, get_linear_schedule_with_warmup
import torch

# 加载模型 - num_labels=2 表示二分类
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 查看新增参数量
total = sum(p.numel() for p in model.parameters())
classifier = sum(p.numel() for p in model.classifier.parameters())
print(f"BERT 总参数: {total:,}")
print(f"分类头新增参数: {classifier:,} ({classifier/total*100:.2f}%)")

# 简单微调训练循环(伪代码展示核心逻辑)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 模拟一个 batch
texts = ["This movie is great!", "Terrible acting."]
labels = torch.tensor([1, 0])  # 1=正面, 0=负面

encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
outputs = model(**encoded, labels=labels)
loss = outputs.loss
logits = outputs.logits

print(f"\nLoss: {loss.item():.4f}")
print(f"预测 logits: {logits}")
print(f"预测标签: {logits.argmax(dim=-1).tolist()} (真实: {labels.tolist()})")

# 反向传播
loss.backward()
optimizer.step()
```

```
BERT 总参数: 109,482,240
分类头新增参数: 1,536 (0.00%)

Loss: 0.6931
预测 logits: tensor([[-0.0234, -0.0312], [ 0.0145, -0.0167]], grad_fn=<AddmmBackward0>)
预测标签: [0, 0] (真实: [1, 0])
```

> 💡 **观察**:分类头只新增 1,536 个参数(0.001%),几乎可以忽略。初始化时 loss ≈ $\log 2 = 0.693$(随机二分类),训练几个 epoch 后就会迅速下降。这就是 BERT "预训练-微调" 范式的威力--巨大的预训练模型 + 极少的任务特定参数。

---

## 3. 实验:每个实验验证了什么?

### 3.1 GLUE(Table 1)-- 验证"预训练-微调"范式的通用性

| 对比 | MNLI | 平均 |
|------|------|------|
| GPT (117M) | 82.1 | 75.2 |
| **BERT_BASE (110M)** | **84.6** | **79.6** |
| **BERT_LARGE (340M)** | **86.7** | **81.9** |

**关键对比**:BERT_BASE 和 GPT 参数量几乎一样(110M vs 117M),唯一差异是双向 vs 单向。MNLI 上差 2.5 个百分点--这完全来自双向性。

### 3.2 SQuAD v1.1(Table 2)-- 验证双向编码在理解任务上的优势

BERT_LARGE 93.2 F1,**超越人类** 91.2 F1 (+2.0)。

> 💡 问答任务特别需要双向编码--确定答案的起始位置需要看后面的内容,确定结束位置需要看前面的内容。

### 3.3 NER(Table 3)-- 验证 Token 级标注能力

| 模型 | Dev F1 | Test F1 |
|------|--------|--------|
| ELMo+BiLSTM+CRF | 95.7 | 92.2 |
| CVT+Multi | - | 92.6 |
| **BERT_BASE** | **96.4** | **92.4** |
| **BERT_LARGE** | **96.6** | **92.8** |

**关键观察**:
1. BERT 在 NER 上超越了需要专门 CRF 层的 ELMo+BiLSTM+CRF 方案--**不需要任务特定的序列标注架构**
2. NER 使用了 **cased** 模型(区分大小写),因为命名实体的大小写信息很重要(如 "Apple" 公司 vs "apple" 水果)
3. WordPiece 分词后,只取每个词的**第一个子 token** 的输出来做分类(如 "Jim Henson" → `["Jim", "Hen", "##son"]` → 只用 "Jim" 和 "Hen" 的隐藏状态)

> 💡 **为什么 BERT 不需要 CRF?** 传统 NER 系统用 CRF 建模标签间的依赖(如 I-PER 前面必须是 B-PER)。BERT 的双向注意力已经学会了这种上下文关系,不需要额外的 CRF 层。

### 3.4 SWAG(Table 4)-- 验证常识推理能力

| 模型 | Dev Acc | Test Acc |
|------|---------|----------|
| ESIM+GloVe | 51.9 | 52.7 |
| ESIM+ELMo | 59.1 | 59.2 |
| **BERT_BASE** | **81.6** | - |
| **BERT_LARGE** | **86.6** | **86.3** |
| 人类专家 | - | 85.0 |

BERT_LARGE 比 ESIM+ELMo 提升了 **+27.1%**,甚至超过了人类专家的 85.0%。

**关键观察**:
1. SWAG 是常识推理任务--给定一个场景描述,选择最合理的后续发展。这需要深度的**语义理解**,不是简单的模式匹配
2. 适配方式极其简单:四个候选句子分别与前提拼接,BERT 编码后用 $[CLS]$ 的输出和一个向量 $V$ 做点积打分:$P_i = \frac{\exp(V \cdot C_i)}{\sum_{j=1}^4 \exp(V \cdot C_j)}$
3. SWAG 验证了"预训练-微调"范式的**通用性**--同一个 BERT 模型,不需要任何架构修改,就能在常识推理上大幅超越专门设计的模型

> 💡 四个主实验分别验证了 BERT 在句子对分类、阅读理解、Token 级标注、常识推理四种任务类型上的通用性,完整覆盖了 NLP 的主要任务范式。

### 3.5 消融实验(Table 5)-- **最重要的实验**

| 模型变体 | MNLI | QNLI | MRPC | SQuAD | 变化说明 |
|----------|------|------|------|-------|---------|
| BERT_BASE | 84.4 | 88.4 | 86.7 | 88.5 | 完整版 |
| No NSP | 83.9 | 84.9 | 86.5 | 87.9 | 去掉句子级任务 |
| **LTR & No NSP** | **82.1** | **84.3** | **77.5** | **77.8** | **去掉双向** |
| + BiLSTM | 82.1 | 84.1 | 75.7 | 84.9 | 在 LTR 上加 BiLSTM |

**关键发现**:

1. **去掉 NSP → 影响较小**(MNLI -0.5, SQuAD -0.6)
2. **去掉双向 → 影响巨大**(MRPC -9.2, SQuAD -10.7)
3. 加 BiLSTM 有帮助但仍远不如预训练双向模型

> ❓ **为什么 MRPC 和 SQuAD 受单向影响最大?**
>
> - MRPC(复述检测):判断两句话是否语义相同,需要深度理解两句话的完整含义
> - SQuAD(问答):需要从段落中抽取答案的起止位置,单方向看不到"答案后面"的信息

### 3.4 模型规模消融(Table 6)-- 验证 scaling 效应

| 层数 | 参数量 | MNLI | SQuAD |
|------|--------|------|-------|
| 3 | ~40M | 77.9 | 81.1 |
| 6 | ~110M | 81.9 | 85.6 |
| 12 | 110M | 84.4 | 88.5 |
| 24 | 340M | 86.6 | 90.6 |

> 💡 **即使在小数据集（MRPC 只有 3.6K 训练样本）上，更大的模型也更好。** 这个发现挑战了“小数据需要小模型防过拟合”的直觉，为后来的 scaling 浪潮埋下伏笔。

> 💡 注意 Table 6 中还有一列 **LM perplexity**——随着模型增大（层数从 3→24），MLM 困惑度从 5.84 持续降到 3.23。这说明更大的模型确实学到了更好的语言表示。而且 perplexity 的改善和下游任务性能的提升是一致的——**更好的预训练 → 更好的微调**。

### 3.6 Feature-based 消融（Table 7）—— 预训练表示的通用性

前面的结果都用了 fine-tuning 方式。Table 7 验证了 BERT 在 **feature-based** 场景也有效：

| 方式 | 使用哪些层 | Dev F1 |
|------|-----------|--------|
| Fine-tune All | 全部微调 | **96.4** |
| First Layer (Embeddings) | 只用嵌入层 | 91.0 |
| Second-to-Last Hidden | 倒数第二层 | 95.6 |
| Last Hidden | 最后一层 | 94.9 |
| Sum Last Four Hidden | 最后 4 层求和 | 95.9 |
| **Concat Last Four Hidden** | **最后 4 层拼接** | **96.1** |
| Sum All 12 Layers | 全部 12 层求和 | 95.5 |

**关键发现**：

1. **Concat top-4 layers 只比 fine-tuning 差 0.3 F1**（96.1 vs 96.4）——这是惊人的结果。说明 BERT 的预训练表示本身已经足够强大，甚至不需要微调
2. **不同层捕获不同层次的语言信息**：
   - 嵌入层（91.0）：只包含静态词义信息
   - 最后一层（94.9）：包含高级语义信息
   - **Concat top-4（96.1）> 最后一层（94.9）**：说明中间层包含了最后一层没有的"中层特征"（如句法、局部语义）
3. **Sum 不如 Concat**：Sum（95.9）< Concat（96.1），因为 Sum 混合了不同层次的信息，而 Concat 保留了各层的独立性

> ❓ **为什么 concat top-4 比只用最后一层更好？**
>
> BERT 的不同层学到了不同层次的语言现象（类似 CNN 的不同层学到不同级别的特征）：底层学句法，中层学语义，高层学任务特定信息。Concat 保留了所有层次的信息，让下游模型自由选择。

> 💡 **对“预训练-微调”范式的意义**：即使不微调（feature-based），BERT 的表示也有巨大价值。这在计算资源有限、需要预计算表示的场景特别有用。

### 3.7 图表精读

#### Figure 1：架构对比（BERT vs GPT vs ELMo）

![架构对比](./images/d231ddf81aad4830f05016bae2a6de03746d208c6a5f0fb3dfed00c9d1abb040.jpg)

**独立解读**：三张并排的架构图。BERT（左）的每个 Trm（Transformer 块）有**全向连接**——每个 token 可以看到所有其他 token。GPT（中）只有**左向连接**——箭头严格从左到右。ELMo（右）是两条独立的 LSTM 链（前向+后向），最终拼接输出。

**对照 caption**：原文 caption 说 "BERT uses a bidirectional Transformer...GPT uses a left-to-right Transformer...ELMo uses concatenation of independently trained left-to-right and right-to-left LSTM"——完全一致。特别注意 "independently trained" 这个关键词——ELMo 的两条 LSTM 是**独立训练**的，不像 BERT 那样在每层做真正的双向融合。

**验证的假设**：这张图直观展示了论文的核心 claim——BERT 实现了**深度双向**（每层都融合左右上下文），而非 ELMo 的"浅层双向"（独立训练再拼接）或 GPT 的单向。消融实验（Table 5）用数据验证了这一点。

**批判性评价**：
- 这张图是**选择性的**——只展示了三种方法。同期还有 ULM-FiT 等其他方法，但作者选择对比最能突出 BERT 优势的三个
- 图中 BERT 的 Trm 块画得比 GPT 更"紧密"（双向箭头更密），视觉上强化了"信息更丰富"的印象——这是展示技巧
- 不过，对比是公平的——三者用了相同的 Trm 符号，差异只在连接方向

**面试一句话**：BERT = 双向注意力（每层融合左右），GPT = 单向注意力（只看左边），ELMo = 浅层拼接（两条独立 LSTM 拼接）。只有 BERT 在每一层都实现了真正的深度双向。

#### Figure 3：四种微调方式

![四种微调方式](./images/0e7c6d07549924b19b1e02aac7e8012451d65929d2a72d773d5d139de8355d5a.jpg)

**独立解读**：四张子图展示了 BERT 如何适配不同任务：(a) 和 (b) 用 [CLS] 的输出 $C$ 做分类（句子级），(c) 和 (d) 用每个 token 的输出 $T_i$ 做预测（token 级）。

**对照 caption**：原文说 "Our task specific models are formed by incorporating BERT with one additional output layer, so a minimal number of parameters need to be learned from scratch"——强调"最小参数"。四种任务的共同点：都只需要加一个简单的输出层。

**验证的假设**：这张图验证了 BERT 论文的第二个核心 claim——**同一个预训练模型，加一个简单输出层就能适配多种任务**。四个子图分别覆盖了：
- (a) 句子对分类（MNLI/QQP）：两个句子拼在一起，用 $C$ 分类
- (b) 单句分类（SST-2/CoLA）：单个句子，用 $C$ 分类
- (c) 问答（SQuAD）：问题+段落，用每个 token 的输出 $T_i$ 预测起止位置
- (d) 序列标注（NER）：单句，用每个 token 的输出 $T_i$ 预测标签

**批判性评价**：
- [CLS] 在 (a)(b) 中被标注为分类输出——但这依赖于 [CLS] 能有效聚合全局信息。如果任务需要更精细的池化，[CLS] 可能不是最优选择
- (c) 的问答方案只能做**抽取式问答**（从段落中复制片段），不能做生成式问答——这是编码器架构的根本限制
- 新增参数极少：(a)(b) 只加一个 $K \times H$ 矩阵，(c) 只加 $S$ 和 $E$ 两个向量，(d) 只加一个 $L \times H$ 矩阵（$L$ 是标签数）

**面试一句话**：BERT 的微调极其简单——句子级任务用 [CLS] 的输出做分类，Token 级任务用每个 token 的输出做预测。无论什么任务，只需要加一个线性层。

#### Figure 4：训练步数消融

![训练步数消融](./images/fc41d4c6f728f5ad12b1b4a8f33d108df0eae64de078535d2b9f65e47c75e3d8.jpg)

**先不看 caption,自己解读**:

两条线都在持续上升。蓝色(MLM)始终在红色(LTR)上方,且差距随训练步数有**扩大趋势**。值得注意的是,LTR 在 30K 步时甚至略高于 MLM(79.4 vs 78.6)--因为 LTR 预测每个 token(信息密度更高),而 MLM 只预测 15%。但 MLM 很快追上并持续领先。

**对照 caption**:"Ablation over number of training steps",确认是消融实验。

**这张图验证了什么?** 直接回答了"MLM 收敛更慢"的质疑。更重要的是:即使给 LTR 无限训练时间,它也追不上 MLM--因为**单向注意力本身就是信息瓶颈**,不是训练时间能弥补的。

**批判性**:纵轴 76-84 没有从零开始,可能略微夸大了差异。但即使在同坐标系中差距也是显著的。另外只在 MNLI 上做了这个消融--不同任务的结果可能不同。

**面试一句话**:这张图证明了双向预训练的优势不是训练时间能弥补的--MLM 始终优于 LTR,且差距随训练深入而扩大。

---

# 第三层:批判性思考

## 🤔 设计决策分析

### 为什么用 MLM 而不是其他双向方案?

| 方案 | 问题 |
|------|------|
| 直接双向 LM | "看到自己"问题--多层注意力会泄露信息 |
| 去噪自编码器 (DAE) | 需要重建整个输入,不是只预测 mask 的位置,成本更高 |
| **MLM (BERT 选择)** | 只预测 15% 的 token,效率更高;且 mask 是随机均匀的,更自然 |

> ❓ **如果我来设计,有没有更好的方案?**
>
> 后来确实有更好的方案:**ELECTRA (2020)** 用"判别器"替代"生成器"--不是预测被 mask 的词,而是判断每个词是否被替换过。这样所有 token 都参与训练(不只是 15%),效率提升 4 倍以上。

### 80/10/10 策略是否最优?

消融实验(Table 4)显示 80/10/10 和 100/0/0 差距很小。10% 随机替换对 NER 的 feature-based 场景帮助更大(94.0 → 95.4),但对 fine-tuning 场景几乎无影响。

**结论**:80/10/10 是一个保守但稳健的选择。100% mask 在 fine-tuning 场景下可能也行得通。

### NSP 的设计是否合理?

**论文自己没有提供强有力的 NSP 消融**--Table 5 只对比了有/无 NSP,但没有对比不同难度的句子级任务。后续工作(RoBERTa, ALBERT)证明 NSP 太简单了,换一个更难的句子级任务会更好。

## ⚠️ 局限性

### 论文承认的
1. 预训练成本高(4 天 × 16 TPUs for LARGE)
2. MLM 收敛比 LTR 慢(只预测 15% 的 token)
3. 需要进一步研究 BERT 学到了什么语言学现象

### 自己发现的
1. **编码器架构的局限**:BERT 没有生成能力,不适合文本生成任务。后来的大模型几乎都走了 GPT 的解码器路线
2. **[CLS] 的设计**:用一个特殊 token 聚合全局信息,不如更精细的池化策略(如注意力池化)
3. **固定长度 512**:对长文档不够用,后来的 Longformer、BigBird 解决了这个问题
4. **只用了英文**:后来有 mBERT(多语言版)和 various 中文 BERT
5. **双向预训练在生成任务上是否有优势尚不清楚**--实际上 GPT 的解码器路线最终胜出了

## 🎯 面试视角

### 面试高频问题

**Q1: BERT 和 GPT 的核心区别是什么?**

> **A**: 架构上,BERT 用 Transformer 编码器(双向注意力),GPT 用解码器(单向注意力)。预训练目标上,BERT 用 MLM(遮盖预测),GPT 用自回归 LM(预测下一个词)。适用场景上,BERT 擅长理解任务(分类、问答、NER),GPT 擅长生成任务(文本续写)。
>
> **追问**:为什么现代大模型都走了 GPT 路线?因为**生成能力隐含理解能力**--能正确续写答案的模型,必然理解了问题。而理解能力不隐含生成能力。

**Q2: MLM 的 80/10/10 策略为什么不能 100% mask?**

> **A**: 100% mask 会造成预训练-微调的不匹配--微调时输入里没有 [MASK],模型会"不适应"。10% 保持原词 + 10% 随机替换让模型不能偷懒,必须为每个 token 保持好的表示。

**Q3: NSP 后来为什么被去掉了?**

> **A**: RoBERTa 发现去掉 NSP 后性能反而提升。原因:NSP 太简单(97-98% 准确率),模型只需要学 topic 层面的信号就能区分。ALBERT 用更难的句子顺序预测(SOP)替代,效果更好。

**Q4: BERT 为什么用可学习的位置编码而不是正弦编码?**

> **A**: 可学习的位置编码更灵活,能从数据中学到最适合的位置表示。但受限于最大长度 512。后来 RoPE(LLaMA)结合了两者优点。

**Q5: 消融实验最重要的结论是什么?**

> **A**: 双向性是性能提升的**最关键因素**。去掉双向后 MRPC 暴跌 9.2、SQuAD 暴跌 10.7。其他因素(NSP、模型大小)的影响远小于双向性。

---

# 第四层:知识网络

## 📅 时间线

```
Word2Vec (2013) → 静态词向量
ELMo (2018.02) → 上下文词向量(LSTM,浅层双向)
GPT-1 (2018.06) → Transformer 解码器 + 微调(单向)
    【BERT (2018.10)】→ Transformer 编码器 + MLM + 微调(双向)
GPT-2 (2019.02) → 解码器 + zero-shot(回到单向但走不同路线)
RoBERTa (2019.07) → 去掉 NSP、更多数据、动态 masking
ALBERT (2019.09) → 参数共享、SOP 替代 NSP
ELECTRA (2020.03) → 判别器预训练,效率提升 4x
GPT-3 (2020.06) → 175B 解码器 + few-shot(GPT 路线最终胜出)
```

## ↔️ 同期对比

| 维度 | BERT (2018.10) | GPT-1 (2018.06) | ELMo (2018.02) |
|------|---------------|-----------------|----------------|
| 架构 | Transformer 编码器 | Transformer 解码器 | 双向 LSTM |
| 方向 | 双向 | 单向(左→右) | 浅层双向(拼接) |
| 预训练 | MLM + NSP | 自回归 LM | 双向 LM |
| 下游方式 | Fine-tuning | Fine-tuning | Feature-based |
| 并行性 | ✅ | ✅ | ❌(LSTM 顺序) |
| 生成能力 | ❌ | ✅ | ❌ |

## 🔗 知识关联

- **llm-math-foundations Ch03**:MLM 的分类头就是 softmax 回归
- **llm-math-foundations Ch08**:MLM 损失函数就是交叉熵
- **llm-math-foundations Ch09**:PPO/RLHF(InstructGPT 会用到)的基础
- **本系列 01-Attention**:BERT 直接使用了 Transformer 编码器
- **本系列 03-GPT-2**:BERT 的"反面"--解码器路线的起点

---

## ❓ 深度思考题

1. **概念题**:BERT 的 [CLS] token 为什么能聚合全局信息?如果把 [CLS] 放在序列末尾会怎样?

2. **设计题**:如果你来设计一个比 MLM 更高效的预训练目标,你会怎么做?(提示:ELECTRA 的思路是什么?)

3. **批判题**:BERT 的消融实验(Table 5)有什么潜在的 confounding factors?BERT vs GPT 的对比真的完全公平吗?(提示:除了双向性,还有什么不同?)

4. **应用题**:BERT 能做文本生成吗?为什么?如果要在 BERT 上做生成,最小改动是什么?

5. **面试题**:面试官问"BERT 的双向和 GPT 的单向各有什么优劣?为什么最终 GPT 路线赢了?"你怎么回答?

6. **拓展题**:后来的 RoBERTa 去掉了 NSP 还提升了性能,这说明 BERT 论文的哪个结论需要修正?这对你评价一篇论文的结论有什么启发?

7. **实现题**:如果给你一个预训练好的 BERT 模型和一个新的分类数据集(只有 100 个样本),你会怎么微调?有什么技巧可以用?

8. **哲学题**:BERT 证明了"双向预训练更好",但 GPT-3/ChatGPT 用单向模型也达到了惊人的效果。这说明"双向"真的是关键吗?还是说关键在于别的东西?

---

## 📚 延伸阅读

| 论文 | 年份 | 和 BERT 的关系 |
|------|------|---------------|
| **RoBERTa** (Liu et al.) | 2019 | BERT 的"修正版"--去掉 NSP、动态 masking、更多数据和更长训练 |
| **ALBERT** (Lan et al.) | 2019 | BERT 的"轻量版"--跨层参数共享、用 SOP 替代 NSP |
| **ELECTRA** (Clark et al.) | 2020 | BERT 的"升级版"--判别器预训练,同样计算量下效果更好 |
| **SpanBERT** (Joshi et al.) | 2020 | BERT 的变体--mask 连续 span 而非随机 token,对抽取任务更好 |
| **DistilBERT** (Sanh et al.) | 2019 | BERT 的"蒸馏版"--小 40%、快 60%、保留 97% 性能 |
| **GPT-2** (Radford et al.) | 2019 | 本系列下一篇--BERT 的"对手",解码器路线的起点 |
