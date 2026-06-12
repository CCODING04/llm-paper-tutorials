# 🔬 论文公式 ↔ 代码对照：Transformer 手写实现

> 本文档将《Attention Is All You Need》论文中的核心公式和架构，与 [transformer-zh-en](https://github.com/philexohf/transformer-zh-en) 项目的代码逐行对照。
> 项目实现了完整的 Encoder-Decoder Transformer，用于中英机器翻译，100 万句对训练 BLEU 达 36.87。

## 📁 项目代码结构

```
transformer-zh-en/
├── models/
│   └── transformer.py      # ← 核心！所有模型组件
├── train_2017.py            # 论文原版训练脚本（Warmup LR）
├── train_llm.py             # 改进版训练（AMP + CosineLR + AdamW）
├── config.py                # 超参数配置
├── tokenizer.py             # BPE 分词器
├── dataset.py               # 数据加载
├── infer.py                 # 推理脚本
└── evaluate_bleu.py         # BLEU 评估
```

---

## 1️⃣ Scaled Dot-Product Attention（论文公式 1）

**论文原文（Section 3.2.1）：**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**代码实现（`models/transformer.py` → `scaled_dot_product_attention`）：**

```python
def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    # QK^T / √d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))
    
    # softmax
    attention_weights = F.softmax(scores, dim=-1)
    
    # NaN 兜底（全掩码行会产生 NaN）
    attention_weights = torch.where(
        torch.isnan(attention_weights),
        torch.zeros_like(attention_weights),
        attention_weights
    )
    
    # 乘以 V
    return torch.matmul(attention_weights, value), attention_weights
```

| 论文概念 | 代码对应 | 维度 |
|---------|---------|------|
| Q (Query) | `query` | `[batch, heads, seq, d_k]` |
| K (Key) | `key` | `[batch, heads, seq, d_k]` |
| V (Value) | `value` | `[batch, heads, seq, d_v]` |
| QK^T | `torch.matmul(query, key.transpose(-2, -1))` | `[batch, heads, seq, seq]` |
| √d_k | `math.sqrt(d_k)` | 标量 |
| softmax | `F.softmax(scores, dim=-1)` | `[batch, heads, seq, seq]` |
| mask | `masked_fill(~mask, float('-inf'))` | 论文未详述，实现细节 |

**💡 细节差异：** 论文没有提及 NaN 处理，但实际实现中当整行都被 mask 掉时，`softmax([-inf, -inf, ...])` 会产生 NaN，需要特殊处理。

---

## 2️⃣ Multi-Head Attention（论文 Section 3.2.2）

**论文公式：**

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**代码实现（`MultiHeadAttention` 类）：**

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        self.d_k = d_model // num_heads      # 每个 head 的维度
        # 四个投影矩阵：W_q, W_k, W_v, W_o
        self.W_q = nn.Linear(d_model, d_model)  # 对应 W_i^Q
        self.W_k = nn.Linear(d_model, d_model)  # 对应 W_i^K
        self.W_v = nn.Linear(d_model, d_model)  # 对应 W_i^V
        self.W_o = nn.Linear(d_model, d_model)  # 对应 W^O

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 步骤1: 线性投影 + 拆分多头
        # [batch, seq, d_model] → [batch, seq, heads, d_k] → [batch, heads, seq, d_k]
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_v).transpose(1, 2)
        
        # 步骤2: 并行注意力
        x, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        
        # 步骤3: Concat + 输出投影
        # [batch, heads, seq, d_k] → [batch, seq, d_model]
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(x), attn_weights
```

| 论文概念 | 代码对应 | 说明 |
|---------|---------|------|
| $W_i^Q, W_i^K, W_i^V$ | `self.W_q, W_k, W_v` | 注意：论文是 h 组独立的小矩阵，代码用一个大矩阵再 reshape |
| $W^O$ | `self.W_o` | 输出投影 |
| $\text{Concat}(\text{head}_1, ..., \text{head}_h)$ | `.transpose(1,2).contiguous().view(...)` | 通过 reshape 实现拼接 |
| $h=8$ 头 | `num_heads=8` | |
| $d_k = d_v = d_{model}/h$ | `self.d_k = d_model // num_heads` | 384/8 = 48（项目配置）|

**💡 巧妙之处：** 代码不是真的创建 8 个独立的注意力模块，而是用一次大矩阵乘法 + reshape 来实现等价效果，更高效。

---

## 3️⃣ Positional Encoding（论文公式 3、4，Section 3.5）

**论文公式：**

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$

$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

**代码实现（`PositionalEncoding` 类）：**

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        
        # 10000^(2i/d_model) 的等价计算（用 exp + log 避免 pow 的数值问题）
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维度 → sin
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维度 → cos
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)  # 不参与梯度更新
```

| 论文概念 | 代码对应 | 说明 |
|---------|---------|------|
| pos | `position = torch.arange(0, max_len)` | 位置索引 |
| $10000^{2i/d_{model}}$ | `div_term = torch.exp(...)` | 用 exp(-log) 代替 pow，数值更稳定 |
| 偶数维度 sin | `pe[:, 0::2] = torch.sin(...)` | |
| 奇数维度 cos | `pe[:, 1::2] = torch.cos(...)` | |
| 固定编码 | `register_buffer('pe', pe)` | 不参与训练，随模型保存 |

**💡 数值技巧：** $1/10000^{2i/d} = e^{-2i \cdot \ln(10000)/d}$，用 exp+log 代替 pow 在 float32 下更精确。

---

## 4️⃣ Position-wise Feed-Forward Network（论文公式 2）

**论文公式：**

$$FFN(x) = \max(0, xW_1 + b_1)W_2 + b_2$$

**代码实现（`PositionWiseFFN` 类）：**

```python
class PositionWiseFFN(nn.Module):
    def __init__(self, d_model, d_ffn, dropout=0.1):
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ffn),   # W_1: 升维 384 → 1536
            nn.ReLU(),                    # max(0, ·)
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),   # W_2: 降维 1536 → 384
        )
```

| 论文概念 | 代码对应 | 说明 |
|---------|---------|------|
| $W_1, b_1$ | `nn.Linear(d_model, d_ffn)` | d_model → d_ffn 升维 |
| $\max(0, \cdot)$ | `nn.ReLU()` | ReLU 激活 |
| $W_2, b_2$ | `nn.Linear(d_ffn, d_model)` | d_ffn → d_model 降维 |
| $d_{ff} = 4 \times d_{model}$ | `d_ffn = 1536`（论文 2048）| 项目用 384×4=1536 |

---

## 5️⃣ Add & Norm（论文 Section 3.1，Figure 1）

**论文公式：**

$$\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

**代码实现（`AddNorm` 类）：**

```python
class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout):
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, residual, sublayer_output):
        return self.ln(residual + self.dropout(sublayer_output))
```

**💡 Post-LN vs Pre-LN：** 这个项目使用 **Post-LN**（先残差连接，再 LayerNorm），与论文原始实现一致。后来的研究发现 **Pre-LN**（先 LayerNorm，再残差连接）训练更稳定，但这是后话。

---

## 6️⃣ Encoder Layer（论文 Figure 1 左半部分）

**论文架构：**

```
Input → Self-Attention → Add&Norm → FFN → Add&Norm → Output
```

**代码实现（`EncoderLayer` 类）：**

```python
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn, dropout):
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = PositionWiseFFN(d_model, d_ffn, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, X, mask=None):
        # 子层1: Self-Attention (Q=K=V=X)
        attn_output, _ = self.self_attn(X, X, X, mask)
        X = self.add_norm1(X, attn_output)
        
        # 子层2: FFN
        ffn_output = self.ffn(X)
        X = self.add_norm2(X, ffn_output)
        
        return X
```

**关键：** Encoder 的 Self-Attention 中 Q=K=V=X（都是自身），且无因果掩码，每个 token 可以看到所有其他 token。

---

## 7️⃣ Decoder Layer（论文 Figure 1 右半部分）

**论文架构：**

```
Input → Masked Self-Attn → Add&Norm → Cross-Attn → Add&Norm → FFN → Add&Norm → Output
                              ↑              ↑
                              |         Q=decoder, K=V=encoder
                         因果掩码（防前瞻）
```

**代码实现（`DecoderLayer` 类）：**

```python
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ffn, dropout):
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)     # Masked Self-Attn
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)    # Cross-Attn
        self.ffn = PositionWiseFFN(d_model, d_ffn, dropout)
        self.add_norm1 = AddNorm(d_model, dropout)
        self.add_norm2 = AddNorm(d_model, dropout)
        self.add_norm3 = AddNorm(d_model, dropout)

    def forward(self, X, enc_output, tgt_mask=None, src_mask=None):
        # 子层1: Masked Self-Attention (Q=K=V=X, 因果掩码)
        attn_output, _ = self.self_attn(X, X, X, tgt_mask)
        X = self.add_norm1(X, attn_output)
        
        # 子层2: Cross-Attention (Q=decoder, K=V=encoder输出)
        cross_attn_output, _ = self.cross_attn(X, enc_output, enc_output, src_mask)
        X = self.add_norm2(X, cross_attn_output)
        
        # 子层3: FFN
        ffn_output = self.ffn(X)
        X = self.add_norm3(X, ffn_output)
        
        return X
```

| 子层 | Q | K | V | mask | 说明 |
|------|---|---|---|------|------|
| Masked Self-Attn | decoder | decoder | decoder | 因果掩码 | 防止看到未来 token |
| Cross-Attn | decoder | encoder | encoder | padding 掩码 | 关注源语言信息 |
| FFN | — | — | — | — | 逐位置非线性变换 |

**💡 这是 Transformer 最精妙的设计之一：** Cross-Attention 是 Encoder 和 Decoder 之间信息传递的唯一桥梁。

---

## 8️⃣ 完整数据流

**论文 Figure 1 完整架构：**

```
源语言 (中文)
    │
    ▼
┌─────────────┐
│  Embedding   │  × √d_model
│  + PE        │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Encoder ×N  │  (项目 N=4，论文 N=6)
│  Self-Attn   │
│  FFN          │
└──────┬──────┘
       │ enc_output
       │
       ├──────────────────────────┐
       │                          │
目标语言 (英文)                     │
    │                              │
    ▼                              │
┌─────────────┐                    │
│  Embedding   │  × √d_model       │
│  + PE        │                   │
└──────┬──────┘                    │
       │                           │
       ▼                           ▼
┌─────────────┐                    │
│  Decoder ×N  │  (项目 N=4)        │
│  Masked Attn │                   │
│  Cross-Attn  │◄──────────────────┘
│  FFN          │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Linear      │  d_model → vocab_size
│  Softmax     │
└──────┬──────┘
       │
       ▼
   输出概率分布
```

**代码入口（`Transformer.forward`）：**

```python
def forward(self, src, tgt):
    encoder_output, src_mask = self.encode(src)      # Encoder 全过程
    decoder_output = self.decode(tgt, encoder_output, src_mask)  # Decoder 全过程
    output = self.linear(decoder_output)             # 投影到词表
    return output
```

---

## 9️⃣ 学习率调度（论文公式 3，Section 5.3）

**论文公式：**

$$lr = d_{model}^{-0.5} \cdot \min(step^{-0.5}, \ step \cdot warmup\_steps^{-1.5})$$

**代码实现（`train_2017.py` → `get_lr`）：**

```python
def get_lr(step, d_model, warmup_steps):
    step = max(1, step)
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))
```

**调度器使用：**

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: get_lr(step, d_model, warmup_steps))
```

| 参数 | 论文值 | 项目值 | 说明 |
|------|--------|--------|------|
| base lr | — | 1.0 | LambdaLR 的 base，实际 lr 由公式决定 |
| warmup_steps | 4000 | 4000 | 预热步数 |
| betas | (0.9, 0.98) | (0.9, 0.98) | Adam 的 β₁, β₂ |
| eps | — | 1e-9 | 注意：比默认 1e-8 小一个量级 |

**💡 为什么 eps=1e-9？** 因为 lr 调度公式中 warmup 阶段 lr 从 0 开始增长，很小的梯度更新需要更高精度的分母。

---

## 🔟 超参数对照

| 参数 | 论文 Base | 项目配置 | 差异原因 |
|------|-----------|---------|---------|
| d_model | 512 | 384 | 消费级 GPU 显存优化 |
| nhead | 8 | 8 | 一致 |
| encoder layers | 6 | 4 | 小数据防过拟合 |
| decoder layers | 6 | 4 | 同上 |
| d_ff | 2048 | 1536 | 384×4，保持 4× 比例 |
| dropout | 0.1 | 0.1 | 一致 |
| warmup_steps | 4000 | 4000 | 一致 |
| 参数量 | ~65M | 53.5M | 因层数和维度减小 |

**💡 思考题：** 为什么减小层数和维度反而可能更好？在小数据上，过大的模型容易过拟合——模型把训练数据"背下来"了而不是学到泛化规律。

---

## 🧪 推荐练手实验

按照从易到难的顺序：

### Level 1：读懂代码（1-2 小时）
- [ ] 对照本文档，逐个组件阅读 `models/transformer.py`
- [ ] 在每个 `forward` 方法中加 `print(x.shape)` 追踪数据流
- [ ] 运行 `python models/transformer.py` 的验证代码

### Level 2：动手修改（2-3 小时）
- [ ] 改 `d_model` 从 384 → 512，观察参数量变化
- [ ] 改 encoder/decoder 层数从 4 → 2，对比训练速度
- [ ] 把 ReLU 换成 GELU，观察 loss 变化

### Level 3：深入理解（3-5 小时）
- [ ] 实现 Pre-LN 版本（LayerNorm 在子层之前），对比训练稳定性
- [ ] 修改 warmup_steps（1000 vs 4000 vs 8000），画学习率曲线
- [ ] 添加 attention 可视化，看模型翻译时关注哪些词

### Level 4：完整训练（需要 GPU）
- [ ] 准备数据，运行完整训练流程
- [ ] 调参实验，记录 BLEU 变化
- [ ] 实现贪心解码 → beam search 解码，对比翻译质量

---

## 🔗 链接

- **练手项目**：[philexohf/transformer-zh-en](https://github.com/philexohf/transformer-zh-en) — 纯手写 Transformer 中英翻译
- **论文原文**：[Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
- **本仓库教程**：[01-attention-is-all-you-need](../README.md)
