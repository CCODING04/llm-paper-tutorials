# BERT 论文图表分析

## 图 1：BERT vs GPT vs ELMo 架构对比（Figure 1）

论文最核心的架构对比图，展示了三种预训练模型的设计差异：

### BERT（本文提出）
- **架构**：多层双向 Transformer 编码器（Trm 蓝色椭圆）
- **连接**：每个 Trm 层的每个位置节点与上下所有位置的 Trm 节点双向连接
- **核心**：完全双向的注意力机制，每个位置同时利用前后文
- **输入/输出**：E₁-Eₙ → 双向 Trm → T₁-Tₙ

### OpenAI GPT
- **架构**：多层单向 Transformer
- **连接**：Trm 节点仅指向左侧前序位置（箭头仅向左）
- **局限**：每个位置只能获取前文信息，无法感知后文
- **对比**：和 BERT 的核心差异——单向 vs 双向

### ELMo
- **架构**：前向 LSTM + 后向 LSTM（两个独立分支）
- **连接**：前向 LSTM 从左到右，后向 LSTM 从右到左
- **局限**：双向是"两个独立单向 LSTM 的拼接"，非完全双向注意力
- **对比**：LSTM 无法并行，上下文建模深度弱于 Transformer

---

## 图 2：BERT 输入表示（Figure 2）

展示 BERT 输入向量的构建方式。

### 输入示例
`[CLS] my dog is cute [SEP] he likes play ##ing [SEP]`

### 三层嵌入（逐元素相加）
1. **Token Embeddings**：词/子词向量，包括特殊标记 E[CLS]、E[SEP]，子词 E[##ing]
2. **Segment Embeddings**：区分两个句子，第一句用 Eₐ，第二句用 E_B
3. **Position Embeddings**：可学习的位置向量 E₀-E₁₀

### 关键设计
- `[CLS]` 标记：聚合全局信息，用于句子级分类任务
- `[SEP]` 标记：分隔句子对
- WordPiece 分词（`##` 前缀）：处理 OOV 词
- 三个嵌入维度统一（BERT-base 为 768 维），可直接相加

---

## 图 3：BERT 用于抽取式问答（SQuAD）

展示 BERT 微调到问答任务的方式：
- **输入**：Question + [SEP] + Paragraph 拼接
- **输出**：通过 Start/End Span 预测头定位答案的起始和结束位置
- **本质**：将问答转化为 token 级别的起始/结束位置分类任务
- **无需复杂架构**：只需加一个简单输出层

---

## 图 4：BERT 用于序列标注（NER）

展示 BERT 微调到序列标注任务的方式：
- **输入**：单句 + [CLS]
- **输出**：每个 token 连接分类头，预测 BIO 标签（如 O、B-PER）
- **优势**：双向编码天然适配需要前后文信息的标注任务

---

## 图 5：预训练目标对比（MNLI 任务）

折线图，对比 Masked LM vs Left-to-Right 在 MNLI 上的准确率：
- **横轴**：预训练步数（0~100 万步）
- **纵轴**：MNLI 验证集准确率（76~84）
- **结论**：Masked LM（双向）在所有步数下都优于 Left-to-Right（单向）
- **最终差距**：~84.5 vs ~82.2，双向预训练优势明显
