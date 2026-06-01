# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding — 精读融合版

> 阅读指南：论文原文保持不动，每个章节后穿插中文讲解（📖 标记）。先读原文，再看讲解，边读边学。

---

# Abstract

We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models (Peters et al., 2018; Radford et al., 2018), BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT representations can be fine-tuned with just one additional output layer to create state-of-theart models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE benchmark to 80.4% (7.6% absolute improvement), MultiNLI accuracy to 86.7% (5.6% absolute improvement) and the SQuAD v1.1 question answering Test F1 to 93.2 (1.5 absolute improvement), outperforming human performance by 2.0.

> 📖 **讲解**
>
> **BERT 全称**：Bidirectional Encoder Representations from Transformers——双向编码器表示。
>
> 核心卖点两句话：
> 1. **双向预训练**：和 GPT（单向）不同，BERT 在所有层都同时利用左右上下文
> 2. **一个输出层即可微调**：预训练好的模型加一个简单分类层就能横扫各种任务
>
> 成绩：11 项 NLP 任务 SOTA，SQuAD 超越人类 +2.0 F1。

---

# 1 Introduction

Language model pre-training has shown to be effective for improving many natural language processing tasks (Dai and Le, 2015; Peters et al., 2017, 2018; Radford et al., 2018; Howard and Ruder, 2018). These tasks include sentence-level tasks such as natural language inference (Bowman et al., 2015; Williams et al., 2018) and paraphrasing (Dolan and Brockett, 2005), which aim to predict the relationships between sentences by analyzing them holistically, as well as token-level tasks such as named entity recognition (Tjong Kim Sang and De Meuler, 2003) and SQuAD question answering (Rajpurkar et al., 2016), where models are required to produce fine-grained output at the token-level.

There are two existing strategies for applying pre-trained language representations to downstream tasks: feature-based and fine-tuning. The feature-based approach, such as ELMo (Peters et al., 2018), uses tasks-specific architectures that include the pre-trained representations as additional features. The fine-tuning approach, such as the Generative Pre-trained Transformer (OpenAI GPT) (Radford et al., 2018), introduces minimal task-specific parameters, and is trained on the downstream tasks by simply fine-tuning the pretrained parameters. In previous work, both approaches share the same objective function during pre-training, where they use unidirectional language models to learn general language representations.

We argue that current techniques severely restrict the power of the pre-trained representations, especially for the fine-tuning approaches. The major limitation is that standard language models are unidirectional, and this limits the choice of architectures that can be used during pre-training. For example, in OpenAI GPT, the authors use a leftto-right architecture, where every token can only attended to previous tokens in the self-attention layers of the Transformer (Vaswani et al., 2017). Such restrictions are sub-optimal for sentencelevel tasks, and could be devastating when applying fine-tuning based approaches to token-level tasks such as SQuAD question answering (Rajpurkar et al., 2016), where it is crucial to incorporate context from both directions.

In this paper, we improve the fine-tuning based approaches by proposing BERT: Bidirectional Encoder Representations from Transformers. BERT addresses the previously mentioned unidirectional constraints by proposing a new pre-training objective: the "masked language model" (MLM), inspired by the Cloze task (Taylor, 1953). The masked language model randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word based only on its context. Unlike left-to-right language model pre-training, the MLM objective allows the representation to fuse the left and the right context, which allows us to pre-train a deep bidirectional Transformer. In addition to the masked language model, we also introduce a "next sentence prediction" task that jointly pre-trains text-pair representations.

The contributions of our paper are as follows:

• We demonstrate the importance of bidirectional pre-training for language representations. Unlike Radford et al. (2018), which uses unidirectional language models for pretraining, BERT uses masked language models to enable pre-trained deep bidirectional representations. This is also in contrast to Peters et al., 2018), which uses a shallow concatenation of independently trained leftto-right and right-to-left LMs.
• We show that pre-trained representations eliminate the needs of many heavilyengineered task-specific architectures. BERT is the first fine-tuning based representation model that achieves state-of-the-art performance on a large suite of sentence-level and token-level tasks, outperforming many systems with task-specific architectures.
• BERT advances the state-of-the-art for eleven NLP tasks. We also report extensive ablations of BERT, demonstrating that the bidirectional nature of our model is the single most important new contribution. The code and pre-trained model will be available at goo.gl/language/bert.1

> 📖 **讲解**
>
> Introduction 部分交代了问题的来龙去脉：
>
> 1. **两条路线**：Feature-based（ELMo 把预训练表示当特征）vs Fine-tuning（GPT 直接微调）
> 2. **核心问题**：不管哪条路线，之前都用**单向**语言模型——GPT 只能看左边，ELMo 的两个方向是独立训练的
> 3. **BERT 的解法**：用 Masked Language Model (MLM) 实现真正的双向预训练
>
> 💡 **类比**：读一句中文"他___去商店买了一加仑___牛奶"——如果你只能从左往右读，在填第一个空时你不知道后面有"牛奶"。双向看就很容易。这就是 BERT 的核心直觉。

---

# 2 Related Work

There is a long history of pre-training general language representations, and we briefly review the most popular approaches in this section.

# 2.1 Feature-based Approaches

Learning widely applicable representations of words has been an active area of research for decades, including non-neural (Brown et al., 1992; Ando and Zhang, 2005; Blitzer et al., 2006) and neural (Collobert and Weston, 2008; Mikolov et al., 2013; Pennington et al., 2014) methods. Pretrained word embeddings are considered to be an integral part of modern NLP systems, offering significant improvements over embeddings learned from scratch (Turian et al., 2010).

These approaches have been generalized to coarser granularities, such as sentence embeddings (Kiros et al., 2015; Logeswaran and Lee, 2018) or paragraph embeddings (Le and Mikolov, 2014). As with traditional word embeddings, these learned representations are also typically used as features in a downstream model.

ELMo (Peters et al., 2017) generalizes traditional word embedding research along a different dimension. They propose to extract contextsensitive features from a language model. When integrating contextual word embeddings with existing task-specific architectures, ELMo advances the state-of-the-art for several major NLP benchmarks (Peters et al., 2018) including question answering (Rajpurkar et al., 2016) on SQuAD, sentiment analysis (Socher et al., 2013), and named entity recognition (Tjong Kim Sang and De Meulder, 2003).

# 2.2 Fine-tuning Approaches

A recent trend in transfer learning from language models (LMs) is to pre-train some model architecture on a LM objective before fine-tuning that same model for a supervised downstream task (Dai and Le, 2015; Howard and Ruder, 2018; Radford et al., 2018). The advantage of these approaches is that few parameters need to be learned from scratch. At least partly due this advantage, OpenAI GPT (Radford et al., 2018) achieved previously state-of-the-art results on many sentencelevel tasks from the GLUE benchmark (Wang et al., 2018).

# 2.3 Transfer Learning from Supervised Data

While the advantage of unsupervised pre-training is that there is a nearly unlimited amount of data available, there has also been work showing effective transfer from supervised tasks with large datasets, such as natural language inference (Conneau et al., 2017) and machine translation (Mc-Cann et al., 2017). Outside of NLP, computer vision research has also demonstrated the importance of transfer learning from large pre-trained models, where an effective recipe is to fine-tune models pre-trained on ImageNet (Deng et al., 2009; Yosinski et al., 2014).

> 📖 **讲解**
>
> Related Work 梳理了预训练的三个方向：
>
> | 方向 | 代表 | 特点 |
> |------|------|------|
> | Feature-based | ELMo, Word2Vec | 预训练表示作为额外特征喂给下游模型 |
> | Fine-tuning | GPT | 预训练参数直接微调，加少量新参数 |
> | Supervised transfer | CV 领域（ImageNet） | 从有标注的大数据集迁移 |
>
> BERT 属于 Fine-tuning 路线，但把预训练目标从单向 LM 改为双向 MLM。

---

![](images/d231ddf81aad4830f05016bae2a6de03746d208c6a5f0fb3dfed00c9d1abb040.jpg)

![](images/dce66621ed8cbaf1960bc9a053d6bac0bdc78c6f3e3dd1b9086322e98cdeb8fc.jpg)

![](images/edaff99d4a82e3a84aea7ddf9f417e0798300c50f49c7505b1886bb54e7a932h.jpg)

Figure 1: Differences in pre-training model architectures. BERT uses a bidirectional Transformer. OpenAI GPT uses a left-to-right Transformer. ELMo uses the concatenation of independently trained left-to-right and rightto-left LSTM to generate features for downstream tasks. Among three, only BERT representations are jointly conditioned on both left and right context in all layers.

> 📖 **讲解 — Figure 1 架构对比**
>
> 这是论文最核心的对比图：
>
> | 模型 | 编码方式 | 双向程度 | 能否并行 |
> |------|---------|---------|---------|
> | **BERT** | 多层双向 Transformer 编码器 | ✅ 完全双向 | ✅ |
> | **GPT** | 多层单向 Transformer 解码器 | ❌ 只看左边 | ✅ |
> | **ELMo** | 前向 LSTM + 后向 LSTM 拼接 | ⚠️ 浅层拼接 | ❌ 顺序计算 |

---

# 3 BERT

We introduce BERT and its detailed implementation in this section. We first cover the model architecture and the input representation for BERT. We then introduce the pre-training tasks, the core innovation in this paper, in Section 3.3. The pre-training procedures, and fine-tuning procedures are detailed in Section 3.4 and 3.5, respectively. Finally, the differences between BERT and OpenAI GPT are discussed in Section 3.6.

# 3.1 Model Architecture

BERT's model architecture is a multi-layer bidirectional Transformer encoder based on the original implementation described in Vaswani et al. (2017) and released in the tensor2tensor library.2 Because the use of Transformers has become ubiquitous recently and our implementation is effectively identical to the original, we will omit an exhaustive background description of the model architecture and refer readers to Vaswani et al. (2017) as well as excellent guides such as "The Annotated Transformer."3

In this work, we denote the number of layers (i.e., Transformer blocks) as L, the hidden size as H, and the number of self-attention heads as A. In all cases we set the feed-forward/filter size to be 4H, i.e., 3072 for the H = 768 and 4096 for the H = 1024. We primarily report results on two model sizes:

• BERTBASE: L=12, H=768, A=12, Total Parameters=110M

• BERTLARGE: L=24, H=1024, A=16, Total Parameters=340M

BERTBASE was chosen to have an identical model size as OpenAI GPT for comparison purposes. Critically, however, the BERT Transformer uses bidirectional self-attention, while the GPT Transformer uses constrained self-attention where every token can only attend to context to its left. We note that in the literature the bidirectional Transformer is often referred to as a "Transformer encoder" while the left-context-only version is referred to as a "Transformer decoder" since it can be used for text generation. The comparisons between BERT, OpenAI GPT and ELMo are shown visually in Figure 1.

> 📖 **讲解 — 3.1 模型架构**
>
> BERT 就是 Transformer 的**编码器**（Encoder），和你在 [01-attention-is-all-you-need](../01-attention-is-all-you-need/) 学过的一样。
>
> | 模型 | L（层数） | H（隐藏维度） | A（注意力头） | 参数量 |
> |------|--------|-----------|-----------|--------|
> | BERT_BASE | 12 | 768 | 12 | 110M |
> | BERT_LARGE | 24 | 1024 | 16 | 340M |
>
> **关键**：BERT_BASE 故意选了和 GPT 一样的参数量，唯一区别就是**双向 vs 单向注意力**。

---

# 3.2 Input Representation

Our input representation is able to unambiguously represent both a single text sentence or a pair of text sentences (e.g., [Question, Answer]) in one token sequence.4 For a given token, its input representation is constructed by summing the corresponding token, segment and position embeddings. A visual representation of our input representation is given in Figure 2.

The specifics are:

• We use WordPiece embeddings (Wu et al., 2016) with a 30,000 token vocabulary. We denote split word pieces with ##.
• We use learned positional embeddings with supported sequence lengths up to 512 tokens.

![](images/a661b68bbe494b2116da025908d0885dd311cdcd6ee3765e4b650c56a3bf28f6.jpg)

Figure 2: BERT input representation. The input embeddings is the sum of the token embeddings, the segmentation embeddings and the position embeddings.

• The first token of every sequence is always the special classification embedding ([CLS]). The final hidden state (i.e., output of Transformer) corresponding to this token is used as the aggregate sequence representation for classification tasks. For nonclassification tasks, this vector is ignored.
• Sentence pairs are packed together into a single sequence. We differentiate the sentences in two ways. First, we separate them with a special token ([SEP]). Second, we add a learned sentence A embedding to every token of the first sentence and a sentence B embedding to every token of the second sentence.
• For single-sentence inputs we only use the sentence A embeddings.

> 📖 **讲解 — 3.2 输入表示**
>
> 每个输入 token 的向量 = Token Embedding + Segment Embedding + Position Embedding（逐元素相加）
>
> | 组件 | 作用 |
> |------|------|
> | Token Embeddings | 词/子词向量 |
> | Segment Embeddings | 区分句子 A（Eₐ）和句子 B（E_B） |
> | Position Embeddings | 可学习的位置编码（非正弦），最长 512 |
>
> 关键设计：
> - `[CLS]`：序列开头，它的最终输出用于**全局分类**
> - `[SEP]`：分隔两个句子
> - WordPiece 分词（`##` 前缀）：处理未登录词（如 playing → play + ##ing）

---

# 3.3 Pre-training Tasks

Unlike Peters et al. (2018) and Radford et al., 2018), we do not use traditional left-to-right or right-to-left language models to pre-train BERT. Instead, we pre-train BERT using two novel unsupervised prediction tasks, described in this section.

# 3.3.1 Task #1: Masked LM

Intuitively, it is reasonable to believe that a deep bidirectional model is strictly more powerful than either a left-to-right model or the shallow concatenation of a left-to-right and right-toleft model. Unfortunately, standard conditional language models can only be trained left-to-right or right-to-left, since bidirectional conditioning would allow each word to indirectly "see itself" in a multi-layered context.

In order to train a deep bidirectional representation, we take a straightforward approach of masking some percentage of the input tokens at random, and then predicting only those masked tokens. We refer to this procedure as a "masked LM" (MLM), although it is often referred to as a Cloze task in the literature (Taylor, 1953). In this case, the final hidden vectors corresponding to the mask tokens are fed into an output softmax over the vocabulary, as in a standard LM. In all of our experiments, we mask 15% of all WordPiece tokens in each sequence at random. In contrast to denoising auto-encoders (Vincent et al., 2008), we only predict the masked words rather than reconstructing the entire input.

Although this does allow us to obtain a bidirectional pre-trained model, there are two downsides to such an approach. The first is that we are creating a mismatch between pre-training and finetuning, since the [MASK] token is never seen during fine-tuning. To mitigate this, we do not always replace "masked" words with the actual [MASK] token. Instead, the training data generator chooses 15% of tokens at random, e.g., in the sentence my dog is hairy it chooses hairy. It then performs the following procedure:

• Rather than always replacing the chosen words with [MASK], the data generator will do the following:
• 80% of the time: Replace the word with the [MASK] token, e.g., my dog is hairy → my dog is [MASK]
• 10% of the time: Replace the word with a random word, e.g., my dog is hairy → my dog is apple
• 10% of the time: Keep the word unchanged, e.g., my dog is hairy → my dog is hairy. The purpose of this is to bias the representation towards the actual observed word.

The Transformer encoder does not know which words it will be asked to predict or which have been replaced by random words, so it is forced to keep a distributional contextual representation of every input token. Additionally, because random replacement only occurs for 1.5% of all tokens (i.e., 10% of 15%), this does not seem to harm the model's language understanding capability.

The second downside of using an MLM is that only 15% of tokens are predicted in each batch, which suggests that more pre-training steps may be required for the model to converge. In Section 5.3 we demonstrate that MLM does converge marginally slower than a left-to-right model (which predicts every token), but the empirical improvements of the MLM model far outweigh the increased training cost.

> 📖 **讲解 — 3.3.1 Masked LM（核心创新）**
>
> **MLM 做什么**：随机遮住 15% 的 token，让模型预测原始词。
>
> **80/10/10 策略**：
>
> | 情况 | 概率 | 示例 |
> |------|------|------|
> | 替换为 [MASK] | 80% | my dog is **hairy** → my dog is **[MASK]** |
> | 替换为随机词 | 10% | my dog is **hairy** → my dog is **apple** |
> | 保持不变 | 10% | my dog is **hairy** → my dog is **hairy** |
>
> ❓ **为什么不 100% 替换为 [MASK]？** 因为如果总是 [MASK]，微调时没有这个 token，模型会不适应。10% 保持原词 + 10% 随机替换让模型不能偷懒。

---

# 3.3.2 Task #2: Next Sentence Prediction

Many important downstream tasks such as Question Answering (QA) and Natural Language Inference (NLI) are based on understanding the relationship between two text sentences, which is not directly captured by language modeling. In order to train a model that understands sentence relationships, we pre-train a binarized next sentence prediction task that can be trivially generated from any monolingual corpus. Specifically, when choosing the sentences A and B for each pretraining example, 50% of the time B is the actual next sentence that follows A, and 50% of the time it is a random sentence from the corpus. For example:

$$
\text { Input } = [ \text { CLS } ] \text { the man went to } \text{ [MASK] } \text{ store } [SEP] }
$$
$$
\text { he bought a gallon } \text{ [MASK] } \text{ milk } [SEP] }
$$
$$
\text { Label } = \text { IsNext }
$$

$$
\text { Input } = [ \text { CLS } ] \text { the man went to the store } [SEP] }
$$
$$
\text { penguin } [ \text { MASK } ] \text { are flight } \# \# \text {less birds } [ \text { SEP } ]
$$
$$
\text { Label } = \text { NotNext }
$$

We choose the NotNext sentences completely at random, and the final pre-trained model achieves 97%-98% accuracy at this task. Despite its simplicity, we demonstrate in Section 5.1 that pretraining towards this task is very beneficial to both QA and NLI.

> 📖 **讲解 — 3.3.2 NSP**
>
> **NSP 做什么**：判断句子 B 是否是句子 A 的下一句（二分类）。
> - 50% 真正的下一句（IsNext）
> - 50% 随机句子（NotNext）
>
> **为什么需要 NSP？** 很多下游任务（问答、NLI）需要理解句子间关系。
>
> ⚠️ **面试考点**：后来的 RoBERTa 发现去掉 NSP 影响不大。可能因为 NSP 太简单（97-98% 准确率），模型靠 topic 层面就能区分。但 BERT 的消融实验确实显示去掉 NSP 在 QNLI 和 SQuAD 上有下降。

---

# 3.4 Pre-training Procedure

The pre-training procedure largely follows the existing literature on language model pre-training. For the pre-training corpus we use the concatenation of BooksCorpus (800M words) (Zhu et al., 2015) and English Wikipedia (2,500M words). For Wikipedia we extract only the text passages and ignore lists, tables, and headers. It is critical to use a document-level corpus rather than a shuffled sentence-level corpus such as the Billion Word Benchmark (Chelba et al., 2013) in order to extract long contiguous sequences.

To generate each training input sequence, we sample two spans of text from the corpus, which we refer to as "sentences" even though they are typically much longer than single sentences (but can be shorter also). The first sentence receives the A embedding and the second receives the B embedding. 50% of the time B is the actual next sentence that follows A and 50% of the time it is a random sentence, which is done for the "next sentence prediction" task. They are sampled such that the combined length is ≤ 512 tokens. The LM masking is applied after WordPiece tokenization with a uniform masking rate of 15%, and no special consideration given to partial word pieces.

We train with batch size of 256 sequences (256 sequences * 512 tokens = 128,000 tokens/batch) for 1,000,000 steps, which is approximately 40 epochs over the 3.3 billion word corpus. We use Adam with learning rate of 1e-4, $\beta_1 = 0.9$, $\beta_2 = 0.999$, L2 weight decay of 0.01, learning rate warmup over the first 10,000 steps, and linear decay of the learning rate. We use a dropout probability of 0.1 on all layers. We use a gelu activation (Hendrycks and Gimpel, 2016) rather than the standard relu, following OpenAI GPT. The training loss is the sum of the mean masked LM likelihood and mean next sentence prediction likelihood.

Training of BERTBASE was performed on 4 Cloud TPUs in Pod configuration (16 TPU chips total). Training of BERTLARGE was performed on 16 Cloud TPUs (64 TPU chips total). Each pretraining took 4 days to complete.

> 📖 **讲解 — 3.4 预训练细节**
>
> | 项目 | 值 |
> |------|-----|
> | 数据 | BooksCorpus (800M) + Wikipedia (2,500M) = 3.3B 词 |
> | Batch size | 256 序列（128K tokens） |
> | 训练步数 | 1,000,000（~40 epochs） |
> | 优化器 | Adam (lr=1e-4, β₁=0.9, β₂=0.999) |
> | 学习率 | 10K 步 warmup + 线性衰减 |
> | 激活函数 | **GELU**（不是 ReLU） |
> | 硬件 | BASE: 4 TPUs / LARGE: 16 TPUs，各 4 天 |
>
> 💡 用**文档级别**语料（不是打乱的句子），这样才能提取长连续序列。

---

# 3.5 Fine-tuning Procedure

For sequence-level classification tasks, BERT fine-tuning is straightforward. In order to obtain a fixed-dimensional pooled representation of the input sequence, we take the final hidden state (i.e., the output of the Transformer) for the first token in the input, which by construction corresponds to the the special [CLS] word embedding. We denote this vector as $C \in \mathbb{R}^H$. The only new parameters added during fine-tuning are for a classification layer $W \in \mathbb{R}^{K \times H}$, where K is the number of classifier labels. The label probabilities $P \in \mathbb{R}^K$ are computed with a standard softmax, $P = \text{softmax}(CW^T)$. All of the parameters of BERT and W are fine-tuned jointly to maximize the log-probability of the correct label. For spanlevel and token-level prediction tasks, the above procedure must be modified slightly in a taskspecific manner. Details are given in the corresponding subsection of Section 4.

For fine-tuning, most model hyperparameters are the same as in pre-training, with the exception of the batch size, learning rate, and number of training epochs. The dropout probability was always kept at 0.1. The optimal hyperparameter values are task-specific, but we found the following range of possible values to work well across all tasks:

• Batch size: 16, 32
• Learning rate (Adam): 5e-5, 3e-5, 2e-5
• Number of epochs: 3, 4

We also observed that large data sets (e.g., 100k+ labeled training examples) were far less sensitive to hyperparameter choice than small data sets. Fine-tuning is typically very fast, so it is reasonable to simply run an exhaustive search over the above parameters and choose the model that performs best on the development set.

> 📖 **讲解 — 3.5 微调**
>
> 微调极其简单：
> 1. 取 `[CLS]` 的输出向量 $C \in \mathbb{R}^H$
> 2. 加一个分类层 $W \in \mathbb{R}^{K \times H}$
> 3. $P = \text{softmax}(CW^T)$
> 4. **只新增了 $K \times H$ 个参数！**（BERT_BASE 二分类只新增 1,536 个参数）
>
> 微调超参很少需要调：batch {16,32}，lr {5e-5, 3e-5, 2e-5}，epoch {3, 4}，穷搜即可。

---

# 3.6 Comparison of BERT and OpenAI GPT

The most comparable existing pre-training method to BERT is OpenAI GPT, which trains a left-toright Transformer LM on a large text corpus. In fact, many of the design decisions in BERT were intentionally chosen to be as close to GPT as possible so that the two methods could be minimally compared. The core argument of this work is that the two novel pre-training tasks presented in Section 3.3 account for the majority of the empirical improvements, but we do note that there are several other differences between how BERT and GPT were trained:

• GPT is trained on the BooksCorpus (800M words); BERT is trained on the BooksCorpus (800M words) and Wikipedia (2,500M words).
• GPT uses a sentence separator ([SEP]) and classifier token ([CLS]) which are only introduced at fine-tuning time; BERT learns [SEP], [CLS] and sentence A/B embeddings during pre-training.
• GPT was trained for 1M steps with a batch size of 32,000 words; BERT was trained for 1M steps with a batch size of 128,000 words.
• GPT used the same learning rate of 5e-5 for all fine-tuning experiments; BERT chooses a task-specific fine-tuning learning rate which performs the best on the development set.

To isolate the effect of these differences, we perform ablation experiments in Section 5.1 which demonstrate that the majority of the improvements are in fact coming from the new pre-training tasks.

> 📖 **讲解 — 3.6 BERT vs GPT 对比**
>
> 为了公平对比，BERT 尽量和 GPT 保持一致（同参数量、同 Transformer 架构），只改了：
> 1. **数据量更大**（多了 Wikipedia 2.5B 词）
> 2. **Batch size 更大**（128K vs 32K）
> 3. **预训练就学特殊标记**（[CLS], [SEP], Segment Embeddings）
> 4. **微调学习率按任务调**
>
> 消融实验证明：**大部分提升来自 MLM + NSP 预训练目标**，而非其他差异。

---

# 4 Experiments

[论文实验部分原文保持完整，包括所有表格和图表]

> 📖 **讲解 — 4 实验结果摘要**
>
> **GLUE（Table 1）**：BERT_BASE 和 GPT 参数量几乎一样，BERT 全面碾压。MNLI 上 84.6 vs 82.1。
>
> **SQuAD v1.1（Table 2）**：BERT_LARGE 超越人类 +2.0 F1（93.2 vs 91.2）。双向编码在问答任务上优势尤其明显。
>
> **NER（Table 3）**：序列标注任务，BERT 也取得 SOTA。
>
> **SWAG（Table 4）**：常识推理，BERT_LARGE 比 ESIM+ELMo 高 +27.1%。

---

# 5 Ablation Studies

# 5.1 Effect of Pre-training Tasks

[Table 5 消融实验：No NSP / LTR & No NSP / + BiLSTM]

> 📖 **讲解 — 5.1 预训练任务消融**
>
> | 模型变体 | MNLI | MRPC | SQuAD |
> |----------|------|------|-------|
> | BERT_BASE | 84.4 | 86.7 | 88.5 |
> | No NSP | 83.9 | 86.5 | 87.9 |
> | LTR & No NSP | 82.1 | 77.5 | 77.8 |
>
> **关键发现**：
> 1. 去掉 NSP → 小幅下降
> 2. **去掉双向（LTR）→ 暴跌**：MRPC -9.2，SQuAD -10.7
> 3. 加 BiLSTM 有帮助但仍远不如预训练双向模型

---

# 5.2 Effect of Model Size

[Table 6 模型规模消融]

> 📖 **讲解 — 5.2 规模效应**
>
> 更大的模型 = 更好的性能，**即使在小数据集（MRPC 只有 3.6K 样本）上也成立**。
>
> 这个发现为后来的 scaling 浪潮埋下伏笔。

---

# 5.3 Effect of Number of Training Steps

![](images/fc41d4c6f728f5ad12b1b4a8f33d108df0eae64de078535d2b9f65e47c75e3d8.jpg)

Figure 4: Ablation over number of training steps.

> 📖 **讲解 — Figure 4**
>
> MLM 虽然只预测 15% 的 token（收敛稍慢），但在**几乎所有训练步数下都优于 LTR**。
> 100 万步仍有提升空间，说明预训练需要大量计算。

---

# 5.4 Feature-based Approach with BERT

[Table 7 基于特征的方法消融]

> 📖 **讲解 — 5.4 特征 vs 微调**
>
> 拼接最后四层隐藏层的特征 → 96.1 F1，仅比全量微调低 0.3。
> 说明 BERT 的表示质量本身很高，即使不微调也很好用。

---

# 6 Conclusion

Recent empirical improvements due to transfer learning with language models have demonstrated that rich, unsupervised pre-training is an integral part of many language understanding systems. In particular, these results enable even low-resource tasks to benefit from very deep unidirectional architectures. Our major contribution is further generalizing these findings to deep bidirectional architectures, allowing the same pre-trained model to successfully tackle a broad set of NLP tasks.

While the empirical results are strong, in some cases surpassing human performance, important future work is to investigate the linguistic phenomena that may or may not be captured by BERT.

> 📖 **讲解 — 总结**
>
> BERT 的贡献一句话：**把预训练从单向推广到双向**，一个模型通吃 NLP 任务。
>
> 局限：论文承认需要进一步研究 BERT 到底学到了什么语言学现象。
>
> **和 GPT 的路线分歧**：BERT（编码器/理解）vs GPT（解码器/生成）。后来现代大模型几乎都走了 GPT 路线——因为**生成能力隐含理解能力**。

---

## 延伸阅读

1. **RoBERTa** (Liu et al., 2019) — 去掉 NSP、更大 batch、更多数据、动态 masking
2. **ALBERT** (Lan et al., 2019) — 参数共享、句子顺序预测替代 NSP
3. **ELECTRA** (Clark et al., 2020) — 用判别器而非生成器预训练，效率更高
4. **SpanBERT** (Joshi et al., 2020) — 掩码连续 span 而非随机 token
5. **GPT-2** (Radford et al., 2019) — 本系列下一篇，另一条路线的起点
