# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

Jacob Devlin Ming-Wei Chang Kenton Lee Kristina Toutanova

Google AI Language

{jacobdevlin,mingweichang,kentonl,kristout}@google.com

# Abstract

We introduce a new language representation model called BERT, which stands for Bidirectional Encoder Representations from Transformers. Unlike recent language representation models (Peters et al., 2018; Radford et al., 2018), BERT is designed to pre-train deep bidirectional representations by jointly conditioning on both left and right context in all layers. As a result, the pre-trained BERT representations can be fine-tuned with just one additional output layer to create state-of-theart models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications.

BERT is conceptually simple and empirically powerful. It obtains new state-of-the-art results on eleven natural language processing tasks, including pushing the GLUE benchmark to 80.4% (7.6% absolute improvement), MultiNLI accuracy to 86.7% (5.6% absolute improvement) and the SQuAD v1.1 question answering Test F1 to 93.2 (1.5 absolute improvement), outperforming human performance by 2.0.

# 1 Introduction

Language model pre-training has shown to be effective for improving many natural language processing tasks (Dai and Le, 2015; Peters et al., 2017, 2018; Radford et al., 2018; Howard and Ruder, 2018). These tasks include sentence-level tasks such as natural language inference (Bowman et al., 2015; Williams et al., 2018) and paraphrasing (Dolan and Brockett, 2005), which aim to predict the relationships between sentences by analyzing them holistically, as well as token-level tasks such as named entity recognition (Tjong Kim Sang and De Meulder, 2003) and SQuAD question answering (Rajpurkar et al., 2016), where models are required to produce fine-grained output at the token-level.

There are two existing strategies for applying pre-trained language representations to downstream tasks: feature-based and fine-tuning. The feature-based approach, such as ELMo (Peters et al., 2018), uses tasks-specific architectures that include the pre-trained representations as additional features. The fine-tuning approach, such as the Generative Pre-trained Transformer (OpenAI GPT) (Radford et al., 2018), introduces minimal task-specific parameters, and is trained on the downstream tasks by simply fine-tuning the pretrained parameters. In previous work, both approaches share the same objective function during pre-training, where they use unidirectional language models to learn general language representations.

We argue that current techniques severely restrict the power of the pre-trained representations, especially for the fine-tuning approaches. The major limitation is that standard language models are unidirectional, and this limits the choice of architectures that can be used during pre-training. For example, in OpenAI GPT, the authors use a leftto-right architecture, where every token can only attended to previous tokens in the self-attention layers of the Transformer (Vaswani et al., 2017). Such restrictions are sub-optimal for sentencelevel tasks, and could be devastating when applying fine-tuning based approaches to token-level tasks such as SQuAD question answering (Rajpurkar et al., 2016), where it is crucial to incorporate context from both directions.

In this paper, we improve the fine-tuning based approaches by proposing BERT: Bidirectional Encoder Representations from Transformers. BERT addresses the previously mentioned unidirectional constraints by proposing a new pre-training objective: the “masked language model” (MLM), inspired by the Cloze task (Taylor, 1953). The masked language model randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word based only on its context. Unlike left-to-right language model pre-training, the MLM objective allows the representation to fuse the left and the right context, which allows us to pre-train a deep bidirectional Transformer. In addition to the masked language model, we also introduce a “next sentence prediction” task that jointly pre-trains text-pair representations.

The contributions of our paper are as follows:

• We demonstrate the importance of bidirectional pre-training for language representations. Unlike Radford et al. (2018), which uses unidirectional language models for pretraining, BERT uses masked language models to enable pre-trained deep bidirectional representations. This is also in contrast to Peters et al. (2018), which uses a shallow concatenation of independently trained leftto-right and right-to-left LMs.   
• We show that pre-trained representations eliminate the needs of many heavilyengineered task-specific architectures. BERT is the first fine-tuning based representation model that achieves state-of-the-art performance on a large suite of sentence-level and token-level tasks, outperforming many systems with task-specific architectures.   
• BERT advances the state-of-the-art for eleven NLP tasks. We also report extensive ablations of BERT, demonstrating that the bidirectional nature of our model is the single most important new contribution. The code and pre-trained model will be available at goo.gl/language/bert.1

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

![](images/d231ddf81aad4830f05016bae2a6de03746d208c6a5f0fb3dfed00c9d1abb040.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph BERT(Ours)
        T1["T₁"] --> Trm1["Trm"]
        T2["T₂"] --> Trm1
        TN["Tₙ"] --> Trm2["Trm"]
        Trm1 --> Trm3["Trm"]
        Trm2 --> Trm3
        Trm3 --> Trm4["Trm"]
        Trm1 --> E1["E₁"]
        Trm2 --> E2["E₂"]
        Trm3 --> EN["Eₙ"]
    end
    style BERT(Ours) fill:#f9f9f9,stroke:#333
    style E1 fill:#ffcccc,stroke:#333
    style E2 fill:#ffcccc,stroke:#333
    style EN fill:#ffcccc,stroke:#333
```
</details>

![](images/dce66621ed8cbaf1960bc9a053d6bac0bdc78c6f3e3dd1b9086322e98cdeb8fc.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    T1["T₁"] --> Trm1["Trm"]
    T2["T₂"] --> Trm2["Trm"]
    TN["Tₙ"] --> TrmN["Trm"]
    Trm1 --> E1["E₁"]
    Trm2 --> E2["E₂"]
    TrmN --> EN["EN"]
    E1 --> Trm3["Trm"]
    E2 --> Trm4["Trm"]
    E1 --> Trm5["Trm"]
    E2 --> Trm6["Trm"]
    Trm3 --> E1
    Trm4 --> E2
    Trm5 --> E1
    Trm6 --> E2
    Trm3 --> Trm7["Trm"]
    Trm4 --> Trm8["Trm"]
    Trm5 --> Trm9["Trm"]
    Trm6 --> Trm10["Trm"]
    Trm7 --> E1
    Trm8 --> E2
    Trm9 --> E1
    Trm10 --> E2
```
</details>

![](images/edaff99d4a82e3a84aea7ddf9f417e0798300c50f49c7505b1886bb54e7a932f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph ELMo
        T1 --> Lstm1["Lstm"]
        T1 --> Lstm2["Lstm"]
        T1 --> Lstm3["Lstm"]
        T1 --> Lstm4["Lstm"]
        T1 --> Lstm5["Lstm"]
        T1 --> Lstm6["Lstm"]
        T1 --> Lstm7["Lstm"]
        T1 --> Lstm8["Lstm"]
        T1 --> Lstm9["Lstm"]
        T2 --> Lstm1
        T2 --> Lstm2
        T2 --> Lstm3
        T2 --> Lstm4
        T2 --> Lstm5
        T2 --> Lstm6
        T2 --> Lstm7
        T2 --> Lstm8
        T2 --> Lstm9
        T3 --> Lstm1
        T3 --> Lstm2
        T3 --> Lstm3
        T3 --> Lstm4
        T3 --> Lstm5
        T3 --> Lstm6
        T3 --> Lstm7
        T3 --> Lstm8
        T3 --> Lstm9
    end

    subgraph ELMo
        Lstm1 <--> Lstm2
        Lstm2 <--> Lstm3
        Lstm3 <--> Lstm4
        Lstm4 <--> Lstm5
        Lstm5 <--> Lstm6
        Lstm6 <--> Lstm7
        Lstm7 <--> Lstm8
        Lstm8 <--> Lstm9
    end

    subgraph ELMo
        E1 --> Lstm1
        E1 --> Lstm2
        E1 --> Lstm3
        E1 --> Lstm4
        E1 --> Lstm5
        E1 --> Lstm6
        E1 --> E1_...
        E2 --> Lstm1
        E2 --> Lstm2
        E2 --> Lstm3
        E2 --> Lstm4
        E2 --> Lstm5
        E2 --> Lstm6
        E2 --> E1_...
        E3 --> Lstm1
        E3 --> Lstm2
        E3 --> Lstm3
        E3 --> Lstm4
        E3 --> Lstm5
        E3 --> Lstm6
        E3 --> E1_...
        E4 --> Lstm1
        E4 --> Lstm2
        E4 --> Lstm3
        E4 --> Lstm4
        E4 --> Lstm5
        E4 --> Lstm6
        E4 --> E1_...
        E5 --> Lstm1
        E5 --> Lstm2
        E5 --> Lstm3
        E5 --> Lstm4
        E5 --> Lstm5
        E5 --> Lstm6
        E5 --> EN_...
    end

    style ELMo fill:#f9f,stroke:#333,stroke-width:2px
```
</details>

Figure 1: Differences in pre-training model architectures. BERT uses a bidirectional Transformer. OpenAI GPT uses a left-to-right Transformer. ELMo uses the concatenation of independently trained left-to-right and rightto-left LSTM to generate features for downstream tasks. Among three, only BERT representations are jointly conditioned on both left and right context in all layers.

# 3 BERT

We introduce BERT and its detailed implementation in this section. We first cover the model architecture and the input representation for BERT. We then introduce the pre-training tasks, the core innovation in this paper, in Section 3.3. The pre-training procedures, and fine-tuning procedures are detailed in Section 3.4 and 3.5, respectively. Finally, the differences between BERT and OpenAI GPT are discussed in Section 3.6.

# 3.1 Model Architecture

BERT’s model architecture is a multi-layer bidirectional Transformer encoder based on the original implementation described in Vaswani et al. (2017) and released in the tensor2tensor library.2 Because the use of Transformers has become ubiquitous recently and our implementation is effectively identical to the original, we will omit an exhaustive background description of the model architecture and refer readers to Vaswani et al. (2017) as well as excellent guides such as “The Annotated Transformer.”3

In this work, we denote the number of layers (i.e., Transformer blocks) as L, the hidden size as H, and the number of self-attention heads as A. In all cases we set the feed-forward/filter size to be 4H, i.e., 3072 for the H = 768 and 4096 for the H = 1024. We primarily report results on two model sizes:

• BERTBASE: L=12, H=768, A=12, Total Parameters=110M

• BERTLARGE: L=24, H=1024, A=16, Total Parameters=340M

BERTBASE was chosen to have an identical model size as OpenAI GPT for comparison purposes. Critically, however, the BERT Transformer uses bidirectional self-attention, while the GPT Transformer uses constrained self-attention where every token can only attend to context to its left. We note that in the literature the bidirectional Transformer is often referred to as a “Transformer encoder” while the left-context-only version is referred to as a “Transformer decoder” since it can be used for text generation. The comparisons between BERT, OpenAI GPT and ELMo are shown visually in Figure 1.

# 3.2 Input Representation

Our input representation is able to unambiguously represent both a single text sentence or a pair of text sentences (e.g., [Question, Answer]) in one token sequence.4 For a given token, its input representation is constructed by summing the corresponding token, segment and position embeddings. A visual representation of our input representation is given in Figure 2.

The specifics are:

• We use WordPiece embeddings (Wu et al., 2016) with a 30,000 token vocabulary. We denote split word pieces with ##.   
• We use learned positional embeddings with supported sequence lengths up to 512 tokens.

![](images/a661b68bbe494b2116da025908d0885dd311cdcd6ee3765e4b650c56a3bf28f6.jpg)

<details>
<summary>other</summary>

| Input | [CLS] | my | dog | is | cute | [SEP] | he | likes | play | ##ing | [SEP] |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Token Embeddings | E[CLS] | Emy | Edog | Eis | Ecute | E[SEP] | He | Likes | Eplay | E##ing | E[SEP] |
| Segment Embeddings | EA | EA | EA | EA | EA | EA | EB | EB | EB | EB | EB |
| Position Embeddings | E0 | E1 | E2 | E3 | E4 | E5 | E6 | E7 | E8 | E9 | E10 |
</details>

Figure 2: BERT input representation. The input embeddings is the sum of the token embeddings, the segmentation embeddings and the position embeddings.

• The first token of every sequence is always the special classification embedding ([CLS]). The final hidden state (i.e., output of Transformer) corresponding to this token is used as the aggregate sequence representation for classification tasks. For nonclassification tasks, this vector is ignored.   
• Sentence pairs are packed together into a single sequence. We differentiate the sentences in two ways. First, we separate them with a special token ([SEP]). Second, we add a learned sentence A embedding to every token of the first sentence and a sentence B embedding to every token of the second sentence.   
• For single-sentence inputs we only use the sentence A embeddings.

# 3.3 Pre-training Tasks

Unlike Peters et al. (2018) and Radford et al. (2018), we do not use traditional left-to-right or right-to-left language models to pre-train BERT. Instead, we pre-train BERT using two novel unsupervised prediction tasks, described in this section.

# 3.3.1 Task #1: Masked LM

Intuitively, it is reasonable to believe that a deep bidirectional model is strictly more powerful than either a left-to-right model or the shallow concatenation of a left-to-right and right-toleft model. Unfortunately, standard conditional language models can only be trained left-to-right or right-to-left, since bidirectional conditioning would allow each word to indirectly “see itself” in a multi-layered context.

In order to train a deep bidirectional representation, we take a straightforward approach of masking some percentage of the input tokens at random, and then predicting only those masked tokens. We refer to this procedure as a “masked LM” (MLM), although it is often referred to as a Cloze task in the literature (Taylor, 1953). In this case, the final hidden vectors corresponding to the mask tokens are fed into an output softmax over the vocabulary, as in a standard LM. In all of our experiments, we mask 15% of all WordPiece tokens in each sequence at random. In contrast to denoising auto-encoders (Vincent et al., 2008), we only predict the masked words rather than reconstructing the entire input.

Although this does allow us to obtain a bidirectional pre-trained model, there are two downsides to such an approach. The first is that we are creating a mismatch between pre-training and finetuning, since the [MASK] token is never seen during fine-tuning. To mitigate this, we do not always replace “masked” words with the actual [MASK] token. Instead, the training data generator chooses 15% of tokens at random, e.g., in the sentence my dog is hairy it chooses hairy. It then performs the following procedure:

• Rather than always replacing the chosen words with [MASK], the data generator will do the following:   
• 80% of the time: Replace the word with the [MASK] token, e.g., my dog is hairy → my dog is [MASK]   
• 10% of the time: Replace the word with a random word, e.g., my dog is hairy → my dog is apple   
• 10% of the time: Keep the word unchanged, e.g., my dog is hairy → my dog is hairy. The purpose of this is to bias the representation towards the actual observed word.

The Transformer encoder does not know which words it will be asked to predict or which have been replaced by random words, so it is forced to keep a distributional contextual representation of every input token. Additionally, because random replacement only occurs for 1.5% of all tokens (i.e., 10% of 15%), this does not seem to harm the model’s language understanding capability.

The second downside of using an MLM is that only 15% of tokens are predicted in each batch, which suggests that more pre-training steps may be required for the model to converge. In Section 5.3 we demonstrate that MLM does converge marginally slower than a left-to-right model (which predicts every token), but the empirical improvements of the MLM model far outweigh the increased training cost.

# 3.3.2 Task #2: Next Sentence Prediction

Many important downstream tasks such as Question Answering (QA) and Natural Language Inference (NLI) are based on understanding the relationship between two text sentences, which is not directly captured by language modeling. In order to train a model that understands sentence relationships, we pre-train a binarized next sentence prediction task that can be trivially generated from any monolingual corpus. Specifically, when choosing the sentences A and B for each pretraining example, 50% of the time B is the actual next sentence that follows A, and 50% of the time it is a random sentence from the corpus. For example:

$$
\text { Input } = [ \text { CLS } ] \text {   the   man   went   to   [MASK]   store   [SEP] }
$$

$$
\text {   he   bought   a   gallon   [MASK]   milk   [SEP]   }
$$

$$
\text { Label } = \text {   IsNext   }
$$

$$
\text { Input } = [ \text { CLS } ] \text {   the   man   [MASK]   to   the   store   [SEP] }
$$

$$
\text { penguin } [ \text { MASK } ] \text { are   flight   \#   \#less   birds } [ \text { SEP } ]
$$

$$
\text { Label } = \text {   NotNext   }
$$

We choose the NotNext sentences completely at random, and the final pre-trained model achieves 97%-98% accuracy at this task. Despite its simplicity, we demonstrate in Section 5.1 that pretraining towards this task is very beneficial to both QA and NLI.

# 3.4 Pre-training Procedure

The pre-training procedure largely follows the existing literature on language model pre-training. For the pre-training corpus we use the concatenation of BooksCorpus (800M words) (Zhu et al., 2015) and English Wikipedia (2,500M words). For Wikipedia we extract only the text passages and ignore lists, tables, and headers. It is critical to use a document-level corpus rather than a shuffled sentence-level corpus such as the Billion Word Benchmark (Chelba et al., 2013) in order to extract long contiguous sequences.

To generate each training input sequence, we sample two spans of text from the corpus, which we refer to as “sentences” even though they are typically much longer than single sentences (but can be shorter also). The first sentence receives the A embedding and the second receives the B embedding. 50% of the time B is the actual next sentence that follows A and 50% of the time it is a random sentence, which is done for the “next sentence prediction” task. They are sampled such that the combined length is ≤ 512 tokens. The LM masking is applied after WordPiece tokenization with a uniform masking rate of 15%, and no special consideration given to partial word pieces.

We train with batch size of 256 sequences (256 sequences \* 512 tokens = 128,000 tokens/batch) for 1,000,000 steps, which is approximately 40 epochs over the 3.3 billion word corpus. We use Adam with learning rate of 1e-4, $\beta _ { 1 } = 0 . 9 ,$ , $\beta _ { 2 } ~ = ~ 0 . 9 9 9$ , L2 weight decay of 0.01, learning rate warmup over the first 10,000 steps, and linear decay of the learning rate. We use a dropout probability of 0.1 on all layers. We use a gelu activation (Hendrycks and Gimpel, 2016) rather than the standard relu, following OpenAI GPT. The training loss is the sum of the mean masked LM likelihood and mean next sentence prediction likelihood.

Training of BERTBASE was performed on 4 Cloud TPUs in Pod configuration (16 TPU chips total).5 Training of BERTLARGE was performed on 16 Cloud TPUs (64 TPU chips total). Each pretraining took 4 days to complete.

# 3.5 Fine-tuning Procedure

For sequence-level classification tasks, BERT fine-tuning is straightforward. In order to obtain a fixed-dimensional pooled representation of the input sequence, we take the final hidden state (i.e., the output of the Transformer) for the first token in the input, which by construction corresponds to the the special [CLS] word embedding. We denote this vector as $C \in \mathbb { R } ^ { H }$ . The only new parameters added during fine-tuning are for a classification layer $W \in \mathbb { R } ^ { K \times H }$ , where K is the number of classifier labels. The label probabilities $P \in \mathbb { R } ^ { K }$ are computed with a standard softmax, $P = \mathrm { s o f t m a x } ( C W ^ { T } )$ . All of the parameters of BERT and W are fine-tuned jointly to maximize the log-probability of the correct label. For spanlevel and token-level prediction tasks, the above procedure must be modified slightly in a taskspecific manner. Details are given in the corresponding subsection of Section 4.

For fine-tuning, most model hyperparameters are the same as in pre-training, with the exception of the batch size, learning rate, and number of training epochs. The dropout probability was always kept at 0.1. The optimal hyperparameter values are task-specific, but we found the following range of possible values to work well across all tasks:

• Batch size: 16, 32   
• Learning rate (Adam): 5e-5, 3e-5, 2e-5   
• Number of epochs: 3, 4

We also observed that large data sets (e.g., 100k+ labeled training examples) were far less sensitive to hyperparameter choice than small data sets. Fine-tuning is typically very fast, so it is reasonable to simply run an exhaustive search over the above parameters and choose the model that performs best on the development set.

# 3.6 Comparison of BERT and OpenAI GPT

The most comparable existing pre-training method to BERT is OpenAI GPT, which trains a left-toright Transformer LM on a large text corpus. In fact, many of the design decisions in BERT were intentionally chosen to be as close to GPT as possible so that the two methods could be minimally compared. The core argument of this work is that the two novel pre-training tasks presented in Section 3.3 account for the majority of the empirical improvements, but we do note that there are several other differences between how BERT and GPT were trained:

• GPT is trained on the BooksCorpus (800M words); BERT is trained on the BooksCorpus (800M words) and Wikipedia (2,500M words).

• GPT uses a sentence separator ([SEP]) and classifier token ([CLS]) which are only introduced at fine-tuning time; BERT learns [SEP], [CLS] and sentence A/B embeddings during pre-training.   
• GPT was trained for 1M steps with a batch size of 32,000 words; BERT was trained for 1M steps with a batch size of 128,000 words.   
• GPT used the same learning rate of 5e-5 for all fine-tuning experiments; BERT chooses a task-specific fine-tuning learning rate which performs the best on the development set.

To isolate the effect of these differences, we perform ablation experiments in Section 5.1 which demonstrate that the majority of the improvements are in fact coming from the new pre-training tasks.

# 4 Experiments

In this section, we present BERT fine-tuning results on 11 NLP tasks.

# 4.1 GLUE Datasets

The General Language Understanding Evaluation (GLUE) benchmark (Wang et al., 2018) is a collection of diverse natural language understanding tasks. Most of the GLUE datasets have already existed for a number of years, but the purpose of GLUE is to (1) distribute these datasets with canonical Train, Dev, and Test splits, and (2) set up an evaluation server to mitigate issues with evaluation inconsistencies and Test set overfitting. GLUE does not distribute labels for the Test set and users must upload their predictions to the GLUE server for evaluation, with limits on the number of submissions.

The GLUE benchmark includes the following datasets, the descriptions of which were originally summarized in Wang et al. (2018):

MNLI Multi-Genre Natural Language Inference is a large-scale, crowdsourced entailment classification task (Williams et al., 2018). Given a pair of sentences, the goal is to predict whether the second sentence is an entailment, contradiction, or neutral with respect to the first one.

QQP Quora Question Pairs is a binary classification task where the goal is to determine if two questions asked on Quora are semantically equivalent (Chen et al., 2018).

![](images/0e7c6d07549924b19b1e02aac7e8012451d65929d2a72d773d5d139de8355d5a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Class Label"] --> B["C"]
    A --> C["T₁"]
    A --> D["..."]
    A --> E["Tₙ"]
    A --> F["T_[SEP"]]
    A --> G["T₁'"]
    A --> H["Tₘ'"]
    B --> I["..."]
    C --> J["E_CLS"]
    C --> K["E₁"]
    C --> L["..."]
    D --> M["E_N"]
    D --> N["E_[SEP"]]
    D --> O["E₁'"]
    D --> P["E_M'"]
    E --> Q["[CLS"]]
    F --> R["Tok 1"]
    G --> S["..."]
    H --> T["Tok N"]
    I --> U["[SEP"]]
    J --> V["Tok 1"]
    K --> W["..."]
    L --> X["Tok M"]
    M --> Y["Sentence 1"]
    N --> Z["Sentence 2"]
    O --> AA["Sentence 2"]
    P --> AB["Sentence 2"]
```
</details>

(a) Sentence Pair Classification Tasks: MNLI, QQP,QNLI, STS-B, MRPC, RTE,SWAG

![](images/6a6a562cf173c8626a67123893600d3256f7bd0994f388a5b5b263529742200b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Class Label"] --> B["C"]
    A --> C["T₁"]
    A --> D["T₂"]
    A --> E["..."]
    A --> F["Tₙ"]
    B --> G["BERT"]
    C --> G
    D --> G
    E --> G
    F --> G
    G --> H["E〔CLS〕"]
    G --> I["E₁"]
    G --> J["E₂"]
    G --> K["..."]
    G --> L["Eₙ"]
    H --> M["[CLS"]]
    I --> N["Tok 1"]
    J --> O["Tok 2"]
    K --> P["..."]
    L --> Q["Tok N"]
    M --> R["Single Sentence"]
    N --> R
    O --> R
    P --> R
    Q --> R
```
</details>

(b) Single Sentence Classification Tasks: SST-2,CoLA

![](images/f4029b3d51d098bd876c0d9aa50eb6520d53aa2cea98c55eae00626b4a9c129d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Question
        C["C"] --> T1["T₁"]
        T1 --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ...
    end
    subgraph Paragraph
        E["EN"] --> E1["E₁"]
        E1 --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ...
    end
    BERT["BERT"] --> TN["Tₙ"]
    BERT --> T["M' T[SEP"]]
    BERT --> TM["T₁'"]
    BERT --> TM'
    BERT --> TM'
    BERT --> E["ELCS"]
    BERT --> EN
    BERT --> EM
    BERT --> E1
    BERT --> EN
    BERT --> EM
    BERT --> E1'
    BERT --> EM'
    BERT --> EM'
    BERT --> E1
    BERT --> EM'
    BERT --> E1'
    BERT --> EM'
    BERT --> E1
    BERT --> EM'
    BERT --> E1'
    BERT --> EM'
    BERT --> E1
    BERT --> EM'
    BERT --> E1'
    BERT --> EM'
    BERT --> EM'
    BERT --> E1
    BERT --> EM'
    BERT --> EM'
    BERT --> E1
    BERT --> EM'
    BERT --> EM'
    BERT --> E1
    BERT --> EM'
    BERT --> EM'
    BERT --> E1
    BERT --> EM'
    BERT --> EM'
    BERT --> E1
    BERT --> EM'
    BERT --> EM'
    BERT --> E1
    BERT --> Em
    BERT --> Em'
    BERT --> Em'
    BERT --> Em'
    BERT --> Em'
    BERT --> Em'
    BERT --> Em'
    BERT --> Em'
    BERT --> Em'
    BERT --> Em'
```
</details>

(c) Question Answering Tasks: SQuAD v1.1

![](images/2d85e469b51c53a8550cdb595ceb0c22fb416191c576968638db7a3a73f88e1a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph BERT
        C --> T1
        T1 --> T2
        T2 --> ... --> TN
        ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... --> ... 
    end

    subgraph Single Sentence
        E["CLS"] --> E1
        E1 --> E2
        E2 --> EN
        [CLS] --> E1
        Tok1 --> E2
        Tok2 --> EN
        TokN --> EN
    end

    style BERT fill:#f9f9f9,stroke:#333,stroke-width:2px
    style Single Sentence fill:#ffcccc,stroke:#333,stroke-width:2px
```
</details>

(d) Single Sentence Tagging Tasks: CoNLL-2003 NER   
Figure 3: Our task specific models are formed by incorporating BERT with one additional output layer, so a minimal number of parameters need to be learned from scratch. Among the tasks, (a) and (b) are sequence-level tasks while (c) and (d) are token-level tasks. In the figure, E represents the input embedding, Ti represents the contextual representation of token i, [CLS] is the special symbol for classification output, and [SEP] is the special symbol to separate non-consecutive token sequences.

QNLI Question Natural Language Inference is a version of the Stanford Question Answering Dataset (Rajpurkar et al., 2016) which has been converted to a binary classification task (Wang et al., 2018). The positive examples are (question, sentence) pairs which do contain the correct answer, and the negative examples are (question, sentence) from the same paragraph which do not contain the answer.

SST-2 The Stanford Sentiment Treebank is a binary single-sentence classification task consisting of sentences extracted from movie reviews with human annotations of their sentiment (Socher et al., 2013).

CoLA The Corpus of Linguistic Acceptability is a binary single-sentence classification task, where the goal is to predict whether an English sentence is linguistically “acceptable” or not (Warstadt et al., 2018).

STS-B The Semantic Textual Similarity Benchmark is a collection of sentence pairs drawn from news headlines and other sources (Cer et al., 2017). They were annotated with a score from 1 to 5 denoting how similar the two sentences are in terms of semantic meaning.

MRPC Microsoft Research Paraphrase Corpus consists of sentence pairs automatically extracted from online news sources, with human annotations for whether the sentences in the pair are semantically equivalent (Dolan and Brockett, 2005).

<table><tr><td>System</td><td>MNLI-(m/mm)392k</td><td>QQP363k</td><td>QNLI108k</td><td>SST-267k</td><td>CoLA8.5k</td><td>STS-B5.7k</td><td>MRPC3.5k</td><td>RTE2.5k</td><td>Average-</td></tr><tr><td>Pre-OpenAI SOTA</td><td>80.6/80.1</td><td>66.1</td><td>82.3</td><td>93.2</td><td>35.0</td><td>81.0</td><td>86.0</td><td>61.7</td><td>74.0</td></tr><tr><td>BiLSTM+ELMo+Attn</td><td>76.4/76.1</td><td>64.8</td><td>79.9</td><td>90.4</td><td>36.0</td><td>73.3</td><td>84.9</td><td>56.8</td><td>71.0</td></tr><tr><td>OpenAI GPT</td><td>82.1/81.4</td><td>70.3</td><td>88.1</td><td>91.3</td><td>45.4</td><td>80.0</td><td>82.3</td><td>56.0</td><td>75.2</td></tr><tr><td>BERTBASE</td><td>84.6/83.4</td><td>71.2</td><td>90.1</td><td>93.5</td><td>52.1</td><td>85.8</td><td>88.9</td><td>66.4</td><td>79.6</td></tr><tr><td>BERTLARGE</td><td>86.7/85.9</td><td>72.1</td><td>91.1</td><td>94.9</td><td>60.5</td><td>86.5</td><td>89.3</td><td>70.1</td><td>81.9</td></tr></table>

Table 1: GLUE Test results, scored by the GLUE evaluation server. The number below each task denotes the number of training examples. The “Average” column is slightly different than the official GLUE score, since we exclude the problematic WNLI set. OpenAI GPT = (L=12, H=768, A=12); $\mathrm { B E R T _ { B A S E } } = ( \mathrm { L } { = } 1 2$ , H=768, A=12); $\mathtt { B E R T \_ A R G E } = ( \mathrm { L } = 2 4 $ , H=1024, A=16). BERT and OpenAI GPT are single-model, single task. All results obtained from https://gluebenchmark.com/leaderboard and https://blog.openai. com/language-unsupervised/.

RTE Recognizing Textual Entailment is a binary entailment task similar to MNLI, but with much less training data (Bentivogli et al., 2009).6

WNLI Winograd NLI is a small natural language inference dataset deriving from (Levesque et al., 2011). The GLUE webpage notes that there are issues with the construction of this dataset, and every trained system that’s been submitted to GLUE has has performed worse than the 65.1 baseline accuracy of predicting the majority class. We therefore exclude this set out of fairness to OpenAI GPT. For our GLUE submission, we always predicted the majority class.

# 4.1.1 GLUE Results

To fine-tune on GLUE, we represent the input sequence or sequence pair as described in Section 3, and use the final hidden vector $C \in \mathbb { R } ^ { H }$ corresponding to the first input token ([CLS]) as the aggregate representation. This is demonstrated visually in Figure 3 (a) and (b). The only new parameters introduced during fine-tuning is a classification layer $W \in \mathbb { R } ^ { K \times \smile }$ , where K is the number of labels. We compute a standard classification loss with C and W , i.e., log(softmax $( C W ^ { T } ) )$ ).

We use a batch size of 32 and 3 epochs over the data for all GLUE tasks. For each task, we ran fine-tunings with learning rates of 5e-5, 4e-5, 3e-5, and 2e-5 and selected the one that performed best on the Dev set. Additionally, for $\mathbf { B E R T _ { L A R G E } }$ we found that fine-tuning was sometimes unstable on small data sets (i.e., some runs would produce degenerate results), so we ran several random restarts and selected the model that performed best on the Dev set. With random restarts, we use the same pre-trained checkpoint but perform different finetuning data shuffling and classifier layer initialization. We note that the GLUE data set distribution does not include the Test labels, and we only made a single GLUE evaluation server submission for each BERTBASE and BERTLARGE.

Results are presented in Table 1. Both $\mathbf { B E R T _ { B A S E } }$ and $\mathbf { B E R T _ { L A R G E } }$ outperform all existing systems on all tasks by a substantial margin, obtaining 4.4% and 6.7% respective average accuracy improvement over the state-of-the-art. Note that $\mathbf { B E R T _ { B A S E } }$ and OpenAI GPT are nearly identical in terms of model architecture outside of the attention masking. For the largest and most widely reported GLUE task, MNLI, BERT obtains a 4.7% absolute accuracy improvement over the state-of-the-art. On the official GLUE leaderboard, $^ 3 \mathrm { \mathbf { B E R T _ { L A R G E } } }$ obtains a score of 80.4, compared to the top leaderboard system, OpenAI GPT, which obtains 72.8 as of the date of writing.

It is interesting to observe that BERTLARGE significantly outperforms BERTBASE across all tasks, even those with very little training data. The effect of BERT model size is explored more thoroughly in Section 5.2.

# 4.2 SQuAD v1.1

The Standford Question Answering Dataset (SQuAD) is a collection of 100k crowdsourced question/answer pairs (Rajpurkar et al., 2016). Given a question and a paragraph from Wikipedia containing the answer, the task is to predict the answer text span in the paragraph. For example:

• Input Question:

Where do water droplets collide with ice crystals to form precipitation?

• Input Paragraph:

.. Precipitation forms as smaller droplets coalesce via collision with other rain drops or ice crystals within a cloud.

• Output Answer:

within a cloud

This type of span prediction task is quite different from the sequence classification tasks of GLUE, but we are able to adapt BERT to run on SQuAD in a straightforward manner. Just as with GLUE, we represent the input question and paragraph as a single packed sequence, with the question using the A embedding and the paragraph using the B embedding. The only new parameters learned during fine-tuning are a start vector $S \in \mathbb { R } ^ { H }$ and an end vector $E \in \mathbb { R } ^ { H }$ . Let the final hidden vector from BERT for the $i ^ { \mathrm { t h } }$ input token be denoted as $T _ { i } \in \mathbb { R } ^ { H }$ . See Figure 3 (c) for a visualization. Then, the probability of word i being the start of the answer span is computed as a dot product between $T _ { i }$ and S followed by a softmax over all of the words in the paragraph:

$$
P _ {i} = \frac {e ^ {S \cdot T _ {i}}}{\sum_ {j} e ^ {S \cdot T _ {j}}}
$$

The same formula is used for the end of the answer span, and the maximum scoring span is used as the prediction. The training objective is the loglikelihood of the correct start and end positions.

We train for 3 epochs with a learning rate of 5e-5 and a batch size of 32. At inference time, since the end prediction is not conditioned on the start, we add the constraint that the end must come after the start, but no other heuristics are used. The tokenized labeled span is aligned back to the original untokenized input for evaluation.

Results are presented in Table 2. SQuAD uses a highly rigorous testing procedure where the submitter must manually contact the SQuAD organizers to run their system on a hidden test set, so we only submitted our best system for testing. The result shown in the table is our first and only Test submission to SQuAD. We note that the top results from the SQuAD leaderboard do not have up-todate public system descriptions available, and are allowed to use any public data when training their systems. We therefore use very modest data augmentation in our submitted system by jointly training on SQuAD and TriviaQA (Joshi et al., 2017).

<table><tr><td rowspan="2">System</td><td colspan="2">Dev</td><td colspan="2">Test</td></tr><tr><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><td colspan="5">Leaderboard (Oct 8th, 2018)</td></tr><tr><td>Human</td><td>-</td><td>-</td><td>82.3</td><td>91.2</td></tr><tr><td>#1 Ensemble - nlnet</td><td>-</td><td>-</td><td>86.0</td><td>91.7</td></tr><tr><td>#2 Ensemble - QANet</td><td>-</td><td>-</td><td>84.5</td><td>90.5</td></tr><tr><td>#1 Single - nlnet</td><td>-</td><td>-</td><td>83.5</td><td>90.1</td></tr><tr><td>#2 Single - QANet</td><td>-</td><td>-</td><td>82.5</td><td>89.3</td></tr><tr><td colspan="5">Published</td></tr><tr><td>BiDAF+ELMo (Single)</td><td>-</td><td>85.8</td><td>-</td><td>-</td></tr><tr><td>R.M. Reader (Single)</td><td>78.9</td><td>86.3</td><td>79.5</td><td>86.6</td></tr><tr><td>R.M. Reader (Ensemble)</td><td>81.2</td><td>87.9</td><td>82.3</td><td>88.5</td></tr><tr><td colspan="5">Ours</td></tr><tr><td>BERTBASE(Single)</td><td>80.8</td><td>88.5</td><td>-</td><td>-</td></tr><tr><td>BERTLARGE(Single)</td><td>84.1</td><td>90.9</td><td>-</td><td>-</td></tr><tr><td>BERTLARGE(Ensemble)</td><td>85.8</td><td>91.8</td><td>-</td><td>-</td></tr><tr><td>BERTLARGE(Sgl.+TriviaQA)</td><td>84.2</td><td>91.1</td><td>85.1</td><td>91.8</td></tr><tr><td>BERTLARGE(Ens.+TriviaQA)</td><td>86.2</td><td>92.2</td><td>87.4</td><td>93.2</td></tr></table>

Table 2: SQuAD results. The BERT ensemble is 7x systems which use different pre-training checkpoints and fine-tuning seeds.

Our best performing system outperforms the top leaderboard system by +1.5 F1 in ensembling and +1.3 F1 as a single system. In fact, our single BERT model outperforms the top ensemble system in terms of F1 score. If we fine-tune on only SQuAD (without TriviaQA) we lose 0.1-0.4 F1 and still outperform all existing systems by a wide margin.

# 4.3 Named Entity Recognition

To evaluate performance on a token tagging task, we fine-tune BERT on the CoNLL 2003 Named Entity Recognition (NER) dataset. This dataset consists of 200k training words which have been annotated as Person, Organization, Location, Miscellaneous, or Other (non-named entity).

For fine-tuning, we feed the final hidden representation $T _ { i } ~ \in ~ \mathbb { R } ^ { H }$ for to each token i into a classification layer over the NER label set. The predictions are not conditioned on the surrounding predictions (i.e., non-autoregressive and no CRF). To make this compatible with WordPiece tokenization, we feed each CoNLL-tokenized input word into our WordPiece tokenizer and use the hidden state corresponding to the first sub-token as input to the classifier. For example:

<table><tr><td>System</td><td>Dev F1</td><td>Test F1</td></tr><tr><td>ELMo+BiLSTM+CRF</td><td>95.7</td><td>92.2</td></tr><tr><td>CVT+Multi (Clark et al., 2018)</td><td>-</td><td>92.6</td></tr><tr><td>BERTBASE</td><td>96.4</td><td>92.4</td></tr><tr><td>BERTLARGE</td><td>96.6</td><td>92.8</td></tr></table>

Table 3: CoNLL-2003 Named Entity Recognition results. The hyperparameters were selected using the Dev set, and the reported Dev and Test scores are averaged over 5 random restarts using those hyperparameters.

<table><tr><td>Jim</td><td>Hen</td><td>##son</td><td>was</td><td>a</td><td>puppet</td><td>##eer</td></tr><tr><td>I-PER</td><td>I-PER</td><td>X</td><td>O</td><td>O</td><td>O</td><td>X</td></tr></table>

Where no prediction is made for X. Since the WordPiece tokenization boundaries are a known part of the input, this is done for both training and test. A visual representation is also given in Figure 3 (d). A cased WordPiece model is used for NER, whereas an uncased model is used for all other tasks.

Results are presented in Table 3. BERTLARGE outperforms the existing SOTA, Cross-View Training with multi-task learning (Clark et al., 2018), by +0.2 on CoNLL-2003 NER Test.

# 4.4 SWAG

The Situations With Adversarial Generations (SWAG) dataset contains 113k sentence-pair completion examples that evaluate grounded commonsense inference (Zellers et al., 2018).

Given a sentence from a video captioning dataset, the task is to decide among four choices the most plausible continuation. For example:

A girl is going across a set of monkey bars. She

(i) jumps up across the monkey bars.   
(ii) struggles onto the bars to grab her head.   
(iii) gets to the end and stands on a wooden plank.   
(iv) jumps up and does a back flip.

Adapting BERT to the SWAG dataset is similar to the adaptation for GLUE. For each example, we construct four input sequences, which each contain the concatenation of the the given sentence (sentence A) and a possible continuation (sentence B). The only task-specific parameters we introduce is a vector $\bar { \boldsymbol { V } } \in \mathbb { R } ^ { \bar { H } }$ , whose dot product with the final aggregate representation $C _ { i } \in \mathbb { R } ^ { H }$ denotes a score for each choice i. The probability distribution is the softmax over the four choices:

<table><tr><td>System</td><td>Dev</td><td>Test</td></tr><tr><td>ESIM+GloVe</td><td>51.9</td><td>52.7</td></tr><tr><td>ESIM+ELMo</td><td>59.1</td><td>59.2</td></tr><tr><td>BERTBASE</td><td>81.6</td><td>-</td></tr><tr><td>BERTLARGE</td><td>86.6</td><td>86.3</td></tr><tr><td>Human (expert) $^{\dagger}$ </td><td>-</td><td>85.0</td></tr><tr><td>Human (5 annotations) $^{\dagger}$ </td><td>-</td><td>88.0</td></tr></table>

Table 4: SWAG Dev and Test accuracies. Test results were scored against the hidden labels by the SWAG authors. †Human performance is measure with 100 samples, as reported in the SWAG paper.

$$
P _ {i} = \frac {e ^ {V \cdot C _ {i}}}{\sum_ {j = 1} ^ {4} e ^ {V \cdot C _ {j}}}
$$

We fine-tune the model for 3 epochs with a learning rate of 2e-5 and a batch size of 16. Results are presented in Table 4. BERTLARGE outperforms the authors’ baseline ESIM+ELMo system by +27.1%.

# 5 Ablation Studies

Although we have demonstrated extremely strong empirical results, the results presented so far have not isolated the specific contributions from each aspect of the BERT framework. In this section, we perform ablation experiments over a number of facets of BERT in order to better understand their relative importance.

# 5.1 Effect of Pre-training Tasks

One of our core claims is that the deep bidirectionality of BERT, which is enabled by masked LM pre-training, is the single most important improvement of BERT compared to previous work. To give evidence for this claim, we evaluate two new models which use the exact same pre-training data, fine-tuning scheme and Transformer hyperparameters as BERTBASE:

1. No NSP: A model which is trained using the “masked LM” (MLM) but without the “next sentence prediction” (NSP) task.   
2. LTR & No NSP: A model which is trained using a Left-to-Right (LTR) LM, rather than

an MLM. In this case, we predict every input word and do not apply any masking. The left-only constraint was also applied at finetuning, because we found it is always worse to pre-train with left-only-context and finetune with bidirectional context. Additionally, this model was pre-trained without the NSP task. This is directly comparable to OpenAI GPT, but using our larger training dataset, our input representation, and our fine-tuning scheme.

Results are presented in Table 5. We first examine the impact brought by the NSP task. We can see that removing NSP hurts performance significantly on QNLI, MNLI, and SQuAD. These results demonstrate that our pre-training method is critical in obtaining the strong empirical results presented previously.

Next, we evaluate the impact of training bidirectional representations by comparing “No NSP” to “LTR & No $\mathrm { N S P ^ { \prime } }$ . The LTR model performs worse than the MLM model on all tasks, with extremely large drops on MRPC and SQuAD. For SQuAD it is intuitively clear that an LTR model will perform very poorly at span and token prediction, since the token-level hidden states have no right-side context. For MRPC is unclear whether the poor performance is due to the small data size or the nature of the task, but we found this poor performance to be consistent across a full hyperparameter sweep with many random restarts.

In order make a good faith attempt at strengthening the LTR system, we tried adding a randomly initialized BiLSTM on top of it for finetuning. This does significantly improve results on SQuAD, but the results are still far worse than the pre-trained bidirectional models. It also hurts performance on all four GLUE tasks.

<table><tr><td rowspan="2">Tasks</td><td colspan="5">Dev Set</td></tr><tr><td>MNLI-m (Acc)</td><td>QNLI (Acc)</td><td>MRPC (Acc)</td><td>SST-2 (Acc)</td><td>SQuAD (F1)</td></tr><tr><td> $BERT_{BASE}$ </td><td>84.4</td><td>88.4</td><td>86.7</td><td>92.7</td><td>88.5</td></tr><tr><td>No NSP</td><td>83.9</td><td>84.9</td><td>86.5</td><td>92.6</td><td>87.9</td></tr><tr><td>LTR &amp; No NSP</td><td>82.1</td><td>84.3</td><td>77.5</td><td>92.1</td><td>77.8</td></tr><tr><td>+ BiLSTM</td><td>82.1</td><td>84.1</td><td>75.7</td><td>91.6</td><td>84.9</td></tr></table>

Table 5: Ablation over the pre-training tasks using the $\mathbf { B E R T _ { B A S E } }$ architecture. “No NSP” is trained without the next sentence prediction task. “LTR & No NSP” is trained as a left-to-right LM without the next sentence prediction, like OpenAI GPT. “+ BiLSTM” adds a randomly initialized BiLSTM on top of the “LTR + No $\mathrm { N S P ^ { , } }$ model during fine-tuning.

We recognize that it would also be possible to train separate LTR and RTL models and represent each token as the concatenation of the two models, as ELMo does. However: (a) this is twice as expensive as a single bidirectional model; (b) this is non-intuitive for tasks like QA, since the RTL model would not be able to condition the answer on the question; (c) this it is strictly less powerful than a deep bidirectional model, since a deep bidirectional model could choose to use either left or right context.

# 5.2 Effect of Model Size

In this section, we explore the effect of model size on fine-tuning task accuracy. We trained a number of BERT models with a differing number of layers, hidden units, and attention heads, while otherwise using the same hyperparameters and training procedure as described previously.

Results on selected GLUE tasks are shown in Table 6. In this table, we report the average Dev Set accuracy from 5 random restarts of fine-tuning. We can see that larger models lead to a strict accuracy improvement across all four datasets, even for MRPC which only has 3,600 labeled training examples, and is substantially different from the pre-training tasks. It is also perhaps surprising that we are able to achieve such significant improvements on top of models which are already quite large relative to the existing literature. For example, the largest Transformer explored in Vaswani et al. (2017) is (L=6, H=1024, A=16) with 100M parameters for the encoder, and the largest Transformer we have found in the literature is (L=64, H=512, A=2) with 235M parameters (Al-Rfou et al., 2018). By contrast, BERTBASE contains 110M parameters and $\mathbf { B E R T _ { L A R G E } }$ contains 340M parameters.

<table><tr><td colspan="4">Hyperparams</td><td colspan="3">Dev Set Accuracy</td></tr><tr><td>#L</td><td>#H</td><td>#A</td><td>LM (ppl)</td><td>MNLI-m</td><td>MRPC</td><td>SST-2</td></tr><tr><td>3</td><td>768</td><td>12</td><td>5.84</td><td>77.9</td><td>79.8</td><td>88.4</td></tr><tr><td>6</td><td>768</td><td>3</td><td>5.24</td><td>80.6</td><td>82.2</td><td>90.7</td></tr><tr><td>6</td><td>768</td><td>12</td><td>4.68</td><td>81.9</td><td>84.8</td><td>91.3</td></tr><tr><td>12</td><td>768</td><td>12</td><td>3.99</td><td>84.4</td><td>86.7</td><td>92.9</td></tr><tr><td>12</td><td>1024</td><td>16</td><td>3.54</td><td>85.7</td><td>86.9</td><td>93.3</td></tr><tr><td>24</td><td>1024</td><td>16</td><td>3.23</td><td>86.6</td><td>87.8</td><td>93.7</td></tr></table>

Table 6: Ablation over BERT model size. #L = the number of layers; #H = hidden size; #A = number of attention heads. “LM (ppl)” is the masked LM perplexity of held-out training data.

It has been known for many years that increasing the model size will lead to continual improvements on large-scale tasks such as machine translation and language modeling, which is demonstrated by the LM perplexity of held-out training data shown in Table 6. However, we believe that this is the first work to demonstrate that scaling to extreme model sizes also leads to large improvements on very small scale tasks, provided that the model has been sufficiently pre-trained.

# 5.3 Effect of Number of Training Steps

Figure 4 presents MNLI Dev accuracy after finetuning from a checkpoint that has been pre-trained for k steps. This allows us to answer the following questions:

1. Question: Does BERT really need such a large amount of pre-training (128,000 words/batch \* 1,000,000 steps) to achieve high fine-tuning accuracy?

Answer: Yes, BERTBASE achieves almost 1.0% additional accuracy on MNLI when trained on 1M steps compared to 500k steps.

2. Question: Does MLM pre-training converge slower than LTR pre-training, since only 15% of words are predicted in each batch rather than every word?

Answer: The MLM model does converge slightly slower than the LTR model. However, in terms of absolute accuracy the MLM model begins to outperform the LTR model almost immediately.

![](images/fc41d4c6f728f5ad12b1b4a8f33d108df0eae64de078535d2b9f65e47c75e3d8.jpg)

<details>
<summary>line</summary>

| Pre-training Steps (Thousands) | BERT_BASE (Masked LM) | BERT_BASE (Left-to-Right) |
| ------------------------------ | --------------------- | ------------------------- |
| 0                              | 78.5                  | 79.5                      |
| 200                            | 82.0                  | 80.5                      |
| 400                            | 83.0                  | 81.5                      |
| 600                            | 84.0                  | 82.0                      |
| 800                            | 84.5                  | 82.5                      |
| 1,000                          | 84.5                  | 82.5                      |
</details>

Figure 4: Ablation over number of training steps. This shows the MNLI accuracy after fine-tuning, starting from model parameters that have been pre-trained for k steps. The x-axis is the value of k.

# 5.4 Feature-based Approach with BERT

All of the BERT results presented so far have used the fine-tuning approach, where a simple classification layer is added to the pre-trained model, and all parameters are jointly fine-tuned on a downstream task. However, the feature-based approach, where fixed features are extracted from the pretrained model, has certain advantages. First, not all NLP tasks can be easily be represented by a Transformer encoder architecture, and therefore require a task-specific model architecture to be added. Second, there are major computational benefits to being able to pre-compute an expensive representation of the training data once and then run many experiments with less expensive models on top of this representation.

In this section we evaluate how well BERT performs in the feature-based approach by generating ELMo-like pre-trained contextual representations on the CoNLL-2003 NER task. To do this, we use the same input representation as in Section 4.3, but use the activations from one or more layers without fine-tuning any parameters of BERT. These contextual embeddings are used as input to a randomly initialized two-layer 768-dimensional BiL-STM before the classification layer.

Results are shown in Table 7. The best performing method is to concatenate the token representations from the top four hidden layers of the pretrained Transformer, which is only 0.3 F1 behind fine-tuning the entire model. This demonstrates that BERT is effective for both the fine-tuning and feature-based approaches.

<table><tr><td>Layers</td><td>Dev F1</td></tr><tr><td>Finetune All</td><td>96.4</td></tr><tr><td>First Layer (Embeddings)</td><td>91.0</td></tr><tr><td>Second-to-Last Hidden</td><td>95.6</td></tr><tr><td>Last Hidden</td><td>94.9</td></tr><tr><td>Sum Last Four Hidden</td><td>95.9</td></tr><tr><td>Concat Last Four Hidden</td><td>96.1</td></tr><tr><td>Sum All 12 Layers</td><td>95.5</td></tr></table>

Table 7: Ablation using BERT with a feature-based approach on CoNLL-2003 NER. The activations from the specified layers are combined and fed into a two-layer BiLSTM, without backpropagation to BERT.

# 6 Conclusion

Recent empirical improvements due to transfer learning with language models have demonstrated that rich, unsupervised pre-training is an integral part of many language understanding systems. In particular, these results enable even low-resource tasks to benefit from very deep unidirectional architectures. Our major contribution is further generalizing these findings to deep bidirectional architectures, allowing the same pre-trained model to successfully tackle a broad set of NLP tasks.

While the empirical results are strong, in some cases surpassing human performance, important future work is to investigate the linguistic phenomena that may or may not be captured by BERT.

# References

Rami Al-Rfou, Dokook Choe, Noah Constant, Mandy Guo, and Llion Jones. 2018. Character-level language modeling with deeper self-attention. arXiv preprint arXiv:1808.04444.   
Rie Kubota Ando and Tong Zhang. 2005. A framework for learning predictive structures from multiple tasks and unlabeled data. Journal of Machine Learning Research, 6(Nov):1817–1853.   
Luisa Bentivogli, Bernardo Magnini, Ido Dagan, Hoa Trang Dang, and Danilo Giampiccolo. 2009. The fifth PASCAL recognizing textual entailment challenge. In TAC. NIST.   
John Blitzer, Ryan McDonald, and Fernando Pereira. 2006. Domain adaptation with structural correspondence learning. In Proceedings of the 2006 conference on empirical methods in natural language processing, pages 120–128. Association for Computational Linguistics.   
Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In EMNLP. Association for Computational Linguistics.   
Peter F Brown, Peter V Desouza, Robert L Mercer, Vincent J Della Pietra, and Jenifer C Lai. 1992. Class-based n-gram models of natural language. Computational linguistics, 18(4):467–479.   
Daniel Cer, Mona Diab, Eneko Agirre, Inigo Lopez-Gazpio, and Lucia Specia. 2017. Semeval-2017 task 1: Semantic textual similarity-multilingual and cross-lingual focused evaluation. arXiv preprint arXiv:1708.00055.   
Ciprian Chelba, Tomas Mikolov, Mike Schuster, Qi Ge, Thorsten Brants, Phillipp Koehn, and Tony Robinson. 2013. One billion word benchmark for measur-

ing progress in statistical language modeling. arXiv preprint arXiv:1312.3005.   
Z. Chen, H. Zhang, X. Zhang, and L. Zhao. 2018. Quora question pairs.   
Kevin Clark, Minh-Thang Luong, Christopher D Manning, and Quoc V Le. 2018. Semi-supervised sequence modeling with cross-view training. arXiv preprint arXiv:1809.08370.   
Ronan Collobert and Jason Weston. 2008. A unified architecture for natural language processing: Deep neural networks with multitask learning. In Proceedings of the 25th International Conference on Machine Learning, ICML ’08.   
Alexis Conneau, Douwe Kiela, Holger Schwenk, Lo¨ıc Barrault, and Antoine Bordes. 2017. Supervised learning of universal sentence representations from natural language inference data. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 670–680, Copenhagen, Denmark. Association for Computational Linguistics.   
Andrew M Dai and Quoc V Le. 2015. Semi-supervised sequence learning. In Advances in neural information processing systems, pages 3079–3087.   
J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. 2009. ImageNet: A Large-Scale Hierarchical Image Database. In CVPR09.   
William B Dolan and Chris Brockett. 2005. Automatically constructing a corpus of sentential paraphrases. In Proceedings of the Third International Workshop on Paraphrasing (IWP2005).   
Dan Hendrycks and Kevin Gimpel. 2016. Bridging nonlinearities and stochastic regularizers with gaussian error linear units. CoRR, abs/1606.08415.   
Jeremy Howard and Sebastian Ruder. 2018. Universal language model fine-tuning for text classification. In ACL. Association for Computational Linguistics.   
Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. 2017. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In ACL.   
Ryan Kiros, Yukun Zhu, Ruslan R Salakhutdinov, Richard Zemel, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. 2015. Skip-thought vectors. In Advances in neural information processing systems, pages 3294–3302.   
Quoc Le and Tomas Mikolov. 2014. Distributed representations of sentences and documents. In International Conference on Machine Learning, pages 1188–1196.   
Hector J Levesque, Ernest Davis, and Leora Morgenstern. 2011. The winograd schema challenge. In Aaai spring symposium: Logical formalizations of commonsense reasoning, volume 46, page 47.

Lajanugen Logeswaran and Honglak Lee. 2018. An efficient framework for learning sentence representations. In International Conference on Learning Representations.   
Bryan McCann, James Bradbury, Caiming Xiong, and Richard Socher. 2017. Learned in translation: Contextualized word vectors. In NIPS.   
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013. Distributed representations of words and phrases and their compositionality. In Advances in Neural Information Processing Systems 26, pages 3111–3119. Curran Associates, Inc.   
Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. Glove: Global vectors for word representation. In Empirical Methods in Natural Language Processing (EMNLP), pages 1532– 1543.   
Matthew Peters, Waleed Ammar, Chandra Bhagavatula, and Russell Power. 2017. Semi-supervised sequence tagging with bidirectional language models. In ACL.   
Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018. Deep contextualized word representations. In NAACL.   
Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2018. Improving language understanding with unsupervised learning. Technical report, OpenAI.   
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: 100,000+ questions for machine comprehension of text. arXiv preprint arXiv:1606.05250.   
Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 conference on empirical methods in natural language processing, pages 1631–1642.   
Wilson L Taylor. 1953. cloze procedure: A new tool for measuring readability. Journalism Bulletin, 30(4):415–433.   
Erik F Tjong Kim Sang and Fien De Meulder. 2003. Introduction to the conll-2003 shared task: Language-independent named entity recognition. In Proceedings of the seventh conference on Natural language learning at HLT-NAACL 2003-Volume 4, pages 142–147. Association for Computational Linguistics.   
Joseph Turian, Lev Ratinov, and Yoshua Bengio. 2010. Word representations: A simple and general method for semi-supervised learning. In Proceedings of the 48th Annual Meeting of the Association for Computational Linguistics, ACL ’10, pages 384–394.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems, pages 6000–6010.   
Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. 2008. Extracting and composing robust features with denoising autoencoders. In Proceedings of the 25th international conference on Machine learning, pages 1096–1103. ACM.   
Alex Wang, Amapreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. 2018. Glue: A multi-task benchmark and analysis platform for natural language understanding. arXiv preprint arXiv:1804.07461.   
A. Warstadt, A. Singh, and S. R. Bowman. 2018. Corpus of linguistic acceptability.   
Adina Williams, Nikita Nangia, and Samuel R Bowman. 2018. A broad-coverage challenge corpus for sentence understanding through inference. In NAACL.   
Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. 2016. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144.   
Jason Yosinski, Jeff Clune, Yoshua Bengio, and Hod Lipson. 2014. How transferable are features in deep neural networks? In Advances in neural information processing systems, pages 3320–3328.   
Rowan Zellers, Yonatan Bisk, Roy Schwartz, and Yejin Choi. 2018. Swag: A large-scale adversarial dataset for grounded commonsense inference. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing (EMNLP).   
Yukun Zhu, Ryan Kiros, Rich Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler. 2015. Aligning books and movies: Towards story-like visual explanations by watching movies and reading books. In Proceedings of the IEEE international conference on computer vision, pages 19–27.