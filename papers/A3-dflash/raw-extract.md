# DFlash: Block Diffusion for Flash Speculative Decoding

Jian Chen 1 Yesheng Liang 1 Zhijian Liu 1

https://dflash.z-lab.ai

## Abstract

Autoregressive large language models (LLMs) deliver strong performance but require inherently sequential decoding, leading to high inference latency and poor GPU utilization. Speculative decoding mitigates this bottleneck by using a fast draft model whose outputs are verified in parallel by the target LLM. However, existing methods still rely on autoregressive drafting, which remains sequential and constrains practical speedups. Diffusion LLMs offer a promising alternative by enabling parallel generation, but current diffusion models typically underperform compared with autoregressive models. In this paper, we introduce DFlash, a speculative decoding framework that employs a lightweight block diffusion model for parallel drafting. We show that speculative decoding provides a natural and effective setting for diffusion models. By generating draft tokens in a single forward pass, DFlash enables efficient drafting, and by conditioning the draft model on context features extracted from the target model, it achieves high-quality drafts with higher acceptance rates. Experiments show that DFlash achieves over 6× lossless acceleration across a range of models and tasks, delivering up to 2.5× higher speedup than the state-of-the-art speculative decoding method EAGLE-3.

Links: Code (GitHub) | Models (Hugging Face)

## 1. Introduction

Large language models (LLMs) have enabled a wide range of powerful applications, including conversational agents (Yang et al., 2025; Guo et al., 2025) and automated programming tools. Despite their success, LLM inference

1UC San Diego. Correspondence to: Zhijian Liu <zhijian@ ucsd.edu>.

Proceedings of the $\it 4 3 ^ { r d }$ International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

remains dominated by a sequential, token-by-token generation process, where each output depends on the full preceding context. This inherent seriality creates a major performance bottleneck: inference is slow, memory-bound, and fails to fully utilize modern GPUs. With the recent emergence of long Chain-of-Thought (CoT) reasoning models (OpenAI et al., 2024; Guo et al., 2025), this bottleneck has become increasingly critical, as prolonged inference times now dominate the generation process.

Speculative decoding (Leviathan et al., 2023; Li et al., 2025c; 2024; 2025b; Cai et al., 2024) has emerged as a primary solution to this bottleneck. This paradigm employs a lightweight draft model to speculate a sequence of future tokens, which are then verified in parallel by the large target model. While this approach achieves lossless acceleration and has been widely integrated into production frameworks, state-of-the-art methods like EAGLE-3 (Li et al., 2025b) still rely on autoregressive drafting. This serial drafting process is not only inherently inefficient but also susceptible to error accumulation, which effectively caps achievable speedups at approximately 2−3×.

Recently, Diffusion LLMs (dLLMs) (Nie et al., 2025) offer a promising alternative to autoregressive LLMs by enabling parallel text generation and bidirectional context modeling. Block diffusion models (Arriola et al., 2025; Cheng et al., 2025; Wu et al., 2025) can denoise a block of masked tokens simultaneously. However, current open-source dLLMs typically underperform their autoregressive counterparts in terms of generation quality. Furthermore, maintaining acceptable output quality often necessitates a high number of denoising steps, which significantly diminishes their raw inference speed (Qian et al., 2026).

This landscape reveals a critical trade-off: autoregressive models deliver superior performance but suffer from sequential latency, while diffusion models allow for fast, parallel generation but often at the cost of accuracy. A natural research question follows: Can we combine the strengths of both paradigms while mitigating their respective weaknesses? A compelling solution lies in leveraging diffusion models for high-speed, parallel drafting, while relying on high-quality autoregressive models for verification to ensure the final output remains lossless.

![](images/174e35fbc7f5152d2dd3971c146494439009357e31c4f7c5a27fd89a607f497c.jpg)

<details>
<summary>bar chart</summary>

| Benchmark | Baseline | EAGLE-3 | DFlash |
| :--- | :--- | :--- | :--- |
| GSM8K | 1.00 | 2.23 | 5.15 |
| Math500 | 1.00 | 2.05 | 6.08 |
| AIME25 | 1.00 | 2.05 | 5.62 |
| HumanEval | 1.00 | 2.17 | 5.14 |
| MBPP | 1.00 | 1.93 | 4.65 |
| LiveCodeBench | 1.00 | 1.81 | 5.51 |
| MT-Bench | 1.00 | 1.90 | 2.75 |
</details>

Figure 1. Speedup comparison between DFlash, EAGLE-3 against Autoregressive Decoding on Qwen3-8B (Yang et al., 2025) with the Transformers backend. Overall, DFlash achieves more than 2.5× higher speedup than EAGLE-3.

However, utilizing diffusion for drafting is non-trivial, and existing methods are either impractical or offer limited speedups. Methods such as DiffuSpec (Li et al., 2025a) and SpecDiff-2 (Sandler et al., 2025) utilize massive (e.g., 7B parameter) draft models. This significant memory footprint is often prohibitively expensive for real-world serving. Furthermore, while these large drafters offer relatively high quality draft tokens and acceptance lengths, the high drafting latency limits their practical speedups to a modest 3−4×. In contrast, PARD (An et al., 2025) trains small autoregressive models to mimic diffusion-style parallel generation, and then perform speculative decoding for target LLMs. However, the resulting small models lack the modeling capacity of the target LLMs, leading to limited acceptance lengths and a speedup ceiling of approximately 3×.

Is there truly “no free lunch”? Can we build a diffusion drafter that is both lightweight and highly accurate?

In this paper, we introduce DFlash, a speculative decoding framework that uses a lightweight block diffusion model to achieve both fast and high-quality drafting. Our key insight is simple: the target knows best. As observed by Samragh et al. (2025), large autoregressive LLMs’ hidden features implicitly contain information about multiple future tokens. DFlash utilizes these hidden features as context, conditioning the draft model to predict future blocks of tokens in parallel. In effect, the draft model becomes a diffusion adapter that efficiently leverages the deep context features modeled by the large target model. Instead of requiring a tiny draft model to reason from scratch, DFlash fuses the reasoning capabilities of the target model with the parallel generation speed of a small diffusion drafter.

We evaluate DFlash across a wide range of models and benchmarks, and demonstrate its practical benefits under realistic serving setups using SGLang (Zheng et al., 2024). As shown in Figure 1, DFlash achieves up to a 6.1× speedup on Qwen3-8B (Yang et al., 2025), and is nearly 2.5× faster than the state-of-the-art EAGLE-3 across most benchmarks. We believe DFlash represents a significant step forward in accelerating LLM inference and democratizing highperformance AI.

Conflict of Interest Disclosure. The authors declare no financial conflicts of interest related to this work. Research support and compute resources are acknowledged in the Acknowledgements section.

## 2. Related Work

## 2.1. Speculative Decoding

Speculative decoding accelerates LLM inference by mitigating the sequential bottleneck of autoregressive generation. Early methods (Leviathan et al., 2023) employ a smaller draft model to propose token sequences that are verified in parallel by a larger target model. Medusa (Cai et al., 2024) eliminates the external draft model by augmenting the base LLM with multiple prediction heads and using tree attention for parallel verification. The EAGLE series (Li et al., 2025c; 2024; 2025b) further improves speculative decoding by exploiting feature-level context from the frozen target model. EAGLE-1 predicts future hidden-state distributions to boost acceptance rates, EAGLE-2 introduces adaptive drafting trees, and EAGLE-3 refines training objectives to scale speedups.

Despite these advances, most existing methods rely on autoregressive drafting, which remains inherently sequential, limiting their speedups.

## 2.2. Diffusion Language Models

Diffusion large language models (dLLMs) offer an alternative to autoregressive generation by predicting masked tokens in parallel. LLaDA (Nie et al., 2025) was the first to scale dLLMs to billions of parameters, achieving performance comparable to LLaMA-3.1-8B (Grattafiori et al., 2024). However, fully parallel diffusion models suffer from fixed-length generation and lack efficient KV cache support. Block diffusion models (Arriola et al., 2025) address these issues by denoising sequences block-by-block, blending parallelism with autoregressive structure. Building on this idea, Fast-dLLM v2 (Wu et al., 2025) and SDAR (Cheng et al., 2025) adapt pre-trained autoregressive LLMs into block-diffusion variants, enabling parallel generation while preserving generation quality on specific tasks. Nevertheless, existing dLLMs generally underperform state-of-theart autoregressive models and often require many denoising steps, which limits their practical inference speed.

## 2.3. Diffusion-based Speculative Decoding

Recent work explores using diffusion models as drafters within speculative decoding. TiDAR (Liu et al., 2025) jointly trains diffusion and autoregressive objectives, enabling parallel “thinking” via diffusion and sequential “talking” via autoregressive decoding, though final generation quality is not yet lossless.

Other approaches repurpose autoregressive models for diffusion-style drafting. Samragh et al. (2025) observe that autoregressive LLMs implicitly encode future-token information and train a LoRA adapter to enable parallel drafting, while retaining the base model for verification.

DiffuSpec (Li et al., 2025a) and SpecDiff-2 (Sandler et al., 2025) employ large pre-trained dLLMs as speculative drafters, with inference-time search or train–test alignment to improve acceptance. However, these approaches rely on massive drafters (e.g., 7B parameters), incurring substantial memory and latency overhead. While they achieve long acceptance lengths, the high drafting cost often offsets the practical speedups in real-world serving scenarios.

## 3. Preliminaries

This section formalizes the speedup mechanism of speculative decoding and clarifies the efficiency trade-offs between autoregressive and diffusion-based drafting. Our analysis highlights why diffusion drafters are uniquely positioned to achieve both low drafting latency and high acceptance rates.

## 3.1. Speculative Decoding Speedup

Speculative decoding accelerates inference of a target model $\mathcal { M } _ { t }$ using a smaller draft model $\mathcal { M } _ { d } .$ In each decoding cycle, the draft model proposes γ tokens, which are verified in parallel by the target model.

Following Sadhukhan et al. (2025), the average per-token

latency is

$$
L = \frac {T _ {\text { draft }} + T _ {\text { verify }}}{\tau}, \tag {1}
$$

where $T _ { \mathrm { d r a f t } }$ is the time spent generating draft tokens, $T _ { \mathrm { v e r i f y } }$ is the cost of verification, and $\tau \in [ 1 , \gamma + 1 ]$ is the expected number of accepted tokens per cycle, including the bonus token produced by the target model. Let $L _ { \mathrm { t a r g e t } }$ denote the autoregressive per-token latency of $\boldsymbol { \mathcal { M } } _ { t } ;$ the resulting speedup is $\eta = L _ { \mathrm { t a r g e t } } / L$ .

This expression makes the trade-off explicit: speedup improves either by increasing the expected acceptance length τ or by reducing the drafting overhead $T _ { \mathrm { d r a f t } }$ .

## 3.2. Autoregressive vs. Diffusion Drafting

Autoregressive drafters generate tokens sequentially, incurring a drafting cost

$$
T _ {\text { draft }} = \gamma \cdot t _ {\text { step }}, \tag {2}
$$

where $t _ { \mathrm { s t e p } }$ is the latency of a single forward pass. Drafting costs therefore grow linearly with the speculation budget γ.

To keep latency manageable, autoregressive drafters are constrained to very shallow architectures (e.g., a single transformer layer in EAGLE-3). This severely limits the draft quality: while increasing γ increases drafting cost, acceptance length τ quickly saturates due to limited model capacity. In practice, this imbalance restricts achievable speedups.

Diffusion drafters generate all γ tokens in parallel within a single forward pass, yielding

$$
T _ {\text { draft }} = t _ {\text { parallel }}, \tag {3}
$$

where $t _ { \mathrm { p a r a l l e l } }$ denotes the latency of block generation. Modern GPUs execute such parallel operations far more efficiently than multiple sequential passes, making $t _ { \mathrm { p a r a l l e l } } \ll$ $\gamma \cdot t _ { \mathrm { s t e p } }$ for models of comparable size. For moderate block sizes, $T _ { \mathrm { d r a f t } }$ is therefore largely insensitive to γ.

![](images/299a78c361d70f505bbff7dba6ec7e3f8f478b39f5b07943d3333d436d79876a.jpg)

<details>
<summary>bar chart</summary>

| Number of Draft Tokens | EAGLE-3 | DFlash (1) | DFlash (3) | DFlash (5) |
| ---------------------- | ------- | ---------- | ---------- | ---------- |
| 4                      | 7       | 2          | 4          | 5          |
| 8                      | 12      | 2          | 4          | 6          |
| 16                     | 25      | 2          | 4          | 6          |
</details>

Figure 3. Draft cost of 1, 3, 5-layer DFlash and 1-layer EAGLE-3.

This parallelism fundamentally changes the design space. Because drafting cost no longer scales with the number of generated tokens, diffusion drafters can afford deeper, more expressive architectures without sacrificing latency. This increased capacity substantially improves draft quality and acceptance length. Empirically, a five-layer DFlash draft model generating 16 tokens achieves both lower latency (Figure 3) and higher acceptance length than EAGLE-3 generating 8 tokens, placing DFlash on a more favorable Pareto frontier between draft quality and drafting cost.

![](images/4746e708cf7b61761490f9b2a73cf18528372b11c7862361c52d91135b2d722e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Diffusion is good"] --> B["Target Model"]
  B --> C["Target Embedding"]
  C --> D["KV Cache"]
  D --> E["Bidirectional Attention"]
  E --> F["MLP"]
  F --> G["Draft Layer 2"]
  G --> H["..."]
  H --> I["Target LM Head"]
  I --> J["for speculative decoding <eos>"]
    
    subgraph KV Cache
        K["Fused Target Context Feature"]
        L["Target Decode Token"]
        M["Mask Token"]
    end
    
    subgraph Draft Layer 1
        N["Fused Target Context Feature"]
        O["Target Decode Token"]
        P["Mask Token"]
    end
    
    style KV Cache fill:#f9f9f9,stroke-dasharray: 5 5
    style Draft Layer 1 fill:#e6f3ff,stroke:#333
    style Target Embedding fill:#e6f3ff,stroke:#333
    style Target LM Head fill:#e6f3ff,stroke:#333
```
</details>

Figure 2. DFlash Inference Design. Hidden context features extracted from the target model are fused and injected into each draft layer’s Key-Value cache to enable conditional speculation.

## 4. Method

## 4.1. Inference

The system design of DFlash is illustrated in Figure 2. In this section, we explain the key design choices that allow DFlash to achieve high draft acceptance length using a very small and efficient draft model.

Context features from the target model. Prior work like An et al. (2025) naively applied a small diffusion model as a speculative drafter, which leads to poor acceptance length and limited speedups. To validate this, we train a five-layer block diffusion draft model without any conditioning from the target model and evaluate it on several math benchmarks. As the results shown in the Table 10, the resulting speedups are modest, typically around 2–3×. This limitation stems from the lack of rich contextual guidance: without access to the internal representations of the target model, the diffusion drafter must effectively predict future tokens from scratch.

In contrast, the hidden representations of large autoregressive target models encode substantially more information than token-level logits. These features capture long-range dependencies and task-specific semantics, and—crucially—implicitly encode information about future token predictions, as also observed by Samragh et al. (2025).

In DFlash, given an input prompt, the target model first performs a standard prefill pass to generate the first token. During this pass, we extract hidden representations from a fixed set of layers uniformly sampled from shallow to deep. These hidden states are concatenated and passed through a lightweight projection layer to fuse cross-layer information into a compact target context feature, which is then used to condition the draft model.

Conditioning via KV injection enables acceptance scaling. Existing methods such as EAGLE-3 also leverage hidden features from the target model, but they fuse these features with the draft model’s token embeddings and feed them only as inputs to the draft model. As the draft model depth increases, the information from target model becomes more and more diluted, resulting in diminishing gains in acceptance length when adding more draft layers.

DFlash adopts a fundamentally different strategy. We treat the fused target context feature as persistent contextual information and directly inject it into the Key and Value projections of every draft model layer. The projected features are stored in the draft model’s KV cache and reused across drafting iterations. We provide more details about the KV injection mechanism in Section A.3. This design provides strong and consistent conditioning throughout the draft model, enabling acceptance length to scale effectively with the number of draft layers. We analyze this behavior in more detail in Section 5.5.2.

Parallel diffusion drafting. Another key contributor to DFlash’s speed is its low drafting latency. Autoregressive draft models must perform multiple sequential forward passes to generate draft tokens or trees, which limits parallelism and leads to inefficient GPU utilization. In contrast, DFlash predicts the next token block using a block-level diffusion process. All masked positions within a block are decoded in parallel in a single forward pass. Compared to autoregressive drafting, this block-wise parallel generation substantially reduces drafting latency and achieves significantly higher hardware utilization, even when using deeper draft models.

![](images/4878d8bbcd0b69acf0dbb498c080796db62b9c0916e8ef78f476ba3be8f7d52d.jpg)

<details>
<summary>heatmap</summary>

| Position | Target Context Feature | Mask Token | Clean Token | Invisible Token |
| --- | --- | --- | --- | --- |
| p1 | Yes | No | No | No |
| p2 | Yes | No | No | No |
| p3 | Yes | No | No | No |
| p4 | Yes | No | No | No |
| r1 | Yes | No | No | No |
| r2 | Yes | No | No | No |
| ... | ... | ... | ... | ... |
| r1 | Yes | Yes | Yes | Yes |
| <m> | Yes | Yes | Yes | Yes |
| <m> | Yes | Yes | Yes | Yes |
| <m> | Yes | Yes | Yes | Yes |
| r2 | Yes | Yes | Yes | Yes |
| <m> | Yes | Yes | Yes | Yes |
| <m> | Yes | Yes | Yes | Yes |
| <m> | Yes | Yes | Yes | Yes |
| r3 | Yes | Yes | Yes | Yes |
| <m> | Yes | Yes | Yes | Yes |
| <m> | Yes | Yes | Yes | Yes |
| ... | ... | ... | ... | ... |
</details>

Figure 4. DFlash training attention. The target model provides context features (blue) that condition the draft model. The input consists of clean prompt tokens p and clean response tokens r. Within each masked block, a subset of clean response tokens (yellow) is randomly sampled as anchors, while mask tokens m (green) mark positions for parallel prediction. Invisible tokens (white) denote the attention mask, which enforces causal consistency and prevents inter-block information leakage during training.

Overall, DFlash combines diffusion-based parallel drafting with tightly coupled conditioning from the target model, enabling high-quality drafting with substantially reduced drafting latency.

## 4.2. Training

DFlash draft models are trained to align block-level diffusion predictions with the outputs of a frozen autoregressive target model. Rather than directly adopting standard block diffusion training (Arriola et al., 2025), we introduce several key modifications that improve training efficiency, scalability, and alignment with the inference-time speculative decoding behavior.

KV injection. Following the inference pipeline, given a sequence consisting of a prompt and its response, we first pass the entire clean sequence through the target model to extract and fuse the hidden features for all tokens. The hidden features are then injected into the draft model as Key and Value projections, as illustrated in Figure 4.

Random sampling of masked blocks. In standard block diffusion training, the response is uniformly divided into blocks and random positions within each block are masked, with the model trained to denoise the masked tokens.

DFlash instead tailors block construction to the speculative decoding setting. We randomly sample anchor tokens from the response, use each anchor as the first position of a block, and mask the remaining positions. The draft model is trained to predict the next block size − 1 tokens in parallel. This directly matches inference-time behavior, where the draft model always conditions on a clean token produced by the target model (i.e., the bonus token from the previous verification step). Randomizing anchor positions also exposes the draft model to more diverse target context features, improving data efficiency and coverage. As shown in Table 13, this strategy substantially improves both acceptance length and speedup.

During training, all blocks are concatenated into a single sequence and processed jointly using a sparse attention mask as shown in Figure 4. Tokens attend bidirectionally within the same block and to the corresponding injected target context features, while attention across different blocks is disallowed. This design enables multiple draft blocks to be trained efficiently within a single forward and backward pass using Flex Attention (Dong et al., 2024).

Efficient long-context training. Training speculative draft models on long contexts is challenging for methods such as EAGLE-3 due to their costly training-time test. DFlash achieves efficient long-context training by fixing the number of masked blocks per sequence and randomly sampling anchor positions for each sequence at every epoch. This strategy provides effective data augmentation while keeping training cost bounded.

Loss weighting for faster convergence. In speculative decoding, not all tokens are equal. Errors at early positions within a draft block invalidate all subsequent tokens. This makes early predictions disproportionately important for acceptance length. We reflect this asymmetry by weighting the cross-entropy loss to emphasize earlier token positions during training.

Specifically, for a token at position k within a block, we apply an exponentially decaying weight

$$
w _ {k} = \exp \left(- \frac {k - 1}{\gamma}\right), \tag {4}
$$

where γ controls the decay rate. This weighting prioritizes early positions, accelerating convergence and yielding a higher acceptance length than uniform weighting (Figure 5).

Shared embedding and LM head. To improve training efficiency, the draft model shares the token embedding layer and language modeling head with the target model and keeps them frozen during training. Only the draft Transformer layers are updated. This design reduces the number of trainable parameters and encourages the draft model to function as a lightweight diffusion adapter tightly aligned with the target model’s representation space.

Table 1. Decoding speedup over baseline and average acceptance length (τ ) on Qwen3 models with thinking mode disabled and a maximum of 2048 generated tokens. Parenthesized values indicate the draft tree size for EAGLE-3 and the diffusion block size for DFlash.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="6">MATH</td><td colspan="5">CODE</td><td colspan="4">CHAT</td><td></td></tr><tr><td colspan="2">GSM8K</td><td colspan="2">MATH-500</td><td colspan="2">AIME25</td><td colspan="2">HumanEval</td><td colspan="2">MBPP</td><td colspan="2">LCB</td><td colspan="2">MT-Bench</td><td>Avg.</td><td></td></tr><tr><td colspan="2">Temperature = 0</td><td>Speedup</td><td> $\tau$ </td><td>Speedup</td><td> $\tau$ </td><td>Speedup</td><td> $\tau$ </td><td>Speedup</td><td> $\tau$ </td><td>Speedup</td><td> $\tau$ </td><td>Speedup</td><td> $\tau$ </td><td>Speedup</td><td> $\tau$ </td><td>Speedup</td><td> $\tau$ </td></tr><tr><td rowspan="3">Q3-4B</td><td>EAGLE-3 (16)</td><td>1.99×</td><td>3.30</td><td>1.83×</td><td>3.08</td><td>1.79×</td><td>3.05</td><td>1.84×</td><td>3.05</td><td>1.78×</td><td>2.95</td><td>1.73×</td><td>2.91</td><td>1.74×</td><td>3.02</td><td>1.81×</td><td>3.05</td></tr><tr><td>EAGLE-3 (60)</td><td>2.27×</td><td>3.77</td><td>2.10×</td><td>3.52</td><td>2.13×</td><td>3.51</td><td>2.12×</td><td>3.47</td><td>2.02×</td><td>3.38</td><td>1.90×</td><td>3.22</td><td>2.04×</td><td>3.49</td><td>2.08×</td><td>3.48</td></tr><tr><td>DFlash (16)</td><td>5.15×</td><td>6.53</td><td>6.09×</td><td>7.84</td><td>5.68×</td><td>7.27</td><td>5.21×</td><td>6.64</td><td>4.78×</td><td>6.09</td><td>5.41×</td><td>7.09</td><td>2.85×</td><td>4.35</td><td>4.91×</td><td>6.54</td></tr><tr><td rowspan="3">Q3-8B</td><td>EAGLE-3 (16)</td><td>1.94×</td><td>3.23</td><td>1.81×</td><td>3.02</td><td>1.79×</td><td>3.00</td><td>1.89×</td><td>3.17</td><td>1.69×</td><td>2.82</td><td>1.57×</td><td>2.65</td><td>1.63×</td><td>2.83</td><td>1.76×</td><td>2.96</td></tr><tr><td>EAGLE-3 (60)</td><td>2.23×</td><td>3.71</td><td>2.05×</td><td>3.49</td><td>2.05×</td><td>3.44</td><td>2.17×</td><td>3.65</td><td>1.93×</td><td>3.25</td><td>1.81×</td><td>3.03</td><td>1.90×</td><td>3.26</td><td>2.02×</td><td>3.40</td></tr><tr><td>DFlash (16)</td><td>5.15×</td><td>6.54</td><td>6.08×</td><td>7.87</td><td>5.62×</td><td>7.08</td><td>5.14×</td><td>6.50</td><td>4.65×</td><td>5.95</td><td>5.51×</td><td>7.27</td><td>2.75×</td><td>4.24</td><td>4.86×</td><td>6.49</td></tr><tr><td colspan="2">Temperature = 1</td><td>Speedup</td><td> $\tau$ </td><td>Speedup</td><td> $\tau$ </td><td>Speedup</td><td> $\tau$ </td><td>Speedup</td><td> $\tau$ </td><td>Speedup</td><td> $\tau$ </td><td>Speedup</td><td> $\tau$ </td><td>Speedup</td><td> $\tau$ </td><td>Speedup</td><td> $\tau$ </td></tr><tr><td rowspan="3">Q3-4B</td><td>EAGLE-3 (16)</td><td>1.89×</td><td>3.22</td><td>1.75×</td><td>2.99</td><td>1.64×</td><td>2.79</td><td>1.74×</td><td>3.01</td><td>1.69×</td><td>2.89</td><td>1.63×</td><td>2.77</td><td>1.70×</td><td>2.95</td><td>1.72×</td><td>2.95</td></tr><tr><td>EAGLE-3 (60)</td><td>2.12×</td><td>3.68</td><td>1.97×</td><td>3.44</td><td>1.83×</td><td>3.20</td><td>1.94×</td><td>3.39</td><td>1.92×</td><td>3.33</td><td>1.82×</td><td>3.14</td><td>1.91×</td><td>3.36</td><td>1.93×</td><td>3.36</td></tr><tr><td>DFlash (16)</td><td>4.71×</td><td>6.00</td><td>5.09×</td><td>6.67</td><td>3.73×</td><td>4.92</td><td>4.74×</td><td>6.04</td><td>4.42×</td><td>5.66</td><td>4.90×</td><td>6.50</td><td>2.67×</td><td>4.07</td><td>4.24×</td><td>5.69</td></tr><tr><td rowspan="3">Q3-8B</td><td>EAGLE-3 (16)</td><td>1.87×</td><td>3.12</td><td>1.73×</td><td>2.91</td><td>1.63×</td><td>2.74</td><td>1.75×</td><td>3.05</td><td>1.64×</td><td>2.74</td><td>1.56×</td><td>2.57</td><td>1.58×</td><td>2.70</td><td>1.68×</td><td>2.83</td></tr><tr><td>EAGLE-3 (60)</td><td>2.07×</td><td>3.59</td><td>1.94×</td><td>3.38</td><td>1.84×</td><td>3.18</td><td>2.05×</td><td>3.54</td><td>1.85×</td><td>3.16</td><td>1.72×</td><td>2.92</td><td>1.70×</td><td>3.05</td><td>1.88×</td><td>3.26</td></tr><tr><td>DFlash (16)</td><td>4.67×</td><td>5.98</td><td>4.84×</td><td>6.40</td><td>3.57×</td><td>4.73</td><td>4.32×</td><td>5.52</td><td>4.04×</td><td>5.21</td><td>4.93×</td><td>6.69</td><td>2.47×</td><td>3.80</td><td>4.03×</td><td>5.48</td></tr></table>

## 5. Experiments

Models and Evaluations. We conduct experiments on LLaMA-3.1 Instruct (8B) and Qwen3 (4B, 8B, Coder-30B-A3B-Instruct) pre-trained models. We evaluate tasks in three categories: Math: GSM8K (Cobbe et al., 2021), MATH (Lightman et al., 2023), and AIME25 (MAA, 2025); Code: HumanEval (Chen, 2021), MBPP (Austin et al., 2021), and LiveCodeBench (Jain et al., 2024); Chat: MT-Bench (Zheng et al., 2023) and Alpaca (Taori et al., 2023). For each task, we assess the performance of the draft models using average acceptance length (τ ) and end-to-end decoding speedup over the autoregressive baseline. We conduct all experiments on NVIDIA H200 GPUs unless otherwise specified.

Datasets. To provide a diverse set of training data, we collect a mixture of around 800K samples from NVIDIA Nemotron Post-Training Dataset V2 (Nathawani et al., 2025) and CodeAlpaca (Chaudhary, 2023). Instead of directly using the original dataset, we construct our training set with the responses generated by the target model for better target alignment.

Implementation. For DFlash draft models, we set the number of layers to 5 (8 for Qwen3 Coder) and use a block size of 16 (10 for LLaMA 3.1). The target hidden features are extracted from 5 layers uniformly selected between the second layer and the third-to-last layer of the target model.

More training details are presented in Section A.1.

Baselines. We compare DFlash with the vanilla autoregressive decoding (baseline) and state-of-the-art speculative decoding method EAGLE-3 (Li et al., 2025b). We did not include comparisons with other dLLM-based speculative decoding methods (Liu et al., 2025; Samragh et al., 2025; Li et al., 2025a; Sandler et al., 2025) due to lack of opensource implementation. For comparisons with EAGLE-3 on Qwen3 models (Section 5.1), we use the checkpoints released by AngelSlim (Tencent, 2025); for LLaMA-3.1- Instruct (Section 5.5.1), we use the official checkpoint released by EAGLE-3 team.

## 5.1. Instruct Models

In this section, we evaluate DFlash against EAGLE-3 on Qwen3 models with thinking mode disabled, using the Transformers backend. For EAGLE-3, we consider two settings: a tree size of 16, which matches DFlash with block size 16 for a fair drafting-budget comparison, and a tree size of 60, as used in the EAGLE-3 paper to maximize acceptance length with higher verification cost. In both cases, the draft steps and top-k are set to 7 and 10, respectively.

As shown in Table 1, DFlash consistently outperforms EAGLE-3 across all tasks and settings. Under greedy decoding (temperature = 0), DFlash achieves an average speedup of 4.9× over the autoregressive baseline, corresponding to a 2.4× improvement over EAGLE-3 (16). Under nongreedy sampling (temperature = 1), DFlash maintains a 4.1× speedup over baseline and a 2.2× improvement over EAGLE-3. Notably, DFlash also surpasses EAGLE-3 with tree size 60, achieving higher acceptance length while incurring substantially lower verification overhead. These results demonstrate the effectiveness and efficiency of diffusionbased drafting in DFlash.

## 5.2. Reasoning Models

In this section, we evaluate DFlash for Qwen3 models with thinking mode enabled using Transformers. The draft models are trained on target-model outputs with reasoning traces.

As shown in Table 2, DFlash maintains the high acceptance length, achieving speedups of roughly 4.5× and 3.9× over the baseline. This efficiency gain is particularly valuable for the practical deployment of reasoning models, given their prolonged generation time.

Table 2. Decoding speedup over baseline and average acceptance length (τ ) with thinking mode enabled.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Temp.</td><td colspan="2">GPQA</td><td colspan="2">MATH-500</td><td colspan="2">AIME25</td></tr><tr><td>Speedup</td><td> $\tau$ </td><td>Speedup</td><td> $\tau$ </td><td>Speedup</td><td> $\tau$ </td></tr><tr><td rowspan="2">Q3-4B</td><td>0</td><td>4.23 $\times$ </td><td>5.23</td><td>4.59 $\times$ </td><td>5.74</td><td>4.39 $\times$ </td><td>5.54</td></tr><tr><td>1</td><td>3.67 $\times$ </td><td>4.55</td><td>3.93 $\times$ </td><td>4.89</td><td>3.64 $\times$ </td><td>4.68</td></tr><tr><td rowspan="2">Q3-8B</td><td>0</td><td>4.17 $\times$ </td><td>5.17</td><td>4.64 $\times$ </td><td>5.82</td><td>4.51 $\times$ </td><td>5.74</td></tr><tr><td>1</td><td>3.75 $\times$ </td><td>4.65</td><td>4.03 $\times$ </td><td>5.06</td><td>3.70 $\times$ </td><td>4.69</td></tr></table>

## 5.3. Performance on Serving Frameworks

In this section, we evaluate the performance of DFlash on the popular open-source inference framework SGLang using Qwen3-4B, Qwen3-8B, and Qwen3-Coder-30B-A3B-Instruct. All experiments are conducted on a single B200 GPU with the FlashAttention-4 (FA4) backend. We enable Spec-v2 scheduling overlap to maximize achievable throughput.

As shown in Table 3, DFlash consistently provides speedups across all three models over concurrency levels ranging from 1 to 32, achieving up to a 5.1× speedup on Qwen3- 8B. These results demonstrate the practical value of DFlash in real-world serving scenarios, where it can substantially reduce serving cost.

We report additional DFlash speedup results for more models and for vLLM (Kwon et al., 2023) in Section A.4.

## 5.4. Long Context Adaptation

In this section, we show that the base DFLash draft models trained on 4K context can adapt to longer context with minimal fine-tuning. We fine-tune the base Qwen3.5-27B draft model with 1.6K samples from LongAlign-10K (Bai et al., 2024a) for 3 epochs and test the performance on several datasets from LongBench (Bai et al., 2024b).

As shown in Table 4, the base DFlash draft model degrades as context length grows beyond 4K, while the fine-tuned model maintains or even improves the acceptance length.

Table 3. Throughput (tok/s), speedup over baseline, and average acceptance length τ on SGLang (FA4 backend).

<table><tr><td rowspan="2">Task</td><td rowspan="2">Method</td><td colspan="5">Concurrency</td><td rowspan="2">Avg.  $\tau$ </td></tr><tr><td>1</td><td>4</td><td>8</td><td>16</td><td>32</td></tr><tr><td colspan="8">Qwen3-4B</td></tr><tr><td rowspan="2">Math500</td><td>Baseline</td><td>316</td><td>1145</td><td>2201</td><td>4100</td><td>7136</td><td>-</td></tr><tr><td>DFlash</td><td>1531 $4.8 \times$ </td><td>4943 $4.3 \times$ </td><td>9066 $4.1 \times$ </td><td>14477 $3.5 \times$ </td><td>20417 $2.9 \times$ </td><td>8.01</td></tr><tr><td rowspan="2">Human-Eval</td><td>Baseline</td><td>312</td><td>1162</td><td>2217</td><td>4184</td><td>7143</td><td>-</td></tr><tr><td>DFlash</td><td>1247 $4.0 \times$ </td><td>4147 $3.6 \times$ </td><td>6997 $3.2 \times$ </td><td>11234 $2.7 \times$ </td><td>15703 $2.2 \times$ </td><td>6.63</td></tr><tr><td colspan="8">Qwen3-8B</td></tr><tr><td rowspan="2">Math500</td><td>Baseline</td><td>230</td><td>861</td><td>1666</td><td>3133</td><td>5694</td><td>-</td></tr><tr><td>DFlash</td><td>1175 $5.1 \times$ </td><td>3884 $4.5 \times$ </td><td>7485 $4.5 \times$ </td><td>12268 $3.9 \times$ </td><td>16076 $2.8 \times$ </td><td>8.01</td></tr><tr><td rowspan="2">Human-Eval</td><td>Baseline</td><td>229</td><td>868</td><td>1649</td><td>3253</td><td>5462</td><td>-</td></tr><tr><td>DFlash</td><td>955 $4.2 \times$ </td><td>3092 $3.6 \times$ </td><td>6010 $3.6 \times$ </td><td>9919 $3.0 \times$ </td><td>13116 $2.4 \times$ </td><td>6.50</td></tr><tr><td colspan="8">Qwen3-Coder-30B-A3B</td></tr><tr><td rowspan="2">Human-Eval</td><td>Baseline</td><td>229</td><td>686</td><td>1068</td><td>1681</td><td>2713</td><td>-</td></tr><tr><td>DFlash</td><td>802 $3.5 \times$ </td><td>2078 $3.0 \times$ </td><td>3442 $3.2 \times$ </td><td>5429 $3.2 \times$ </td><td>8314 $3.1 \times$ </td><td>8.09</td></tr><tr><td rowspan="2">LCB</td><td>Baseline</td><td>220</td><td>681</td><td>1112</td><td>1733</td><td>2823</td><td>-</td></tr><tr><td>DFlash</td><td>569 $2.6 \times$ </td><td>1621 $2.4 \times$ </td><td>2554 $2.3 \times$ </td><td>4160 $2.4 \times$ </td><td>6401 $2.3 \times$ </td><td>6.42</td></tr><tr><td rowspan="2">MBPP</td><td>Baseline</td><td>228</td><td>682</td><td>1057</td><td>1697</td><td>2735</td><td>-</td></tr><tr><td>DFlash</td><td>720 $3.2 \times$ </td><td>2052 $3.0 \times$ </td><td>3360 $3.2 \times$ </td><td>5522 $3.3 \times$ </td><td>8538 $3.1 \times$ </td><td>7.23</td></tr></table>

This demonstrates that the extracted target features remain representative at long contexts and the draft model can learn the longer-range patterns with lightweight adaptation.

## 5.5. Ablation Study

In this section, we ablate the impact of training data and several key design choices of the DFlash draft model. Unless otherwise specified, all ablation models are trained on 100K samples randomly drawn from the full data mixture. All experiments are conducted on a single H200 GPU with greedy decoding, except those evaluated on SGLang.

## 5.5.1. TRAINING DATA

We compare DFlash against EAGLE-3 on LLaMA-3.1- 8B-Instruct. DFlash is trained on UltraChat (Ding et al., 2023) and ShareGPT, using the exactly same training data as EAGLE-3, and is evaluated against the official EAGLE-3 checkpoints. The DFlash draft model uses a block size of 10, with other configurations matching those of the DFlash Qwen3-8B draft model. All experiments are conducted using SGLang with Spec-v1 (without scheduling overlap), as

Table 4. Acceptance length of the base Qwen3.5-27B DFlash drafter (Base) and the drafter fine-tuned for long context (Long) across various context lengths on LongBench.

<table><tr><td rowspan="2">Context</td><td colspan="2">hotpotqa</td><td colspan="2">qasper</td><td colspan="2">gov_report</td></tr><tr><td>Base</td><td>Long</td><td>Base</td><td>Long</td><td>Base</td><td>Long</td></tr><tr><td>1K</td><td>4.91</td><td>4.99</td><td>5.27</td><td>5.38</td><td>4.53</td><td>4.53</td></tr><tr><td>2K</td><td>4.97</td><td>5.06</td><td>5.50</td><td>5.67</td><td>4.35</td><td>4.38</td></tr><tr><td>4K</td><td>4.91</td><td>5.41</td><td>5.17</td><td>5.80</td><td>3.93</td><td>4.25</td></tr><tr><td>8K</td><td>4.46</td><td>5.76</td><td>4.17</td><td>5.62</td><td>3.32</td><td>4.04</td></tr><tr><td>16K</td><td>3.61</td><td>6.05</td><td>3.57</td><td>6.00</td><td>2.67</td><td>3.81</td></tr><tr><td>32K</td><td>-</td><td>-</td><td>-</td><td>-</td><td>2.09</td><td>3.56</td></tr></table>

Table 5. Speedup over baseline and average acceptance length τ for LLaMA-3.1-8B-Instruct on SGLang (Flashinfer backend, single B200 GPU). Baseline reports absolute throughput (TPS; tokens/s). EAGLE-3 uses 7 draft steps with top-k=10 and either 10 or 60 draft tokens. DFlash uses block size 10.

<table><tr><td rowspan="2">Method</td><td colspan="5">Concurrency</td><td rowspan="2">Avg.  $\tau$ </td></tr><tr><td>1</td><td>4</td><td>8</td><td>16</td><td>32</td></tr><tr><td colspan="7">GSM8K</td></tr><tr><td>Baseline (TPS)</td><td>249</td><td>923</td><td>1739</td><td>3245</td><td>5349</td><td>-</td></tr><tr><td>EAGLE-3 (10)</td><td>1.6×</td><td>1.5×</td><td>1.4×</td><td>1.2×</td><td>1.0×</td><td>3.49</td></tr><tr><td>EAGLE-3 (60)</td><td>1.9×</td><td>1.6×</td><td>1.3×</td><td>0.9×</td><td>0.6×</td><td>4.55</td></tr><tr><td>DFlash (10)</td><td>2.4×</td><td>2.2×</td><td>2.1×</td><td>1.8×</td><td>1.6×</td><td>4.32</td></tr><tr><td colspan="7">HumanEval</td></tr><tr><td>Baseline (TPS)</td><td>245</td><td>922</td><td>1778</td><td>3336</td><td>5854</td><td>-</td></tr><tr><td>EAGLE-3 (10)</td><td>2.0×</td><td>1.9×</td><td>1.8×</td><td>1.5×</td><td>1.2×</td><td>3.62</td></tr><tr><td>EAGLE-3 (60)</td><td>2.0×</td><td>1.7×</td><td>1.3×</td><td>0.9×</td><td>0.6×</td><td>4.65</td></tr><tr><td>DFlash (10)</td><td>2.8×</td><td>2.6×</td><td>2.5×</td><td>2.1×</td><td>1.8×</td><td>4.91</td></tr><tr><td colspan="7">Alpaca</td></tr><tr><td>Baseline (TPS)</td><td>245</td><td>906</td><td>1745</td><td>3237</td><td>5434</td><td>-</td></tr><tr><td>EAGLE-3 (10)</td><td>1.5×</td><td>1.4×</td><td>1.4×</td><td>1.1×</td><td>0.9×</td><td>3.11</td></tr><tr><td>EAGLE-3 (60)</td><td>1.8×</td><td>1.5×</td><td>1.2×</td><td>0.8×</td><td>0.5×</td><td>4.07</td></tr><tr><td>DFlash (10)</td><td>2.2×</td><td>2.0×</td><td>1.8×</td><td>1.5×</td><td>1.4×</td><td>3.73</td></tr></table>

Spec-v2 does not support tree-based drafting for EAGLE-3. Evaluations are performed on a single B200 GPU.

As shown in Table 5, DFlash consistently outperforms EAGLE-3 across all tasks, concurrency levels, and EAGLE-3 tree-size configurations. This performance gap holds for math, code, and chat benchmarks, demonstrating the robustness and efficiency advantages of DFlash over autoregressive tree-based speculative decoding.

## 5.5.2. NUMBER OF DRAFT LAYERS

One advantage of DFlash is that acceptance length scales effectively with the depth of the draft model. However, this comes with a trade-off between drafting cost and draft quality. Deeper draft models are more expressive and achieve higher acceptance lengths, but they also incur higher drafting latency. As a result, the optimal number of layers depends on the deployment setting. As shown in Table 6, while the 8-layer draft model achieves longer acceptance lengths, the 5-layer model attains higher overall speedup due to a better balance between drafting cost and quality.

Table 6. 5-layer draft model has the best average speedup. All DFlash draft models are trained with block size 16 and hidden features extracted from 5 layers of the target model.

<table><tr><td rowspan="2">Setting</td><td colspan="2">Math500</td><td colspan="2">HumanEval</td><td colspan="2">MT-Bench</td></tr><tr><td>Speedup</td><td> $\tau$ </td><td>Speedup</td><td> $\tau$ </td><td>Speedup</td><td> $\tau$ </td></tr><tr><td>3-L</td><td>4.69 $\times$ </td><td>5.64</td><td>3.90 $\times$ </td><td>4.61</td><td>2.38 $\times$ </td><td>3.18</td></tr><tr><td>5-L</td><td>4.71 $\times$ </td><td>5.99</td><td>3.96 $\times$ </td><td>4.94</td><td>2.35 $\times$ </td><td>3.37</td></tr><tr><td>8-L</td><td>4.64 $\times$ </td><td>6.33</td><td>3.96 $\times$ </td><td>5.29</td><td>2.23 $\times$ </td><td>3.50</td></tr></table>

## 5.5.3. NUMBER OF TARGET HIDDEN FEATURES

Table 7. More hidden features from target model increases the acceptance length. All DFlash draft models use 3 draft layers and are trained with block size 16.

<table><tr><td rowspan="2">Setting</td><td colspan="2">Math500</td><td colspan="2">HumanEval</td><td colspan="2">MT-Bench</td></tr><tr><td>Speedup</td><td> $\tau$ </td><td>Speedup</td><td> $\tau$ </td><td>Speedup</td><td> $\tau$ </td></tr><tr><td>3-H</td><td>4.49 $\times$ </td><td>5.38</td><td>3.80 $\times$ </td><td>4.47</td><td>2.32 $\times$ </td><td>3.07</td></tr><tr><td>5-H</td><td>4.69 $\times$ </td><td>5.64</td><td>3.90 $\times$ </td><td>4.61</td><td>2.38 $\times$ </td><td>3.18</td></tr></table>

The number of target hidden features affects both acceptance length and end-to-end speedup. Extracting features from more target layers provides richer semantic and futuretoken information, improving draft quality. As shown in Table 7, conditioning on five hidden features consistently outperforms using three. However, this benefit comes at higher training cost: in offline training, the storage required to cache target hidden states increases linearly with the number of extracted features.

## 5.5.4. TRAINING-INFERENCE TIME BLOCK SIZE

Table 8. Ablation study of training–inference block size (BS) mismatch. All draft models use 8 layers and 5 target hidden features.

<table><tr><td rowspan="2">Train BS</td><td rowspan="2">Test BS</td><td colspan="2">Math500</td><td colspan="2">HumanEval</td><td colspan="2">MT-Bench</td></tr><tr><td>Speedup</td><td>τ</td><td>Speedup</td><td>τ</td><td>Speedup</td><td>τ</td></tr><tr><td>b16</td><td>b16</td><td>4.64x</td><td>6.33</td><td>3.96x</td><td>5.29</td><td>2.23x</td><td>3.50</td></tr><tr><td>b16</td><td>b8</td><td>3.87x</td><td>5.09</td><td>3.39x</td><td>4.44</td><td>2.12x</td><td>3.18</td></tr><tr><td>b8</td><td>b16</td><td>3.78x</td><td>5.02</td><td>3.24x</td><td>4.28</td><td>2.09x</td><td>3.09</td></tr><tr><td>b8</td><td>b8</td><td>3.97x</td><td>5.21</td><td>3.53x</td><td>4.61</td><td>2.22x</td><td>3.29</td></tr></table>

Block size is a critical design choice for the DFlash draft model. An equally important question is whether a pretrained DFlash model can generalize from its training-time block size to different block sizes during inference. To study this, we train two draft models with block sizes 8 and 16 on the same data and evaluate their inference-time scaling behavior, as shown in Table 8.

When training and inference block sizes match (8→8 and 16→16), the block-size-16 model achieves substantially higher acceptance lengths on math and coding tasks. Acceptance histograms on Math500 reveal that the block-8 model frequently fully accepts entire blocks (35.7%), suggesting that block size 8 is often underutilized. In contrast, the block-16 model exhibits a more spread-out acceptance distribution with higher average acceptance length, indicating more effective use of larger blocks.

We further examine cross-block-size generalization at inference time and observe a clear asymmetry. A model trained with a larger block size generalizes well to smaller inferencetime block sizes: using block size 8 with a model trained at block size 16 yields acceptance lengths close to those of a model trained and evaluated at block size 8. However, the reverse does not hold.

Overall, DFlash models trained with larger block sizes generalize well to smaller inference-time block sizes. This property enables dynamic block-size scheduling during inference to improve end-to-end efficiency. In practical serving scenarios, large blocks can increase verification cost under compute-bound settings (e.g., large batch sizes); reducing the block size in such cases can therefore yield better overall speedup. We leave adaptive block-size scheduling to future work.

## 5.5.5. KV INJECTION VS. INPUT FUSION

This ablation studies whether target features should be injected only once at the input layer, as in EAGLE-3 style input fusion, or injected into every draft layer as KV entries. We compare these two conditioning strategies under both autoregressive drafting and block-diffusion drafting. Results are shown in Table 9.

Table 9. Ablation of target-feature conditioning for Qwen3-4B with 5-layer draft models and draft block size 8. Each task column reports τ / speedup.

<table><tr><td>Variant</td><td>Injection</td><td>GSM8K</td><td>HumanEval</td><td>MT-Bench</td></tr><tr><td colspan="5">Autoregressive drafting</td></tr><tr><td>EAGLE-3-5L</td><td>Input</td><td>4.2 / 2.1×</td><td>4.3 / 2.2×</td><td>3.1 / 1.4×</td></tr><tr><td>DFlash-AR</td><td>KV</td><td>4.8 / 2.4×</td><td>4.6 / 2.3×</td><td>3.4 / 1.5×</td></tr><tr><td colspan="5">Block-diffusion drafting</td></tr><tr><td>DFlash</td><td>Input</td><td>3.5 / 2.9×</td><td>3.5 / 2.9×</td><td>2.6 / 2.0×</td></tr><tr><td>DFlash</td><td>KV</td><td>4.2 / 3.3×</td><td>4.0 / 3.2×</td><td>3.0 / 2.2×</td></tr></table>

The results show that KV injection is more effective than input fusion. In autoregressive drafting, DFlash-AR achieves higher acceptance length than EAGLE-3-5L on all tasks. In block-diffusion drafting, DFlash with KV injection also improves acceptance length over DFlash with input fusion on all tasks. This suggests that exposing every draft layer to target features through KV entries is more effective than injecting target features only at the input layer.

DFlash achieves acceptance length comparable to EAGLE-3-5L, but obtains much higher speedup because block diffusion drafts multiple tokens in parallel. Therefore, DFlash benefits from both stronger conditioning through KV injection and faster parallel drafting.

## 6. Conclusion

In this paper, we present DFlash, a diffusion-based speculative decoding framework that rethinks the role of diffusion language models in accelerating autoregressive LLM inference. By confining diffusion models to the drafting stage, DFlash exploits their inherent parallelism while avoiding the quality degradation that has limited their standalone use. Conditioning the diffusion drafter on rich target-model context enables high acceptance rates, allowing DFlash to significantly push inference speed beyond prior speculative decoding methods.

Beyond empirical improvements, DFlash suggests a new development paradigm for diffusion LLMs. Rather than competing with autoregressive models in end-to-end generation quality, diffusion models can serve as lightweight, specialized drafters optimized for fast and accurate block prediction. This reframing permits aggressive reduction in denoising steps to maximize parallelism, while speculative verification provides a principled guarantee of output quality. We hope DFlash establishes diffusion-based drafting as a practical and scalable paradigm for speculative decoding, advancing more efficient and accessible LLM deployment.

## Acknowledgements

The authors would like to express their sincere gratitude to David Wang for leading the fast and high-quality SGLang integration for DFlash, and to Richard Gong and other members of the Modal team for their strong engineering support. Their efforts were truly instrumental in enabling the practical, production-grade deployment of DFlash.

We thank Qualcomm and Amazon for their support of this research. We also acknowledge Modal, Yotta Labs, Eigen AI, and InnoMatrix for providing the compute resources that made this work possible.

## Impact Statement

This paper presents work whose goal is to advance the efficiency of LLM inference through improved speculative decoding. The proposed method is primarily a system- and algorithm-level optimization that reduces LLM inference and serving costs, without altering model capabilities or intended use cases. We do not foresee significant new ethical risks beyond those already associated with large language models in general. Potential societal impacts are therefore consistent with existing deployments of LLMs, including both their benefits and known limitations.

## References

Agarwal, S., Ahmad, L., Ai, J., Altman, S., Applebaum, A., Arbus, E., Arora, R. K., Bai, Y., Baker, B., Bao, H., et al. gpt-oss-120b & gpt-oss-20b model card. arXiv preprint arXiv:2508.10925, 2025. 12  
An, Z., Bai, H., Liu, Z., Li, D., and Barsoum, E. Pard: Accelerating llm inference with low-cost parallel draft model adaptation, 2025. URL https://arxiv.org/abs/2504. 18583. 2, 4  
Arriola, M., Gokaslan, A., Chiu, J. T., Yang, Z., Qi, Z., Han, J., Sahoo, S. S., and Kuleshov, V. Block diffusion: Interpolating between autoregressive and diffusion language models, 2025. URL https://arxiv.org/abs/ 2503.09573. 1, 3, 5  
Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021. 6  
Bai, Y., Lv, X., Zhang, J., He, Y., Qi, J., Hou, L., Tang, J., Dong, Y., and Li, J. Longalign: A recipe for long context alignment of large language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, pp. 1376–1395, 2024a. 7  
Bai, Y., Lv, X., Zhang, J., Lyu, H., Tang, J., Huang, Z., Du, Z., Liu, X., Zeng, A., Hou, L., et al. Longbench: A bilingual, multitask benchmark for long context understanding. In Proceedings of the 62nd annual meeting of the association for computational linguistics (volume 1: Long papers), pp. 3119–3137, 2024b. 7  
Cai, T., Li, Y., Geng, Z., Peng, H., Lee, J. D., Chen, D., and Dao, T. Medusa: Simple llm inference acceleration framework with multiple decoding heads, 2024. URL https://arxiv.org/abs/2401.10774. 1, 2  
Chaudhary, S. Code alpaca: An instruction-following llama model for code generation. https://github.com/ sahil280114/codealpaca, 2023. 6  
Chen, M. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021. 6  
Cheng, S., Bian, Y., Liu, D., Zhang, L., Yao, Q., Tian, Z., Wang, W., Guo, Q., Chen, K., Qi, B., and Zhou, B.

Sdar: A synergistic diffusion-autoregression paradigm for scalable sequence generation, 2025. URL https: //arxiv.org/abs/2510.06303. 1, 3

Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021. 6

Ding, N., Chen, Y., Xu, B., Qin, Y., Zheng, Z., Hu, S., Liu, Z., Sun, M., and Zhou, B. Enhancing chat language models by scaling high-quality instructional conversations, 2023. URL https://arxiv.org/abs/2305.14233. 7

Dong, J., Feng, B., Guessous, D., Liang, Y., and He, H. Flex attention: A programming model for generating optimized attention kernels, 2024. URL https://arxiv. org/abs/2412.05496. 5

Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783. 3

Guo, D., Yang, D., Zhang, H., Song, J., Wang, P., Zhu, Q., Xu, R., Zhang, R., Ma, S., Bi, X., et al. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638, September 2025. ISSN 1476-4687. doi: 10.1038/ s41586-025-09422-z. URL http://dx.doi.org/10. 1038/s41586-025-09422-z. 1

Jain, N., Han, K., Gu, A., Li, W.-D., Yan, F., Zhang, T., Wang, S., Solar-Lezama, A., Sen, K., and Stoica, I. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024. 6

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th symposium on operating systems principles, pp. 611–626, 2023.

Leviathan, Y., Kalman, M., and Matias, Y. Fast inference from transformers via speculative decoding, 2023. URL https://arxiv.org/abs/2211.17192. 1, 2

Li, G., Fu, Z., Fang, M., Zhao, Q., Tang, M., Yuan, C., and Wang, J. Diffuspec: Unlocking diffusion language models for speculative decoding, 2025a. URL https: //arxiv.org/abs/2510.02358. 2, 3, 6

Li, Y., Wei, F., Zhang, C., and Zhang, H. Eagle-2: Faster inference of language models with dynamic draft trees, 2024. URL https://arxiv.org/abs/2406.16858. 1, 2

Li, Y., Wei, F., Zhang, C., and Zhang, H. Eagle-3: Scaling up inference acceleration of large language models via training-time test, 2025b. URL https://arxiv.org/ abs/2503.01840. 1, 2, 6  
Li, Y., Wei, F., Zhang, C., and Zhang, H. Eagle: Speculative sampling requires rethinking feature uncertainty, 2025c. URL https://arxiv.org/abs/2401.15077. 1, 2  
Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let’s verify step by step. arXiv preprint arXiv:2305.20050, 2023. 6  
Liu, J., Dong, X., Ye, Z., Mehta, R., Fu, Y., Singh, V., Kautz, J., Zhang, C., and Molchanov, P. Tidar: Think in diffusion, talk in autoregression, 2025. URL https: //arxiv.org/abs/2511.08923. 3, 6  
MAA. American Invitational Mathematics Examination - AIME, 2025. URL https://maa.org/math-competitions/ american-invitational-mathematics-examination-6  
Nathawani, D., Ding, S., Lavrukhin, V., Gitman, I., Majumdar, S., Bakhturina, E., Ginsburg, B., and Polak Scowcroft, J. Nemotron-Post-Training-Dataset-v2, August 2025. URL https://huggingface.co/datasets/nvidia/ Nemotron-Post-Training-Dataset-v2. 6  
Nie, S., Zhu, F., You, Z., Zhang, X., Ou, J., Hu, J., Zhou, J., Lin, Y., Wen, J.-R., and Li, C. Large language diffusion models, 2025. URL https://arxiv.org/abs/ 2502.09992. 1, 3  
OpenAI, Jaech, A., Kalai, A., Lerer, A., Richardson, A., El-Kishky, A., Low, A., Helyar, A., Madry, A., Beutel, A., et al. Openai o1 system card, 2024. URL https: //arxiv.org/abs/2412.16720. 1  
Qian, Y.-Y., Su, J., Hu, L., Zhang, P., Deng, Z., Zhao, P., and Zhang, H. d3llm: Ultra-fast diffusion llm using pseudotrajectory distillation, 2026. URL https://arxiv.org/ abs/2601.07568. 1  
Sadhukhan, R., Chen, J., Chen, Z., Tiwari, V., Lai, R., Shi, J., Yen, I. E.-H., May, A., Chen, T., and Chen, B. Magicdec: Breaking the latency-throughput tradeoff for long context generation with speculative decoding, 2025. URL https://arxiv.org/abs/2408.11049. 3  
Samragh, M., Kundu, A., Harrison, D., Nishu, K., Naik, D., Cho, M., and Farajtabar, M. Your llm knows the future: Uncovering its multi-token prediction potential, 2025. URL https://arxiv.org/abs/2507.11851. 2, 3, 4, 6

Sandler, J., Christopher, J. K., Hartvigsen, T., and Fioretto, F. Specdiff-2: Scaling diffusion drafter alignment for faster speculative decoding, 2025. URL https://arxiv. org/abs/2511.00606. 2, 3, 6

Taori, R., Gulrajani, I., Zhang, T., Dubois, Y., Li, X., Guestrin, C., Liang, P., and Hashimoto, T. B. Stanford alpaca: An instruction-following llama model. https: //github.com/tatsu-lab/stanford alpaca, 2023. 6

Tencent. AngelSlim, 6 2025. URL https://github.com/ Tencent/AngelSlim. 6

Wu, C., Zhang, H., Xue, S., Diao, S., Fu, Y., Liu, Z., Molchanov, P., Luo, P., Han, S., and Xie, E. Fastdllm v2: Efficient block-diffusion llm, 2025. URL https://arxiv.org/abs/2509.26328. 1, 3

Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report, 2025. URL https://arxiv.org/abs/ 2505.09388. 1, 2

Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E., et al. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems, 36: 46595–46623, 2023. 6

Zheng, L., Yin, L., Xie, Z., Sun, C., Huang, J., Yu, C. H., Cao, S., Kozyrakis, C., Stoica, I., Gonzalez, J. E., Barrett, C., and Sheng, Y. Sglang: Efficient execution of structured language model programs, 2024. URL https://arxiv.org/abs/2312.07104. 2

## A. Appendix

## A.1. Training Implementation

The draft models are optimized for 6 epochs using AdamW with a learning rate of $6 \times 1 0 ^ { - 4 } .$ , a gradient clipping threshold of 1.0, and a cosine schedule with a warmup ratio of 0.04. We train on our training data mixture with a maximum sequence length of 3072 tokens (4096 for Qwen3-Coder); for each sequence, 512 anchor positions are randomly sampled. The hyperparameter γ for the loss decay in Equation 4 is set to 7 for block size 16, 5 for block size 10, and 4 for block size 8 models.

Training can be performed either online or offline. In online training, target hidden features are computed on the fly during each training step. In offline training, target hidden features are precomputed and cached, then loaded during draft model optimization to reduce computational overhead.

## A.2. Diffusion Drafter without Target Feature

Table 10. A 5-layer block diffusion draft model without target context features. The draft model achieves only modest acceptance length and speedup.

<table><tr><td>Temp</td><td>GSM8K Speedup /  $\tau$ </td><td>Math500 Speedup /  $\tau$ </td><td>AIME24 Speedup /  $\tau$ </td><td>AIME25 Speedup /  $\tau$ </td></tr><tr><td>0</td><td>2.83 / 3.38</td><td>3.73 / 4.61</td><td>3.43 / 4.12</td><td>3.35 / 4.07</td></tr><tr><td>1</td><td>2.76 / 3.29</td><td>3.31 / 4.12</td><td>2.66 / 3.23</td><td>2.65 / 3.24</td></tr></table>

## A.3. KV Injection Mechanism and Memory Overhead

DFlash uses KV injection to condition the diffusion drafter on target-model features. We first concatenate hidden states from selected target layers and project them once into the draft hidden dimension:

$$
\mathbf {H} _ {t} = \operatorname{RMSNorm} \left(W _ {c} [ \mathbf {H} ^ {(l _ {1})}; \dots ; \mathbf {H} ^ {(l _ {5})} ]\right).
$$

The projected target features are shared by all draft layers. At layer i, draft tokens produce queries, while both target features and draft tokens are projected into keys and values:

$$
\mathbf {Q} _ {i} = W _ {i} ^ {Q} \mathbf {H} _ {d},
$$

$$
\mathbf {K} _ {i} = [ W _ {i} ^ {K} \mathbf {H} _ {t}; W _ {i} ^ {K} \mathbf {H} _ {d} ] _ {\mathrm{seq}},
$$

$$
\mathbf {V} _ {i} = [ W _ {i} ^ {V} \mathbf {H} _ {t}; W _ {i} ^ {V} \mathbf {H} _ {d} ] _ {\mathrm{seq}}.
$$

Thus, target features only serve as additional KV entries for the masked-block draft tokens. They bypass the draft model’s $Q$ projection, output projection, self-attention update, and FFN.

The memory overhead is small. The only extra parameterized component is the shared projection $\dot { W } _ { c } \in \mathbb { R } ^ { \dot { D } \times 5 D }$ . For Qwen3.5-35B-A3B with D = 2048 and BF16 weights, this

adds

$$
5 \times 2 0 4 8 \times 2 0 4 8 \times 2 \approx 4 2 \mathrm{MB},
$$

which is negligible compared with the roughly 70 GB target model. The activation overhead is also modest: for batch size 1 and sequence length 2048, the projection input and output require about 40 MB and 8 MB, respectively. During decoding with block size 16, the temporary activation is below 400 KB.

## A.4. Results on More Models and vLLM

We further evaluate DFlash across more target models and inference frameworks. Table 11 reports SGLang results on one B200 with concurrency 8. Each entry shows acceptance length / speedup over autoregressive decoding. DFlash consistently improves over native MTP when both are available, and also scales to larger Qwen3.5, Qwen3-Coder, and GPT-OSS (Agarwal et al., 2025) models.

Table 11. Results across more models on SGLang. Each cell reports acceptance length / speedup.

<table><tr><td>Model</td><td>Method</td><td>Math500</td><td>HumanEval</td><td>MT-Bench</td></tr><tr><td rowspan="2">Qwen3.5-4B</td><td>MTP</td><td>6.5 / 1.5 $\times$ </td><td>6.3 / 1.6 $\times$ </td><td>5.3 / 1.3 $\times$ </td></tr><tr><td>DFlash</td><td>7.1 / 3.0 $\times$ </td><td>7.3 / 2.9 $\times$ </td><td>5.6 / 2.3 $\times$ </td></tr><tr><td rowspan="2">Qwen3.5-9B</td><td>MTP</td><td>6.7 / 1.7 $\times$ </td><td>6.6 / 1.7 $\times$ </td><td>5.3 / 1.3 $\times$ </td></tr><tr><td>DFlash</td><td>7.3 / 3.5 $\times$ </td><td>7.9 / 3.4 $\times$ </td><td>5.5 / 2.5 $\times$ </td></tr><tr><td rowspan="2">Qwen3.5-35B-A3B</td><td>MTP</td><td>6.9 / 1.7 $\times$ </td><td>7.1 / 1.6 $\times$ </td><td>5.2 / 1.2 $\times$ </td></tr><tr><td>DFlash</td><td>7.2 / 2.4 $\times$ </td><td>7.9 / 2.3 $\times$ </td><td>5.4 / 1.7 $\times$ </td></tr><tr><td>Qwen3.5-27B</td><td>DFlash</td><td>7.7 / 3.8 $\times$ </td><td>9.1 / 3.9 $\times$ </td><td>5.5 / 2.5 $\times$ </td></tr><tr><td>Qwen3-Coder-Next</td><td>DFlash</td><td>6.0 / 1.9 $\times$ </td><td>7.2 / 1.9 $\times$ </td><td>3.9 / 1.2 $\times$ </td></tr><tr><td>GPT-OSS-20B</td><td>DFlash</td><td>5.1 / 2.2 $\times$ </td><td>4.3 / 2.2 $\times$ </td><td>4.2 / 2.0 $\times$ </td></tr><tr><td>GPT-OSS-120B</td><td>DFlash</td><td>5.4 / 1.6 $\times$ </td><td>4.4 / 1.7 $\times$ </td><td>3.7 / 1.3 $\times$ </td></tr></table>

We also evaluate DFlash in vLLM on Qwen3.5-9B. Table 12 reports throughput and speedup over autoregressive decoding under different concurrency levels. DFlash achieves strong speedup at low and medium concurrency, while still maintaining throughput gains at high concurrency.

Table 12. vLLM results for Qwen3.5-9B. Each cell reports DFlash throughput in tok/s, with speedup over autoregressive decoding in parentheses.

<table><tr><td>Concurrency</td><td>Math500</td><td>HumanEval</td><td>MT-Bench</td></tr><tr><td>1</td><td>849 (4.0×)</td><td>969 (4.6×)</td><td>627 (3.0×)</td></tr><tr><td>8</td><td>5096 (3.2×)</td><td>5434 (3.4×)</td><td>3536 (2.2×)</td></tr><tr><td>16</td><td>7669 (2.5×)</td><td>8131 (2.7×)</td><td>5297 (1.7×)</td></tr><tr><td>32</td><td>9836 (1.9×)</td><td>10258 (2.1×)</td><td>6787 (1.3×)</td></tr></table>

## A.5. Further Ablations

## A.5.1. LOSS DECAY

We ablate the position-dependent loss decay introduced in Section 4.2. Specifically, we compare the default setting with exponentially decaying token weights against a variant trained with uniform token weighting within each draft block. This study isolates the effect of emphasizing earlytoken accuracy during training. Results in Figure 5 show that applying loss decay leads faster and better convergency.

![](images/5547ddba7a0b82aa0c0514e2d329cee3b5f470d0b971edd3d6568c81d20e9c99.jpg)

<details>
<summary>line chart</summary>

| Epoch | With loss decay | Without loss decay |
|-------|-----------------|--------------------|
| 1     | 4.4             | 4.2                |
| 2     | 5.3             | 5.2                |
| 3     | 5.8             | 5.6                |
| 4     | 6.1             | 6.0                |
| 5     | 6.2             | 6.2                |
| 6     | 6.3             | 6.3                |
| 7     | 6.4             | 6.4                |
| 8     | 6.4             | 6.4                |
| 9     | 6.4             | 6.4                |
</details>

Figure 5. The loss decay makes training converge faster and better.

## A.5.2. RANDOM SAMPLING OF MASKED BLOCKS

Table 13. Randomly sample anchor tokens to construct masked blocks during training effectively augments the training data and leads to higher acceptance length and better speedup. Both draft models use three layers and extract five hidden features from the target model. The block size is 16. We use the 100K data introduced in Section 5.5 to train both models.

<table><tr><td rowspan="2">Setting</td><td colspan="2">Math500</td><td colspan="2">HumanEval</td><td colspan="2">MT-Bench</td></tr><tr><td>Speedup</td><td> $\tau$ </td><td>Speedup</td><td> $\tau$ </td><td>Speedup</td><td> $\tau$ </td></tr><tr><td>Standard</td><td>4.13x</td><td>4.94</td><td>3.29x</td><td>3.86</td><td>2.13x</td><td>2.80</td></tr><tr><td>Sample</td><td>4.69x</td><td>5.64</td><td>3.90x</td><td>4.61</td><td>2.38x</td><td>3.18</td></tr></table>