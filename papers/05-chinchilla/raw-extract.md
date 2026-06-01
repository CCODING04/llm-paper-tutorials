# Training Compute-Optimal Large Language Models

Jordan Hoffmann★, Sebastian Borgeaud★, Arthur Mensch★, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals and Laurent Sifre★

★Equal contributions

We investigate the optimal model size and number of tokens for training a transformer language model under a given compute budget. We find that current large language models are significantly undertrained, a consequence of the recent focus on scaling language models whilst keeping the amount of training data constant. By training over 400 language models ranging from 70 million to over 16 billion parameters on 5 to 500 billion tokens, we find that for compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled. We test this hypothesis by training a predicted computeoptimal model, Chinchilla, that uses the same compute budget as Gopher but with 70B parameters and 4× more more data. Chinchilla uniformly and significantly outperforms Gopher (280B), GPT-3 (175B), Jurassic-1 (178B), and Megatron-Turing NLG (530B) on a large range of downstream evaluation tasks. This also means that Chinchilla uses substantially less compute for fine-tuning and inference, greatly facilitating downstream usage. As a highlight, Chinchilla reaches a state-of-the-art average accuracy of 67.5% on the MMLU benchmark, greater than a 7% improvement over Gopher.

# 1. Introduction

Recently a series of Large Language Models (LLMs) have been introduced (Brown et al., 2020; Lieber et al., 2021; Rae et al., 2021; Smith et al., 2022; Thoppilan et al., 2022), with the largest dense language models now having over 500 billion parameters. These large autoregressive transformers (Vaswani et al., 2017) have demonstrated impressive performance on many tasks using a variety of evaluation protocols such as zero-shot, few-shot, and fine-tuning.

The compute and energy cost for training large language models is substantial (Rae et al., 2021; Thoppilan et al., 2022) and rises with increasing model size. In practice, the allocated training compute budget is often known in advance: how many accelerators are available and for how long we want to use them. Since it is typically only feasible to train these large models once, accurately estimating the best model hyperparameters for a given compute budget is critical (Tay et al., 2021).

Kaplan et al. (2020) showed that there is a power law relationship between the number of parameters in an autoregressive language model (LM) and its performance. As a result, the field has been training larger and larger models, expecting performance improvements. One notable conclusion in Kaplan et al. (2020) is that large models should not be trained to their lowest possible loss to be compute optimal. Whilst we reach the same conclusion, we estimate that large models should be trained for many more training tokens than recommended by the authors. Specifically, given a 10× increase computational budget, they suggests that the size of the model should increase 5.5× while the number of training tokens should only increase 1.8×. Instead, we find that model size and the number of training tokens should be scaled in equal proportions.

Following Kaplan et al. (2020) and the training setup of GPT-3 (Brown et al., 2020), many of the recently trained large models have been trained for approximately 300 billion tokens (Table 1), in line with the approach of predominantly increasing model size when increasing compute.

![](images/619dbe87f70f6402a2893ec32e85ef74f0038c4b0f4f5171bc2ed203e3b0050e.jpg)  
Figure 1 | Overlaid predictions. We overlay the predictions from our three different approaches, along with projections from Kaplan et al. (2020). We find that all three methods predict that current large models should be substantially smaller and therefore trained much longer than is currently done. In Figure A3, we show the results with the predicted optimal tokens plotted against the optimal number of parameters for fixed FLOP budgets. Chinchilla outperforms Gopher and the other large models (see Section 4.2).

In this work, we revisit the question: Given a fixed FLOPs budget,1 how should one trade-off model size and the number of training tokens? To answer this question, we model the final pre-training loss2 $L ( N , D )$ as a function of the number of model parameters ??, and the number of training tokens, ??. Since the computational budget ?? is a deterministic function FLOPs(??, ??) of the number of seen training tokens and model parameters, we are interested in minimizing ?? under the constraint $\mathrm { F L O P s } ( N , D ) = C ;$ :

$$
N _ {o p t} (C), D _ {o p t} (C) = \underset {N, D \text {   s.t.   FLOPs } (N, D) = C} {\operatorname{argmin}} L (N, D). \tag {1}
$$

The functions $N _ { o p t } ( C )$ , and $D _ { o p t } ( C )$ describe the optimal allocation of a computational budget ??. We empirically estimate these functions based on the losses of over 400 models, ranging from under 70M to over 16B parameters, and trained on 5B to over 400B tokens – with each model configuration trained for several different training horizons. Our approach leads to considerably different results than that of Kaplan et al. (2020). We highlight our results in Figure 1 and how our approaches differ in Section 2.

Based on our estimated compute-optimal frontier, we predict that for the compute budget used to train Gopher, an optimal model should be 4 times smaller, while being training on 4 times more tokens. We verify this by training a more compute-optimal 70B model, called Chinchilla, on 1.4 trillion tokens. Not only does Chinchilla outperform its much larger counterpart, Gopher, but its reduced model size reduces inference cost considerably and greatly facilitates downstream uses on smaller hardware. The energy cost of a large language model is amortized through its usage for inference an fine-tuning. The benefits of a more optimally trained smaller model, therefore, extend beyond the immediate benefits of its improved performance.

Table 1 | Current LLMs. We show five of the current largest dense transformer models, their size, and the number of training tokens. Other than LaMDA (Thoppilan et al., 2022), most models are trained for approximately 300 billion tokens. We introduce Chinchilla, a substantially smaller model, trained for much longer than 300B tokens. 

<table><tr><td>Model</td><td>Size (# Parameters)</td><td>Training Tokens</td></tr><tr><td>LaMDA (Thoppilan et al., 2022)</td><td>137 Billion</td><td>168 Billion</td></tr><tr><td>GPT-3 (Brown et al., 2020)</td><td>175 Billion</td><td>300 Billion</td></tr><tr><td>Jurassic (Lieber et al., 2021)</td><td>178 Billion</td><td>300 Billion</td></tr><tr><td>Gopher (Rae et al., 2021)</td><td>280 Billion</td><td>300 Billion</td></tr><tr><td>MT-NLG 530B (Smith et al., 2022)</td><td>530 Billion</td><td>270 Billion</td></tr><tr><td>Chinchilla</td><td>70 Billion</td><td>1.4 Trillion</td></tr></table>

# 2. Related Work

Large language models. A variety of large language models have been introduced in the last few years. These include both dense transformer models (Brown et al., 2020; Lieber et al., 2021; Rae et al., 2021; Smith et al., 2022; Thoppilan et al., 2022) and mixture-of-expert (MoE) models (Du et al., 2021; Fedus et al., 2021; Zoph et al., 2022). The largest dense transformers have passed 500 billion parameters (Smith et al., 2022). The drive to train larger and larger models is clear—so far increasing the size of language models has been responsible for improving the state-of-the-art in many language modelling tasks. Nonetheless, large language models face several challenges, including their overwhelming computational requirements (the cost of training and inference increase with model size) (Rae et al., 2021; Thoppilan et al., 2022) and the need for acquiring more high-quality training data. In fact, in this work we find that larger, high quality datasets will play a key role in any further scaling of language models.

Modelling the scaling behavior. Understanding the scaling behaviour of language models and their transfer properties has been important in the development of recent large models (Hernandez et al., 2021; Kaplan et al., 2020). Kaplan et al. (2020) first showed a predictable relationship between model size and loss over many orders of magnitude. The authors investigate the question of choosing the optimal model size to train for a given compute budget. Similar to us, they address this question by training various models. Our work differs from Kaplan et al. (2020) in several important ways. First, the authors use a fixed number of training tokens and learning rate schedule for all models; this prevents them from modelling the impact of these hyperparameters on the loss. In contrast, we find that setting the learning rate schedule to approximately match the number of training tokens results in the best final loss regardless of model size—see Figure A1. For a fixed learning rate cosine schedule to 130B tokens, the intermediate loss estimates (for ??0 << 130B) are therefore overestimates of the loss of a model trained with a schedule length matching ??0. Using these intermediate losses results in underestimating the effectiveness of training models on less data than 130B tokens, and eventually contributes to the conclusion that model size should increase faster than training data size as compute budget increases. In contrast, our analysis predicts that both quantities should scale at roughly the same rate. Secondly, we include models with up to 16B parameters, as we observe that there is slight curvature in the FLOP-loss frontier (see Appendix E)—in fact, the majority of the models used in our analysis have more than 500 million parameters, in contrast the majority of runs in Kaplan et al. (2020) are significantly smaller—many being less than 100M parameters.

Recently, Clark et al. (2022) specifically looked in to the scaling properties of Mixture of Expert language models, showing that the scaling with number of experts diminishes as the model size increases—their approach models the loss as a function of two variables: the model size and the number of experts. However, the analysis is done with a fixed number of training tokens, as in Kaplan et al. (2020), potentially underestimating the improvements of branching.

Estimating hyperparameters for large models. The model size and the number of training tokens are not the only two parameters to chose when selecting a language model and a procedure to train it. Other important factors include learning rate, learning rate schedule, batch size, optimiser, and width-to-depth ratio. In this work, we focus on model size and the number of training steps, and we rely on existing work and provided experimental heuristics to determine the other necessary hyperparameters. Yang et al. (2021) investigates how to choose a variety of these parameters for training an autoregressive transformer, including the learning rate and batch size. McCandlish et al. (2018) finds only a weak dependence between optimal batch size and model size. Shallue et al. (2018); Zhang et al. (2019) suggest that using larger batch-sizes than those we use is possible. Levine et al. (2020) investigates the optimal depth-to-width ratio for a variety of standard model sizes. We use slightly less deep models than proposed as this translates to better wall-clock performance on our hardware.

Improved model architectures. Recently, various promising alternatives to traditional dense transformers have been proposed. For example, through the use of conditional computation large MoE models like the 1.7 trillion parameter Switch transformer (Fedus et al., 2021), the 1.2 Trillion parameter GLaM model (Du et al., 2021), and others (Artetxe et al., 2021; Zoph et al., 2022) are able to provide a large effective model size despite using relatively fewer training and inference FLOPs. However, for very large models the computational benefits of routed models seems to diminish (Clark et al., 2022). An orthogonal approach to improving language models is to augment transformers with explicit retrieval mechanisms, as done by Borgeaud et al. (2021); Guu et al. (2020); Lewis et al. (2020). This approach effectively increases the number of data tokens seen during training (by a factor of ∼ 10 in Borgeaud et al. (2021)). This suggests that the performance of language models may be more dependant on the size of the training data than previously thought.

# 3. Estimating the optimal parameter/training tokens allocation

We present three different approaches to answer the question driving our research: Given a fixed FLOPs budget, how should one trade-off model size and the number of training tokens? In all three cases we start by training a range of models varying both model size and the number of training tokens and use the resulting training curves to fit an empirical estimator of how they should scale. We assume a power-law relationship between compute and model size as done in Clark et al. (2022); Kaplan et al. (2020), though future work may want to include potential curvature in this relationship for large model sizes. The resulting predictions are similar for all three methods and suggest that parameter count and number of training tokens should be increased equally with more compute3— with proportions reported in Table 2. This is in clear contrast to previous work on this topic and warrants further investigation.

![](images/8c3e37a4249372b59363c6dee6e4cea1666b35941e23316079ab570a0a57d03e.jpg)

![](images/2768e5a3d47159ee0dc17e264fb9e3e2d4171f137bcb4c16f8ef57355ad472f4.jpg)

![](images/c31dbf57d064c6600d7f047efe698bc5373da3e02ede4736121853eb98cbf533.jpg)  
Figure 2 | Training curve envelope. On the left we show all of our different runs. We launched a range of model sizes going from 70M to 10B, each for four different cosine cycle lengths. From these curves, we extracted the envelope of minimal loss per FLOP, and we used these points to estimate the optimal model size (center) for a given compute budget and the optimal number of training tokens (right). In green, we show projections of optimal model size and training token count based on the number of FLOPs used to train Gopher $( 5 . { \bar { 7 6 } } \times 1 0 ^ { 2 3 } )$ .

# 3.1. Approach 1: Fix model sizes and vary number of training tokens

In our first approach we vary the number of training steps for a fixed family of models (ranging from 70M to over 10B parameters), training each model for 4 different number of training sequences. From these runs, we are able to directly extract an estimate of the minimum loss achieved for a given number of training FLOPs. Training details for this approach can be found in Appendix D.

For each parameter count ?? we train 4 different models, decaying the learning rate by a factor of 10× over a horizon (measured in number of training tokens) that ranges by a factor of 16×. Then, for each run, we smooth and then interpolate the training loss curve. From this, we obtain a continuous mapping from FLOP count to training loss for each run. Then, for each FLOP count, we determine which run achieves the lowest loss. Using these interpolants, we obtain a mapping from any FLOP count ??, to the most efficient choice of model size ?? and number of training tokens ?? such that $\mathrm { F L O P s } ( N , D ) = { C } . ^ { 4 }$ At 1500 logarithmically spaced FLOP values, we find which model size achieves the lowest loss of all models along with the required number of training tokens. Finally, we fit power laws to estimate the optimal model size and number of training tokens for any given amount of compute (see the center and right panels of Figure 2), obtaining a relationship $N _ { o p t } \propto C ^ { a }$ and $D _ { o p t } \propto C ^ { b }$ . We find that $a = 0 . 5 0$ and ?? = 0.50—as summarized in Table 2. In Section D.4, we show a head-to-head comparison at $1 0 ^ { 2 1 }$ FLOPs, using the model size recommended by our analysis and by the analysis of Kaplan et al. (2020)—using the model size we predict has a clear advantage.

# 3.2. Approach 2: IsoFLOP profiles

In our second approach we vary the model size5 for a fixed set of 9 different training FLOP counts6 (ranging from $6 \times 1 0 ^ { 1 8 }$ to $3 \times 1 0 ^ { 2 1 }$ FLOPs), and consider the final training loss for each point7. in contrast with Approach 1 that considered points (??, ??, ??) along the entire training runs. This allows us to directly answer the question: For a given FLOP budget, what is the optimal parameter count?

![](images/20292889b56122b5bf3b099553359d0ef317ccb1ef5a8fc4ebfe0520fdb88977.jpg)

![](images/df70e843a7c8f67d43d66c299a7f5387e5bb37529bdd955fa8d29e03d694dce3.jpg)

![](images/1370dc827fb2900c2cd166ef68587810a0e5929d6713a245cacd02ac215a7a09.jpg)  
Figure 3 | IsoFLOP curves. For various model sizes, we choose the number of training tokens such that the final FLOPs is a constant. The cosine cycle length is set to match the target FLOP count. We find a clear valley in loss, meaning that for a given FLOP budget there is an optimal model to train (left). Using the location of these valleys, we project optimal model size and number of tokens for larger models (center and right). In green, we show the estimated number of parameters and tokens for an optimal model trained with the compute budget of Gopher.

For each FLOP budget, we plot the final loss (after smoothing) against the parameter count in Figure 3 (left). In all cases, we ensure that we have trained a diverse enough set of model sizes to see a clear minimum in the loss. We fit a parabola to each IsoFLOPs curve to directly estimate at what model size the minimum loss is achieved (Figure 3 (left)). As with the previous approach, we then fit a power law between FLOPs and loss-optimal model size and number of training tokens, shown in Figure 3 (center, right). Again, we fit exponents of the form $N _ { o p t } \propto C ^ { a }$ and $D _ { o p t } \propto C ^ { b }$ and we find that $a = 0 . 4 9$ and $b = 0 . 5 1 - \mathsf { a s }$ summarized in Table 2.

# 3.3. Approach 3: Fitting a parametric loss function

Lastly, we model all final losses from experiments in Approach 1 & 2 as a parametric function of model parameter count and the number of seen tokens. Following a classical risk decomposition (see Section D.2), we propose the following functional form

$$
\hat {L} (N, D) \triangleq E + \frac {A}{N ^ {\alpha}} + \frac {B}{D ^ {\beta}}. \tag {2}
$$

The first term captures the loss for an ideal generative process on the data distribution, and should correspond to the entropy of natural text. The second term captures the fact that a perfectly trained transformer with ?? parameters underperforms the ideal generative process. The final term captures the fact that the transformer is not trained to convergence, as we only make a finite number of optimisation steps, on a sample of the dataset distribution.

Model fitting. To estimate $( A , B , E , \alpha , \beta )$ , we minimize the Huber loss (Huber, 1964) between the predicted and observed log loss using the L-BFGS algorithm (Nocedal, 1980):

$$
\min _ {A, B, E, \alpha , \beta} \quad \sum_ {\text { Runs } i} \operatorname{Huber} _ {\delta} \left(\log \hat {L} (N _ {i}, D _ {i}) - \log L _ {i}\right) \tag {3}
$$

We account for possible local minima by selecting the best fit from a grid of initialisations. The Huber loss $( \delta = 1 0 ^ { - 3 } )$ is robust to outliers, which we find important for good predictive performance over held-out data points. Section D.2 details the fitting procedure and the loss decomposition.

![](images/8f892dbba7fef950139a62d77e5f935c7d6d42812301d850b97c66ff8f89e270.jpg)

![](images/5ee103c96ada062702e73b3319d6f6c062f42874fc4a630f02e4bccc66a7c2fe.jpg)  
Figure 4 | Parametric fit. We fit a parametric modelling of the loss $\hat { L } ( N , D )$ and display contour (left) and isoFLOP slices (right). For each isoFLOP slice, we include a corresponding dashed line in the left plot. In the left plot, we show the efficient frontier in blue, which is a line in log-log space. Specifically, the curve goes through each iso-loss contour at the point with the fewest FLOPs. We project the optimal model size given the Gopher FLOP budget to be 40B parameters.

Efficient frontier. We can approximate the functions $N _ { o p t }$ and $D _ { o p t }$ by minimizing the parametric loss ??ˆ under the constraint FLOPs $( N , D ) \approx 6 N D$ (Kaplan et al., 2020). The resulting $N _ { o p t }$ and $D _ { o p t }$ balance the two terms in Equation (3) that depend on model size and data. By construction, they have a power-law form:

$$
N _ {o p t} (C) = G \left(\frac {C}{6}\right) ^ {a}, \quad D _ {o p t} (C) = G ^ {- 1} \left(\frac {C}{6}\right) ^ {b}, \quad \text { where } \quad G = \left(\frac {\alpha A}{\beta B}\right) ^ {\frac {1}{\alpha + \beta}}, \quad a = \frac {\beta}{\alpha + \beta}, \text { and } b = \frac {\alpha}{\alpha + \beta}. \tag {4}
$$

We show contours of the fitted function ??ˆin Figure 4 (left), and the closed-form efficient computational frontier in blue. From this approach, we find that $a = 0 . 4 6$ and $b = 0 . 5 4$ —as summarized in Table 2.

# 3.4. Optimal model scaling

We find that the three approaches, despite using different fitting methodologies and different trained models, yield comparable predictions for the optimal scaling in parameters and tokens with FLOPs (shown in Table 2). All three approaches suggest that as compute budget increases, model size and the amount of training data should be increased in approximately equal proportions. The first and second approaches yield very similar predictions for optimal model sizes, as shown in Figure 1 and Figure A3. The third approach predicts even smaller models being optimal at larger compute budgets. We note that the observed points (??, ??, ??) for low training FLOPs $( C \leqslant 1 e 2 1 )$ have larger residuals $\left| \left| L - \hat { L } ( N , D ) \right| \right| _ { 2 } ^ { 2 }$ than points with higher computational budgets. The fitted model places increased weight on the points with more FLOPs—automatically considering the low-computational budget points as outliers due to the Huber loss. As a consequence of the empirically observed negative curvature in the frontier $C  N _ { o p t }$ (see Appendix E), this results in predicting a lower $N _ { o p t }$ than the two other approaches.

In Table 3 we show the estimated number of FLOPs and tokens that would ensure that a model of a given size lies on the compute-optimal frontier. Our findings suggests that the current generation of large language models are considerably over-sized, given their respective compute budgets, as shown in Figure 1. For example, we find that a 175 billion parameter model should be trained with a compute budget of $4 . 4 1 \times 1 0 ^ { 2 4 }$ FLOPs and on over 4.2 trillion tokens. A 280 billion Gopher-like model is the optimal model to train given a compute budget of approximately $1 0 ^ { 2 5 }$ FLOPs and should be trained on 6.8 trillion tokens. Unless one has a compute budget of $1 0 ^ { 2 6 }$ FLOPs (over 250× the compute used to train Gopher), a 1 trillion parameter model is unlikely to be the optimal model to train. Furthermore, the amount of training data that is projected to be needed is far beyond what is currently used to train large models, and underscores the importance of dataset collection in addition to engineering improvements that allow for model scale. While there is significant uncertainty extrapolating out many orders of magnitude, our analysis clearly suggests that given the training compute budget for many current LLMs, smaller models should have been trained on more tokens to achieve the most performant model.

Table 2 | Estimated parameter and data scaling with increased training compute. The listed values are the exponents, ?? and ??, on the relationship $N _ { o p t } \propto C ^ { a }$ and $D _ { o p t } \propto C ^ { b }$ . Our analysis suggests a near equal scaling in parameters and data with increasing compute which is in clear contrast to previous work on the scaling of large models. The $1 0 ^ { \mathrm { t h } }$ and $9 0 ^ { \mathrm { t h } }$ percentiles are estimated via bootstrapping data (80% of the dataset is sampled 100 times) and are shown in parenthesis. 

<table><tr><td>Approach</td><td>Coeff. a where  $N_{opt} \propto C^{a}$ </td><td>Coeff. b where  $D_{opt} \propto C^{b}$ </td></tr><tr><td>1. Minimum over training curves</td><td>0.50 (0.488, 0.502)</td><td>0.50 (0.501, 0.512)</td></tr><tr><td>2. IsoFLOP profiles</td><td>0.49 (0.462, 0.534)</td><td>0.51 (0.483, 0.529)</td></tr><tr><td>3. Parametric modelling of the loss</td><td>0.46 (0.454, 0.455)</td><td>0.54 (0.542, 0.543)</td></tr><tr><td>Kaplan et al. (2020)</td><td>0.73</td><td>0.27</td></tr></table>

Table 3 | Estimated optimal training FLOPs and training tokens for various model sizes. For various model sizes, we show the projections from Approach 1 of how many FLOPs and training tokens would be needed to train compute-optimal models. The estimates for Approach 2 & 3 are similar (shown in Section D.3) 

<table><tr><td>Parameters</td><td>FLOPs</td><td>FLOPs (in Gopher unit)</td><td>Tokens</td></tr><tr><td>400 Million</td><td>1.92e+19</td><td>1/29,968</td><td>8.0 Billion</td></tr><tr><td>1 Billion</td><td>1.21e+20</td><td>1/4,761</td><td>20.2 Billion</td></tr><tr><td>10 Billion</td><td>1.23e+22</td><td>1/46</td><td>205.1 Billion</td></tr><tr><td>67 Billion</td><td>5.76e+23</td><td>1</td><td>1.5 Trillion</td></tr><tr><td>175 Billion</td><td>3.85e+24</td><td>6.7</td><td>3.7 Trillion</td></tr><tr><td>280 Billion</td><td>9.90e+24</td><td>17.2</td><td>5.9 Trillion</td></tr><tr><td>520 Billion</td><td>3.43e+25</td><td>59.5</td><td>11.0 Trillion</td></tr><tr><td>1 Trillion</td><td>1.27e+26</td><td>221.3</td><td>21.2 Trillion</td></tr><tr><td>10 Trillion</td><td>1.30e+28</td><td>22515.9</td><td>216.2 Trillion</td></tr></table>

In Appendix C, we reproduce the IsoFLOP analysis on two additional datasets: C4 (Raffel et al., 2020a) and GitHub code (Rae et al., 2021). In both cases we reach the similar conclusion that model size and number of training tokens should be scaled in equal proportions.

# 4. Chinchilla

Based on our analysis in Section 3, the optimal model size for the Gopher compute budget is somewhere between 40 and 70 billion parameters. We test this hypothesis by training a model on the larger end of this range—70B parameters—for 1.4T tokens, due to both dataset and computational efficiency considerations. In this section we compare this model, which we call Chinchilla, to Gopher and other LLMs. Both Chinchilla and Gopher have been trained for the same number of FLOPs but differ in the size of the model and the number of training tokens.

While pre-training a large language model has a considerable compute cost, downstream finetuning and inference also make up substantial compute usage (Rae et al., 2021). Due to being 4× smaller than Gopher, both the memory footprint and inference cost of Chinchilla are also smaller.

# 4.1. Model and training details

The full set of hyperparameters used to train Chinchilla are given in Table 4. Chinchilla uses the same model architecture and training setup as Gopher with the exception of the differences listed below.

• We train Chinchilla on MassiveText (the same dataset as Gopher) but use a slightly different subset distribution (shown in Table A1) to account for the increased number of training tokens.   
• We use AdamW (Loshchilov and Hutter, 2019) for Chinchilla rather than Adam (Kingma and Ba, 2014) as this improves the language modelling loss and the downstream task performance after finetuning.8   
• We train Chinchilla with a slightly modified SentencePiece (Kudo and Richardson, 2018) tokenizer that does not apply NFKC normalisation. The vocabulary is very similar– 94.15% of tokens are the same as those used for training Gopher. We find that this particularly helps with the representation of mathematics and chemistry, for example.   
• Whilst the forward and backward pass are computed in bfloat16, we store a float32 copy of the weights in the distributed optimiser state (Rajbhandari et al., 2020). See Lessons Learned from Rae et al. (2021) for additional details.

In Appendix G we show the impact of the various optimiser related changes between Chinchilla and Gopher. All models in this analysis have been trained on TPUv3/TPUv4 (Jouppi et al., 2017) with JAX (Bradbury et al., 2018) and Haiku (Hennigan et al., 2020). We include a Chinchilla model card (Mitchell et al., 2019) in Table A8.

<table><tr><td>Model</td><td>Layers</td><td>Number Heads</td><td>Key/Value Size</td><td> $d_{model}$ </td><td>Max LR</td><td>Batch Size</td></tr><tr><td>Gopher 280B</td><td>80</td><td>128</td><td>128</td><td>16,384</td><td> $4 \times 10^{-5}$ </td><td>3M → 6M</td></tr><tr><td>Chinchilla 70B</td><td>80</td><td>64</td><td>128</td><td>8,192</td><td> $1 \times 10^{-4}$ </td><td>1.5M → 3M</td></tr></table>

Table 4 | Chinchilla architecture details. We list the number of layers, the key/value size, the bottleneck activation size $\mathrm { d } _ { \mathrm { m o d e l } }$ , the maximum learning rate, and the training batch size (# tokens). The feed-forward size is always set to $4 \times \mathrm { d } _ { \mathrm { m o d e l } }$ . Note that we double the batch size midway through training for both Chinchilla and Gopher.

<table><tr><td></td><td># Tasks</td><td>Examples</td></tr><tr><td>Language Modelling</td><td>20</td><td>WikiText-103, The Pile: PG-19, arXiv, FreeLaw, ...</td></tr><tr><td>Reading Comprehension</td><td>3</td><td>RACE-m, RACE-h, LAMBADA</td></tr><tr><td>Question Answering</td><td>3</td><td>Natural Questions, TriviaQA, TruthfulQA</td></tr><tr><td>Common Sense</td><td>5</td><td>HellaSwag, Winogrande, PIQA, SIQA, BoolQ</td></tr><tr><td>MMLU</td><td>57</td><td>High School Chemistry, Astronomy, Clinical Knowledge, ...</td></tr><tr><td>BIG-bench</td><td>62</td><td>Causal Judgement, Epistemic Reasoning, Temporal Sequences, ...</td></tr></table>

Table 5 | All evaluation tasks. We evaluate Chinchilla on a collection of language modelling along with downstream tasks. We evaluate on largely the same tasks as in Rae et al. (2021), to allow for direct comparison.

# 4.2. Results

We perform an extensive evaluation of Chinchilla, comparing against various large language models. We evaluate on a large subset of the tasks presented in Rae et al. (2021), shown in Table 5. As the focus of this work is on optimal model scaling, we included a large representative subset, and introduce a few new evaluations to allow for better comparison to other existing large models. The evaluation details for all tasks are the same as described in Rae et al. (2021).

# 4.2.1. Language modelling

![](images/a5f4e126766a8d62bba555f1c09c38c5f127c87674eae163882ab5cab732cc98.jpg)  
Figure 5 | Pile Evaluation. For the different evaluation sets in The Pile (Gao et al., 2020), we show the bits-per-byte (bpb) improvement (decrease) of Chinchilla compared to Gopher. On all subsets, Chinchilla outperforms Gopher.

Chinchilla significantly outperforms Gopher on all evaluation subsets of The Pile (Gao et al., 2020), as shown in Figure 5. Compared to Jurassic-1 (178B) Lieber et al. (2021), Chinchilla is more performant on all but two subsets– dm\_mathematics and ubuntu\_irc– see Table A5 for a raw bits-per-byte comparison. On Wikitext103 (Merity et al., 2017), Chinchilla achieves a perplexity of 7.16 compared to 7.75 for Gopher. Some caution is needed when comparing Chinchilla with Gopher on these language modelling benchmarks as Chinchilla is trained on 4× more data than Gopher and thus train/test set leakage may artificially enhance the results. We thus place more emphasis on other tasks for which leakage is less of a concern, such as MMLU (Hendrycks et al., 2020) and BIG-bench (BIG-bench collaboration, 2021) along with various closed-book question answering and common sense analyses.

<table><tr><td>Random</td><td>25.0%</td></tr><tr><td>Average human rater</td><td>34.5%</td></tr><tr><td>GPT-3 5-shot</td><td>43.9%</td></tr><tr><td>Gopher 5-shot</td><td>60.0%</td></tr><tr><td>Chinchilla 5-shot</td><td>67.6%</td></tr><tr><td>Average human expert performance</td><td>89.8%</td></tr><tr><td>June 2022 Forecast</td><td>57.1%</td></tr><tr><td>June 2023 Forecast</td><td>63.4%</td></tr></table>

Table 6 | Massive Multitask Language Understanding (MMLU). We report the average 5-shot accuracy over 57 tasks with model and human accuracy comparisons taken from Hendrycks et al. (2020). We also include the average prediction for state of the art accuracy in June 2022/2023 made by 73 competitive human forecasters in Steinhardt (2021).

# 4.2.2. MMLU

The Massive Multitask Language Understanding (MMLU) benchmark (Hendrycks et al., 2020) consists of a range of exam-like questions on academic subjects. In Table 6, we report Chinchilla’s average 5-shot performance on MMLU (the full breakdown of results is shown in Table A6). On this benchmark, Chinchilla significantly outperforms Gopher despite being much smaller, with an average accuracy of 67.6% (improving upon Gopher by 7.6%). Remarkably, Chinchilla even outperforms the expert forecast for June 2023 of 63.4% accuracy (see Table 6) (Steinhardt, 2021). Furthermore, Chinchilla achieves greater than 90% accuracy on 4 different individual tasks– high\_school\_gov\_and\_politics, international\_law, sociology, and us\_foreign\_policy. To our knowledge, no other model has achieved greater than 90% accuracy on a subset.

In Figure 6, we show a comparison to Gopher broken down by task. Overall, we find that Chinchilla improves performance on the vast majority of tasks. On four tasks (college\_mathematics, econometrics, moral\_scenarios, and formal\_logic) Chinchilla underperforms Gopher, and there is no change in performance on two tasks.

# 4.2.3. Reading comprehension

On the final word prediction dataset LAMBADA (Paperno et al., 2016), Chinchilla achieves 77.4% accuracy, compared to 74.5% accuracy from Gopher and 76.6% from MT-NLG 530B (see Table 7). On RACE-h and RACE-m (Lai et al., 2017), Chinchilla greatly outperforms Gopher, improving accuracy by more than 10% in both cases—see Table 7.

# 4.2.4. BIG-bench

We analysed Chinchilla on the same set of BIG-bench tasks (BIG-bench collaboration, 2021) reported in Rae et al. (2021). Similar to what we observed in MMLU, Chinchilla outperforms Gopher on the vast majority of tasks (see Figure 7). We find that Chinchilla improves the average performance by 10.7%, reaching an accuracy of 65.1% versus 54.4% for Gopher. Of the 62 tasks we consider, Chinchilla performs worse than Gopher on only four—crash\_blossom, dark\_humor\_detection, mathematical\_induction and logical\_args. Full accuracy results for Chinchilla can be found in Table A7.

![](images/34aa4019209102677d217a60bd8819adc95bdcbc2b7060f65984dfa9f5f3a51e.jpg)

Figure 6 | MMLU results compared to Gopher We find that Chinchilla outperforms Gopher by 7.6% on average (see Table 6) in addition to performing better on 51/57 individual tasks, the same on 2/57, and worse on only 4/57 tasks. 

<table><tr><td></td><td>Chinchilla</td><td>Gopher</td><td>GPT-3</td><td>MT-NLG 530B</td></tr><tr><td>LAMBADA Zero-Shot</td><td>77.4</td><td>74.5</td><td>76.2</td><td>76.6</td></tr><tr><td>RACE-m Few-Shot</td><td>86.8</td><td>75.1</td><td>58.1</td><td>-</td></tr><tr><td>RACE-h Few-Shot</td><td>82.3</td><td>71.6</td><td>46.8</td><td>47.9</td></tr></table>

Table 7 | Reading comprehension. On RACE-h and RACE-m (Lai et al., 2017), Chinchilla considerably improves performance over Gopher. Note that GPT-3 and MT-NLG 530B use a different prompt format than we do on RACE-h/m, so results are not comparable to Gopher and Chinchilla. On LAMBADA (Paperno et al., 2016), Chinchilla outperforms both Gopher and MT-NLG 530B.

# 4.2.5. Common sense

We evaluate Chinchilla on various common sense benchmarks: PIQA (Bisk et al., 2020), SIQA (Sap et al., 2019), Winogrande (Sakaguchi et al., 2020), HellaSwag (Zellers et al., 2019), and BoolQ (Clark et al., 2019). We find that Chinchilla outperforms both Gopher and GPT-3 on all tasks and outperforms MT-NLG 530B on all but one task—see Table 8.

On TruthfulQA (Lin et al., 2021), Chinchilla reaches 43.6%, 58.5%, and 66.7% accuracy with 0-shot, 5-shot, and 10-shot respectively. In comparison, Gopher achieved only 29.5% 0-shot and 43.7% 10-shot accuracy. In stark contrast with the findings of Lin et al. (2021), the large improvements (14.1% in 0-shot accuracy) achieved by Chinchilla suggest that better modelling of the pre-training data alone can lead to substantial improvements on this benchmark.

![](images/e008ea7837c9e91f5c201416e83a8c002784237687d4ebecf3cf5f41df45c871.jpg)  
Figure 7 | BIG-bench results compared to Gopher Chinchilla out performs Gopher on all but four BIG-bench tasks considered. Full results are in Table A7.

# 4.2.6. Closed-book question answering

Results on closed-book question answering benchmarks are reported in Table 9. On the Natural Questions dataset (Kwiatkowski et al., 2019), Chinchilla achieves new closed-book SOTA accuracies: 31.5% 5-shot and 35.5% 64-shot, compared to 21% and 28% respectively, for Gopher. On TriviaQA (Joshi et al., 2017) we show results for both the filtered (previously used in retrieval and open-book work) and unfiltered set (previously used in large language model evaluations). In both cases, Chinchilla substantially out performs Gopher. On the filtered version, Chinchilla lags behind the open book SOTA (Izacard and Grave, 2020) by only 7.9%. On the unfiltered set, Chinchilla outperforms GPT-3—see Table 9.

# 4.2.7. Gender bias and toxicity

Large Language Models carry potential risks such as outputting offensive language, propagating social biases, and leaking private information (Bender et al., 2021; Weidinger et al., 2021). We expect Chinchilla to carry risks similar to Gopher because Chinchilla is trained on the same data, albeit with slightly different relative weights, and because it has a similar architecture. Here, we examine gender bias (particularly gender and occupation bias) and generation of toxic language. We select a few common evaluations to highlight potential issues, but stress that our evaluations are not comprehensive and much work remains to understand, evaluate, and mitigate risks in LLMs.

<table><tr><td></td><td>Chinchilla</td><td>Gopher</td><td>GPT-3</td><td>MT-NLG 530B</td><td>Supervised SOTA</td></tr><tr><td>HellaSWAG</td><td>80.8%</td><td>79.2%</td><td>78.9%</td><td>80.2%</td><td>93.9%</td></tr><tr><td>PIQA</td><td>81.8%</td><td>81.8%</td><td>81.0%</td><td>82.0%</td><td>90.1%</td></tr><tr><td>Winogrande</td><td>74.9%</td><td>70.1%</td><td>70.2%</td><td>73.0%</td><td>91.3%</td></tr><tr><td>SIQA</td><td>51.3%</td><td>50.6%</td><td>-</td><td>-</td><td>83.2%</td></tr><tr><td>BoolQ</td><td>83.7%</td><td>79.3%</td><td>60.5%</td><td>78.2%</td><td>91.4%</td></tr></table>

Table 8 | Zero-shot comparison on Common Sense benchmarks. We show a comparison between Chinchilla, Gopher, and MT-NLG 530B on various Common Sense benchmarks. We see that Chinchilla matches or outperforms Gopher and GPT-3 on all tasks. On all but one Chinchilla outperforms the much larger MT-NLG 530B model.

<table><tr><td></td><td>Method</td><td>Chinchilla</td><td>Gopher</td><td>GPT-3</td><td>SOTA (open book)</td></tr><tr><td rowspan="3">Natural Questions (dev)</td><td>0-shot</td><td>16.6%</td><td>10.1%</td><td>14.6%</td><td rowspan="3">54.4%</td></tr><tr><td>5-shot</td><td>31.5%</td><td>24.5%</td><td>-</td></tr><tr><td>64-shot</td><td>35.5%</td><td>28.2%</td><td>29.9%</td></tr><tr><td rowspan="3">TriviaQA (unfiltered, test)</td><td>0-shot</td><td>67.0%</td><td>52.8%</td><td>64.3%</td><td rowspan="3">-</td></tr><tr><td>5-shot</td><td>73.2%</td><td>63.6%</td><td>-</td></tr><tr><td>64-shot</td><td>72.3%</td><td>61.3%</td><td>71.2%</td></tr><tr><td rowspan="3">TriviaQA (filtered, dev)</td><td>0-shot</td><td>55.4%</td><td>43.5%</td><td>-</td><td rowspan="3">72.5%</td></tr><tr><td>5-shot</td><td>64.1%</td><td>57.0%</td><td>-</td></tr><tr><td>64-shot</td><td>64.6%</td><td>57.2%</td><td>-</td></tr></table>

Table 9 | Closed-book question answering. For Natural Questions (Kwiatkowski et al., 2019) and TriviaQA (Joshi et al., 2017), Chinchilla outperforms Gopher in all cases. On Natural Questions, Chinchilla outperforms GPT-3. On TriviaQA we show results on two different evaluation sets to allow for comparison to GPT-3 and to open book SOTA (FiD + Distillation (Izacard and Grave, 2020)).

Gender bias. As discussed in Rae et al. (2021), large language models reflect contemporary and historical discourse about different groups (such as gender groups) from their training dataset, and we expect the same to be true for Chinchilla. Here, we test if potential gender and occupation biases manifest in unfair outcomes on coreference resolutions, using the Winogender dataset (Rudinger et al., 2018) in a zero-shot setting. Winogender tests whether a model can correctly determine if a pronoun refers to different occupation words. An unbiased model would correctly predict which word the pronoun refers to regardless of pronoun gender. We follow the same setup as in Rae et al. (2021) (described further in Section H.3).

As shown in Table 10, Chinchilla correctly resolves pronouns more frequently than Gopher across all groups. Interestingly, the performance increase is considerably smaller for male pronouns (increase of 3.2%) than for female or neutral pronouns (increases of 8.3% and 9.2% respectively). We also consider gotcha examples, in which the correct pronoun resolution contradicts gender stereotypes (determined by labor statistics). Again, we see that Chinchilla resolves pronouns more accurately than Gopher. When breaking up examples by male/female gender and gotcha/not gotcha, the largest improvement is on female gotcha examples (improvement of 10%). Thus, though Chinchilla uniformly overcomes gender stereotypes for more coreference examples than Gopher, the rate of improvement is higher for some pronouns than others, suggesting that the improvements conferred by using a more compute-optimal model can be uneven.

Sample toxicity. Language models are capable of generating toxic language—including insults, hate speech, profanities and threats (Gehman et al., 2020; Rae et al., 2021). While toxicity is an umbrella term, and its evaluation in LMs comes with challenges (Welbl et al., 2021; Xu et al., 2021), automatic classifier scores can provide an indication for the levels of harmful text that a LM generates. Rae et al. (2021) found that improving language modelling loss by increasing the number of model parameters has only a negligible effect on toxic text generation (unprompted); here we analyze whether the same holds true for a lower LM loss achieved via more compute-optimal training. Similar to the protocol of Rae et al. (2021), we generate 25,000 unprompted samples from Chinchilla, and compare their PerspectiveAPI toxicity score distribution to that of Gopher-generated samples. Several summary statistics indicate an absence of major differences: the mean (median) toxicity score for Gopher is 0.081 (0.064), compared to 0.087 (0.066) for Chinchilla, and the $9 5 ^ { \mathrm { t h } }$ percentile scores are 0.230 for Gopher, compared to 0.238 for Chinchilla. That is, the large majority of generated samples are classified as non-toxic, and the difference between the models is negligible. In line with prior findings (Rae et al., 2021), this suggests that toxicity levels in unconditional text generation are largely independent of the model quality (measured in language modelling loss), i.e. that better models of the training dataset are not necessarily more toxic.

<table><tr><td></td><td>Chinchilla</td><td>Gopher</td></tr><tr><td>All</td><td>78.3%</td><td>71.4%</td></tr><tr><td>Male</td><td>71.2%</td><td>68.0%</td></tr><tr><td>Female</td><td>79.6%</td><td>71.3%</td></tr><tr><td>Neutral</td><td>84.2%</td><td>75.0%</td></tr></table>

<table><tr><td></td><td>Chinchilla</td><td>Gopher</td></tr><tr><td>Male gotcha</td><td>62.5%</td><td>59.2%</td></tr><tr><td>Male not gotcha</td><td>80.0%</td><td>76.7%</td></tr><tr><td>Female gotcha</td><td>76.7%</td><td>66.7%</td></tr><tr><td>Female not gotcha</td><td>82.5%</td><td>75.8%</td></tr></table>

Table 10 | Winogender results. Left: Chinchilla consistently resolves pronouns better than Gopher. Right: Chinchilla performs better on examples which contradict gender stereotypes (gotcha examples). However, difference in performance across groups suggests Chinchilla exhibits bias.

# 5. Discussion & Conclusion

The trend so far in large language model training has been to increase the model size, often without increasing the number of training tokens. The largest dense transformer, MT-NLG 530B, is now over 3× larger than GPT-3’s 170 billion parameters from just two years ago. However, this model, as well as the majority of existing large models, have all been trained for a comparable number of tokens—around 300 billion. While the desire to train these mega-models has led to substantial engineering innovation, we hypothesize that the race to train larger and larger models is resulting in models that are substantially underperforming compared to what could be achieved with the same compute budget.

We propose three predictive approaches towards optimally setting model size and training duration, based on the outcome of over 400 training runs. All three approaches predict that Gopher is substantially over-sized and estimate that for the same compute budget a smaller model trained on more data will perform better. We directly test this hypothesis by training Chinchilla, a 70B parameter model, and show that it outperforms Gopher and even larger models on nearly every measured evaluation task.

Whilst our method allows us to make predictions on how to scale large models when given additional compute, there are several limitations. Due to the cost of training large models, we only have two comparable training runs at large scale (Chinchilla and Gopher), and we do not have additional tests at intermediate scales. Furthermore, we assume that the efficient computational frontier can be described by a power-law relationship between the compute budget, model size, and number of training tokens. However, we observe some concavity in log $\left( N _ { o p t } \right)$ at high compute budgets (see Appendix E). This suggests that we may still be overestimating the optimal size of large models. Finally, the training runs for our analysis have all been trained on less than an epoch of data; future work may consider the multiple epoch regime. Despite these limitations, the comparison of Chinchilla to Gopher validates our performance predictions, that have thus enabled training a better (and more

lightweight) model at the same compute budget.

Though there has been significant recent work allowing larger and larger models to be trained, our analysis suggests an increased focus on dataset scaling is needed. Speculatively, we expect that scaling to larger and larger datasets is only beneficial when the data is high-quality. This calls for responsibly collecting larger datasets with a high focus on dataset quality. Larger datasets will require extra care to ensure train-test set overlap is properly accounted for, both in the language modelling loss but also with downstream tasks. Finally, training for trillions of tokens introduces many ethical and privacy concerns. Large datasets scraped from the web will contain toxic language, biases, and private information. With even larger datasets being used, the quantity (if not the frequency) of such information increases, which makes dataset introspection all the more important. Chinchilla does suffer from bias and toxicity but interestingly it seems less affected than Gopher. Better understanding how performance of large language models and toxicity interact is an important future research question.

While we have applied our methodology towards the training of auto-regressive language models, we expect that there is a similar trade-off between model size and the amount of data in other modalities. As training large models is very expensive, choosing the optimal model size and training steps beforehand is essential. The methods we propose are easy to reproduce in new settings.

# 6. Acknowledgements

We’d like to thank Jean-baptiste Alayrac, Kareem Ayoub, Chris Dyer, Nando de Freitas, Demis Hassabis, Geoffrey Irving, Koray Kavukcuoglu, Nate Kushman and Angeliki Lazaridou for useful comments on the manuscript. We’d like to thank Andy Brock, Irina Higgins, Michela Paganini, Francis Song, and other colleagues at DeepMind for helpful discussions. We are also very grateful to the JAX and XLA team for their support and assistance.

# References

M. Artetxe, S. Bhosale, N. Goyal, T. Mihaylov, M. Ott, S. Shleifer, X. V. Lin, J. Du, S. Iyer, R. Pasunuru, G. Anantharaman, X. Li, S. Chen, H. Akin, M. Baines, L. Martin, X. Zhou, P. S. Koura, B. O’Horo, J. Wang, L. Zettlemoyer, M. Diab, Z. Kozareva, and V. Stoyanov. Efficient Large Scale Language Modeling with Mixtures of Experts. arXiv:2112.10684, 2021.   
E. M. Bender, T. Gebru, A. McMillan-Major, and S. Shmitchell. On the dangers of stochastic parrots: Can language models be too big? In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, pages 610–623, 2021.   
BIG-bench collaboration. Beyond the imitation game: Measuring and extrapolating the capabilities of language models. In preparation, 2021. URL https://github.com/google/BIG-bench/.   
Y. Bisk, R. Zellers, J. Gao, Y. Choi, et al. PIQA: Reasoning about physical commonsense in natural language. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 7432–7439, 2020.   
S. Borgeaud, A. Mensch, J. Hoffmann, T. Cai, E. Rutherford, K. Millican, G. van den Driessche, J.-B. Lespiau, B. Damoc, A. Clark, D. de Las Casas, A. Guy, J. Menick, R. Ring, T. Hennigan, S. Huang, L. Maggiore, C. Jones, A. Cassirer, A. Brock, M. Paganini, G. Irving, O. Vinyals, S. Osindero, K. Simonyan, J. W. Rae, E. Elsen, and L. Sifre. Improving language models by retrieving from trillions of tokens. arXiv 2112.04426, 2021.

J. Bradbury, R. Frostig, P. Hawkins, M. J. Johnson, C. Leary, D. Maclaurin, G. Necula, A. Paszke, J. VanderPlas, S. Wanderman-Milne, and Q. Zhang. JAX: composable transformations of Python+NumPy programs. 2018. URL http://github.com/google/jax.   
T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, S. Agarwal, A. Herbert-Voss, G. Krueger, T. Henighan, R. Child, A. Ramesh, D. Ziegler, J. Wu, C. Winter, C. Hesse, M. Chen, E. Sigler, M. Litwin, S. Gray, B. Chess, J. Clark, C. Berner, S. McCandlish, A. Radford, I. Sutskever, and D. Amodei. Language models are few-shot learners. In H. Larochelle, M. Ranzato, R. Hadsell, M. F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb49674 18bfb8ac142f64a-Paper.pdf.   
S. Bubeck. Convex Optimization: Algorithms and Complexity. Foundations and Trends in Machine Learning, 8(3-4):231–357, 2015. URL http://www.nowpublishers.com/article/Detail s/MAL-050.   
A. Clark, D. d. l. Casas, A. Guy, A. Mensch, M. Paganini, J. Hoffmann, B. Damoc, B. Hechtman, T. Cai, S. Borgeaud, G. v. d. Driessche, E. Rutherford, T. Hennigan, M. Johnson, K. Millican, A. Cassirer, C. Jones, E. Buchatskaya, D. Budden, L. Sifre, S. Osindero, O. Vinyals, J. Rae, E. Elsen, K. Kavukcuoglu, and K. Simonyan. Unified scaling laws for routed language models, 2022. URL https://arxiv.org/abs/2202.01169.   
C. Clark, K. Lee, M.-W. Chang, T. Kwiatkowski, M. Collins, and K. Toutanova. Boolq: Exploring the surprising difficulty of natural yes/no questions. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 2924–2936, 2019.   
N. Du, Y. Huang, A. M. Dai, S. Tong, D. Lepikhin, Y. Xu, M. Krikun, Y. Zhou, A. W. Yu, O. Firat, B. Zoph, L. Fedus, M. Bosma, Z. Zhou, T. Wang, Y. E. Wang, K. Webster, M. Pellat, K. Robinson, K. Meier-Hellstern, T. Duke, L. Dixon, K. Zhang, Q. V. Le, Y. Wu, Z. Chen, and C. Cui. Glam: Efficient scaling of language models with mixture-of-experts, 2021. URL https://arxiv.org/abs/2112.06905.   
W. Fedus, B. Zoph, and N. Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. arXiv preprint arXiv:2101.03961, 2021.   
L. Gao, S. Biderman, S. Black, L. Golding, T. Hoppe, C. Foster, J. Phang, H. He, A. Thite, N. Nabeshima, S. Presser, and C. Leahy. The Pile: An 800GB dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.   
S. Gehman, S. Gururangan, M. Sap, Y. Choi, and N. A. Smith. RealToxicityPrompts: Evaluating neural toxic degeneration in language models. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 3356–3369, Online, Nov. 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.findings-emnlp.301. URL https://aclanthology.org/2 020.findings-emnlp.301.   
K. Guu, K. Lee, Z. Tung, P. Pasupat, and M.-W. Chang. REALM: Retrieval-augmented language model pre-training, 2020.   
D. Hendrycks, C. Burns, S. Basart, A. Zou, M. Mazeika, D. Song, and J. Steinhardt. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300, 2020.   
T. Hennigan, T. Cai, T. Norman, and I. Babuschkin. Haiku: Sonnet for JAX. 2020. URL http: //github.com/deepmind/dm-haiku.

D. Hernandez, J. Kaplan, T. Henighan, and S. McCandlish. Scaling laws for transfer, 2021.   
P. J. Huber. Robust Estimation of a Location Parameter. The Annals of Mathematical Statistics, 35 (1):73–101, Mar. 1964. ISSN 0003-4851, 2168-8990. doi: 10.1214/aoms/1177703732. URL https://projecteuclid.org/journals/annals-of-mathematical-statistics/vol ume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/11 77703732.full.   
G. Izacard and E. Grave. Distilling knowledge from reader to retriever for question answering, 2020.   
M. Joshi, E. Choi, D. Weld, and L. Zettlemoyer. TriviaQA: A Large Scale Distantly Supervised Challenge Dataset for Reading Comprehension. arXiv e-prints, art. arXiv:1705.03551, 2017.   
N. P. Jouppi, C. Young, N. Patil, D. Patterson, G. Agrawal, R. Bajwa, S. Bates, S. Bhatia, N. Boden, A. Borchers, R. Boyle, P.-l. Cantin, C. Chao, C. Clark, J. Coriell, M. Daley, M. Dau, J. Dean, B. Gelb, T. V. Ghaemmaghami, R. Gottipati, W. Gulland, R. Hagmann, C. R. Ho, D. Hogberg, J. Hu, R. Hundt, D. Hurt, J. Ibarz, A. Jaffey, A. Jaworski, A. Kaplan, H. Khaitan, D. Killebrew, A. Koch, N. Kumar, S. Lacy, J. Laudon, J. Law, D. Le, C. Leary, Z. Liu, K. Lucke, A. Lundin, G. MacKean, A. Maggiore, M. Mahony, K. Miller, R. Nagarajan, R. Narayanaswami, R. Ni, K. Nix, T. Norrie, M. Omernick, N. Penukonda, A. Phelps, J. Ross, M. Ross, A. Salek, E. Samadiani, C. Severn, G. Sizikov, M. Snelham, J. Souter, D. Steinberg, A. Swing, M. Tan, G. Thorson, B. Tian, H. Toma, E. Tuttle, V. Vasudevan, R. Walter, W. Wang, E. Wilcox, and D. H. Yoon. In-datacenter performance analysis of a tensor processing unit. In Proceedings of the 44th Annual International Symposium on Computer Architecture, ISCA ’17, page 1–12, New York, NY, USA, 2017. Association for Computing Machinery. ISBN 9781450348928. doi: 10.1145/3079856.3080246. URL https://doi.org/10.1145/3079856.3080246.   
J. Kaplan, S. McCandlish, T. Henighan, T. B. Brown, B. Chess, R. Child, S. Gray, A. Radford, J. Wu, and D. Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.   
D. P. Kingma and J. Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.   
T. Kudo and J. Richardson. SentencePiece: A simple and language independent subword tokenizer and detokenizer for neural text processing. arXiv preprint arXiv:1808.06226, 2018.   
T. Kwiatkowski, J. Palomaki, O. Redfield, M. Collins, A. Parikh, C. Alberti, D. Epstein, I. Polosukhin, M. Kelcey, J. Devlin, K. Lee, K. N. Toutanova, L. Jones, M.-W. Chang, A. Dai, J. Uszkoreit, Q. Le, and S. Petrov. Natural questions: a benchmark for question answering research. Transactions of the Association of Computational Linguistics, 2019.   
G. Lai, Q. Xie, H. Liu, Y. Yang, and E. Hovy. RACE: Large-scale ReAding comprehension dataset from examinations. In Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing, pages 785–794, Copenhagen, Denmark, Sept. 2017. Association for Computational Linguistics. doi: 10.18653/v1/D17-1082. URL https://aclanthology.org/D17-1082.   
Y. Levine, N. Wies, O. Sharir, H. Bata, and A. Shashua. The depth-to-width interplay in self-attention. arXiv preprint arXiv:2006.12467, 2020.   
P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel, S. Riedel, and D. Kiela. Retrieval-augmented generation for knowledge-intensive nlp tasks. In Advances in Neural Information Processing Systems, volume 33, pages 9459–9474, 2020.

O. Lieber, O. Sharir, B. Lenz, and Y. Shoham. Jurassic-1: Technical details and evaluation. White Paper. AI21 Labs, 2021.   
S. Lin, J. Hilton, and O. Evans. TruthfulQA: Measuring how models mimic human falsehoods. arXiv preprint arXiv:2109.07958, 2021.   
I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7.   
S. McCandlish, J. Kaplan, D. Amodei, and O. D. Team. An empirical model of large-batch training, 2018.   
S. Merity, C. Xiong, J. Bradbury, and R. Socher. Pointer sentinel mixture models. International Conference on Learning Representations, 2017.   
M. Mitchell, S. Wu, A. Zaldivar, P. Barnes, L. Vasserman, B. Hutchinson, E. Spitzer, I. D. Raji, and T. Gebru. Model cards for model reporting. In Proceedings of the conference on fairness, accountability, and transparency, pages 220–229, 2019.   
J. Nocedal. Updating Quasi-Newton Matrices with Limited Storage. Mathematics of Computation, 35(151):773–782, 1980. ISSN 0025-5718. doi: 10.2307/2006193. URL https://www.jstor. org/stable/2006193.   
D. Paperno, G. Kruszewski, A. Lazaridou, Q. N. Pham, R. Bernardi, S. Pezzelle, M. Baroni, G. Boleda, and R. Fernández. The LAMBADA dataset: Word prediction requiring a broad discourse context, 2016.   
J. Rae, S. Borgeaud, T. Cai, K. Millican, J. Hoffmann, F. Song, J. Aslanides, S. Henderson, R. Ring, S. Young, E. Rutherford, T. Hennigan, J. Menick, A. Cassirer, R. Powell, G. van den Driessche, L. A. Hendricks, M. Rauh, P.-S. Huang, A. Glaese, J. Welbl, S. Dathathri, S. Huang, J. Uesato, J. Mellor, I. Higgins, A. Creswell, N. McAleese, A. Wu, E. Elsen, S. Jayakumar, E. Buchatskaya, D. Budden, E. Sutherland, K. Simonyan, M. Paganini, L. Sifre, L. Martens, X. L. Li, A. Kuncoro, A. Nematzadeh, E. Gribovskaya, D. Donato, A. Lazaridou, A. Mensch, J.-B. Lespiau, M. Tsimpoukelli, N. Grigorev, D. Fritz, T. Sottiaux, M. Pajarskas, T. Pohlen, Z. Gong, D. Toyama, C. de Masson d’Autume, Y. Li, T. Terzi, I. Babuschkin, A. Clark, D. de Las Casas, A. Guy, J. Bradbury, M. Johnson, L. Weidinger, I. Gabriel, W. Isaac, E. Lockhart, S. Osindero, L. Rimell, C. Dyer, O. Vinyals, K. Ayoub, J. Stanway, L. Bennett, D. Hassabis, K. Kavukcuoglu, and G. Irving. Scaling language models: Methods, analysis & insights from training Gopher. arXiv 2112.11446, 2021.   
J. W. Rae, A. Potapenko, S. M. Jayakumar, T. P. Lillicrap, K. Choromanski, V. Likhosherstov, D. Dohan, X. Song, A. Gane, T. Sarlos, et al. Compressive transformers for long-range sequence modelling. Advances in Neural Information Processing Systems, 33:6154–6158, 2020.   
C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020a. URL http://jmlr.org/papers/v21/20-074.html.   
C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of Machine Learning Research, 21(140):1–67, 2020b.   
S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–16. IEEE, 2020.

H. Robbins and S. Monro. A Stochastic Approximation Method. The Annals of Mathematical Statistics, 22(3):400–407, Sept. 1951.   
R. Rudinger, J. Naradowsky, B. Leonard, and B. Van Durme. Gender bias in coreference resolution. In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, New Orleans, Louisiana, June 2018. Association for Computational Linguistics.   
K. Sakaguchi, R. Le Bras, C. Bhagavatula, and Y. Choi. Winogrande: An adversarial winograd schema challenge at scale. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 34, pages 8732–8740, 2020.   
M. Sap, H. Rashkin, D. Chen, R. LeBras, and Y. Choi. SocialIQA: Commonsense reasoning about social interactions. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing, 2019.   
C. J. Shallue, J. Lee, J. Antognini, J. Sohl-Dickstein, R. Frostig, and G. E. Dahl. Measuring the effects of data parallelism on neural network training. arXiv preprint arXiv:1811.03600, 2018.   
J. W. Siegel and J. Xu. Approximation rates for neural networks with general activation functions. Neural Networks, 128:313–321, Aug. 2020. URL https://www.sciencedirect.com/scienc e/article/pii/S0893608020301891.   
S. Smith, M. Patwary, B. Norick, P. LeGresley, S. Rajbhandari, J. Casper, Z. Liu, S. Prabhumoye, G. Zerveas, V. Korthikanti, E. Zhang, R. Child, R. Y. Aminabadi, J. Bernauer, X. Song, M. Shoeybi, Y. He, M. Houston, S. Tiwary, and B. Catanzaro. Using Deepspeed and Megatron to Train Megatronturing NLG 530b, A Large-Scale Generative Language Model. arXiv preprint arXiv:2201.11990, 2022.   
J. Steinhardt. Updates and lessons from AI forecasting, 2021. URL https://bounded-regret.g host.io/ai-forecasting/.   
Y. Tay, M. Dehghani, J. Rao, W. Fedus, S. Abnar, H. W. Chung, S. Narang, D. Yogatama, A. Vaswani, and D. Metzler. Scale efficiently: Insights from pre-training and fine-tuning transformers, 2021.   
R. Thoppilan, D. D. Freitas, J. Hall, N. Shazeer, A. Kulshreshtha, H.-T. Cheng, A. Jin, T. Bos, L. Baker, Y. Du, Y. Li, H. Lee, H. S. Zheng, A. Ghafouri, M. Menegali, Y. Huang, M. Krikun, D. Lepikhin, J. Qin, D. Chen, Y. Xu, Z. Chen, A. Roberts, M. Bosma, Y. Zhou, C.-C. Chang, I. Krivokon, W. Rusch, M. Pickett, K. Meier-Hellstern, M. R. Morris, T. Doshi, R. D. Santos, T. Duke, J. Soraker, B. Zevenbergen, V. Prabhakaran, M. Diaz, B. Hutchinson, K. Olson, A. Molina, E. Hoffman-John, J. Lee, L. Aroyo, R. Rajakumar, A. Butryna, M. Lamm, V. Kuzmina, J. Fenton, A. Cohen, R. Bernstein, R. Kurzweil, B. Aguera-Arcas, C. Cui, M. Croak, E. Chi, and Q. Le. LaMDA: Language models for dialog applications, 2022.   
A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. In Advances in neural information processing systems, pages 5998–6008, 2017.   
L. Weidinger, J. Mellor, M. Rauh, C. Griffin, J. Uesato, P.-S. Huang, M. Cheng, M. Glaese, B. Balle, A. Kasirzadeh, Z. Kenton, S. Brown, W. Hawkins, T. Stepleton, C. Biles, A. Birhane, J. Haas, L. Rimell, L. A. Hendricks, W. Isaac, S. Legassick, G. Irving, and I. Gabriel. Ethical and social risks of harm from language models. arXiv submission, 2021.

J. Welbl, A. Glaese, J. Uesato, S. Dathathri, J. Mellor, L. A. Hendricks, K. Anderson, P. Kohli, B. Coppin, and P.-S. Huang. Challenges in detoxifying language models. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 2447–2469, Punta Cana, Dominican Republic, Nov. 2021. Association for Computational Linguistics. URL https://aclanthology.org/2021. findings-emnlp.210.   
A. Xu, E. Pathak, E. Wallace, S. Gururangan, M. Sap, and D. Klein. Detoxifying language models risks marginalizing minority voices. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2390–2397, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021 .naacl-main.190. URL https://aclanthology.org/2021.naacl-main.190.   
G. Yang, E. J. Hu, I. Babuschkin, S. Sidor, X. Liu, D. Farhi, N. Ryder, J. Pachocki, W. Chen, and J. Gao. Tuning large neural networks via zero-shot hyperparameter transfer. In A. Beygelzimer, Y. Dauphin, P. Liang, and J. W. Vaughan, editors, Advances in Neural Information Processing Systems, 2021. URL https://openreview.net/forum?id=Bx6qKuBM2AD.   
R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi. HellaSwag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 2019.   
G. Zhang, L. Li, Z. Nado, J. Martens, S. Sachdeva, G. Dahl, C. Shallue, and R. B. Grosse. Which algorithmic choices matter at which batch sizes? insights from a noisy quadratic model. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https: //proceedings.neurips.cc/paper/2019/file/e0eacd983971634327ae1819ea8b621 4-Paper.pdf.   
B. Zoph, I. Bello, S. Kumar, N. Du, Y. Huang, J. Dean, N. Shazeer, and W. Fedus. Designing effective sparse expert models, 2022.

# Appendix

# A. Training dataset

In Table A1 we show the training dataset makeup used for Chinchilla and all scaling runs. Note that both the MassiveWeb and Wikipedia subsets are both used for more than one epoch. 

<table><tr><td></td><td>Disk Size</td><td>Documents</td><td>Sampling proportion</td><td>Epochs in 1.4T tokens</td></tr><tr><td>MassiveWeb</td><td>1.9 TB</td><td>604M</td><td>45% (48%)</td><td>1.24</td></tr><tr><td>Books</td><td>2.1 TB</td><td>4M</td><td>30% (27%)</td><td>0.75</td></tr><tr><td>C4</td><td>0.75 TB</td><td>361M</td><td>10% (10%)</td><td>0.77</td></tr><tr><td>News</td><td>2.7 TB</td><td>1.1B</td><td>10% (10%)</td><td>0.21</td></tr><tr><td>GitHub</td><td>3.1 TB</td><td>142M</td><td>4% (3%)</td><td>0.13</td></tr><tr><td>Wikipedia</td><td>0.001 TB</td><td>6M</td><td>1% (2%)</td><td>3.40</td></tr></table>

Table A1 | MassiveText data makeup. For each subset of MassiveText, we list its total disk size, the number of documents and the sampling proportion used during training—we use a slightly different distribution than in Rae et al. (2021) (shown in parenthesis). In the rightmost column show the number of epochs that are used in 1.4 trillion tokens.

# B. Optimal cosine cycle length

One key assumption is made on the cosine cycle length and the corresponding learning rate drop (we use a 10× learning rate decay in line with Rae et al. (2021)).9 We find that setting the cosine cycle length too much longer than the target number of training steps results in sub-optimally trained models, as shown in Figure A1. As a result, we assume that an optimally trained model will have the cosine cycle length correctly calibrated to the maximum number of steps, given the FLOP budget; we follow this rule in our main analysis.

# C. Consistency of scaling results across datasets

We show scaling results from an IsoFLOP (Approach 2) analysis after training on two different datasets: C4 (Raffel et al., 2020b) and GitHub code (we show results with data from Rae et al. (2021)), results are shown in Table A2. For both set of experiments using subsets of MassiveText, we use the same tokenizer as the MassiveText experiments.

We find that the scaling behaviour on these datasets is very similar to what we found on MassiveText, as shown in Figure A2 and Table A2. This suggests that our results are independent of the dataset as long as one does not train for more than one epoch.

![](images/0cff34d0a90da36a51feecc809dcbaaa841b5813f93319aac5a5b977260fcdc5.jpg)

![](images/d8c8425a81f996236651ec62b05ad2b9faa6bbb7244e7ee87a6f741e4a1e603a.jpg)

![](images/d17e75662c499870984cb274e7f49bfdfb04e4d1729172fc0396062f30026982.jpg)

![](images/0f45e4b6675d01d88f62a0ae9bd7b9ba6e15bb1f3009a0f8284e9ea771b54917.jpg)

![](images/2a54701f55e51094139405a70d79ae7d40a0c4538030f07783e4e6913c2bd943.jpg)

![](images/c8245cfd14f7f877b95ffeddc2303cbf10399a7c15645f4906346bec2e67284f.jpg)

Figure A1 | Grid over cosine cycle length. We show 6 curves with the cosine cycle length set to 1, 1.1, 1.25, 1.5, 2, and 5× longer than the target number of training steps. When the cosine cycle length is too long, and the learning rate does not drop appropriately, then performance is impaired. We find that overestimating the number of training steps beyond 25% leads to clear drops in performance. We show results where we have set the number of training steps to two different values (top and bottom).   
![](images/33625298087f07755d681642b81cd89927b2cbe427a290e8c1a3cbeee24a5686.jpg)

![](images/38ba45911bd732567c53a825fbd00f8fd48b49259f6d4745154424b08f537ced.jpg)

![](images/78cda6956212a21dbe1aa89a937e640d06b6c78370d26ca4748345d13e2569ab.jpg)

![](images/66c79043cb7da4daa4d6be9bb4fdc25f8ade22a18e6a27deaa71778f217cc249.jpg)

![](images/be3aa8263121a2152bb687165b81161d81a20c4bdec328e8f6ecf69d700fad8f.jpg)

![](images/d436588b10ea960ad21b01a6a92461bc399a37b92031c856e8df7c2e5a211956.jpg)  
Figure A2 | C4 and GitHub IsoFLOP curves. Using the C4 dataset (Raffel et al., 2020b) and a GitHub dataset (Rae et al., 2021), we generate 4 IsoFLOP profiles and show the parameter and token count scaling, as in Figure 3. Scaling coefficients are shown in Table A2.

<table><tr><td>Approach</td><td>Coef. a where  $N_{opt} \propto C^{a}$ </td><td>Coef. b where  $D_{opt} \propto C^{b}$ </td></tr><tr><td>C4</td><td>0.50</td><td>0.50</td></tr><tr><td>GitHub</td><td>0.53</td><td>0.47</td></tr><tr><td>Kaplan et al. (2020)</td><td>0.73</td><td>0.27</td></tr></table>

Table A2 | Estimated parameter and data scaling with increased training compute on two alternate datasets. The listed values are the exponents, ?? and $b ,$ on the relationship $N _ { o p t } \propto C ^ { a }$ and $D _ { o p t } \propto C ^ { b }$ . Using IsoFLOP profiles, we estimate the scaling on two different datasets.

# D. Details on the scaling analyses

# D.1. Approach 1: Fixing model sizes and varying training sequences

We use a maximum learning rate of $2 \times 1 0 ^ { - 4 }$ for the smallest models and $1 . 2 5 \times 1 0 ^ { - 4 }$ for the largest models. In all cases, the learning rate drops by a factor of 10× during training, using a cosine schedule. We make the assumption that the cosine cycle length should be approximately matched to the number of training steps. We find that when the cosine cycle overshoots the number of training steps by more than 25%, performance is noticeably degraded—see Figure A1.10 We use Gaussian smoothing with a window length of 10 steps to smooth the training curve.

# D.2. Approach 3: Parametric fitting of the loss

In this section, we first show how Equation (2) can be derived. We repeat the equation below for clarity,

$$
\hat {L} (N, D) \triangleq E + \frac {A}{N ^ {\alpha}} + \frac {B}{D ^ {\beta}}, \tag {5}
$$

based on a decomposition of the expected risk between a function approximation term and an optimisation suboptimality term. We then give details on the optimisation procedure for fitting the parameters.

Loss decomposition. Formally, we consider the task of predicting the next token $y \in \mathcal { Y }$ based on the previous tokens in a sequence $x \in \mathcal { Y } ^ { s }$ , with ?? varying from 0 to $s _ { \mathrm { m a x } }$ —the maximum sequence length. We consider a distribution $P \in { \mathcal { D } } ( X \times y )$ of tokens in Y and their past in X. A predictor $f : X \to { \mathcal { D } } ( y )$ computes the probability of each token given the past sequence. The Bayes classifier, $f ^ { \star }$ , minimizes the cross-entropy of $f ( x )$ with the observed tokens $y ,$ with expectation taken on the whole data distribution. We let ?? be the expected risk

$$
L (f) \triangleq \mathbb {E} [ \log f (x) _ {y} ], \quad \text { and   set } \quad f ^ {\star} \triangleq \underset {f \in \mathcal {F} (\mathcal {X}, \mathcal {D} (\mathcal {Y}))} {\operatorname{argmin}} L (f). \tag {6}
$$

The set of all transformers of size ??, that we denote $\mathcal { H } _ { N }$ , forms a subset of all functions that map sequences to distributions of tokens $\chi \to { \mathcal { D } } ( y )$ . Fitting a transformer of size ?? on the expected risk $L ( f )$ amounts to minimizing such risk on a restricted functional space

$$
f _ {N} \triangleq \underset {f \in \mathcal {H} _ {N}} {\operatorname{argmin}} L (f). \tag {7}
$$

When we observe a dataset $( x _ { i } , y _ { i } ) _ { i _ { i \in [ 1 , D ] } }$ of size $D ,$ we do not have access to $\mathbb { E } _ { P }$ , but instead to the empirical expectation $\hat { \mathbb { E } } _ { D }$ over the empirical distribution $\hat { P } _ { D }$ . What happens when we are given ?? datapoints that we can only see once, and when we constrain the size of the hypothesis space to be ??-dimensional ? We are making steps toward minimizing the empirical risk within a finite-dimensional functional space $\mathcal { H } _ { N }$ :

$$
\hat {L} _ {D} (f) \triangleq \hat {\mathbb {E}} _ {D} [ \log f (x) _ {y} ], \quad \text {   setting   } \quad \hat {f} _ {N, D} \triangleq \underset {f \in \mathcal {H} _ {N}} {\operatorname{argmin}} \hat {L} _ {D} (f). \tag {8}
$$

We are never able to obtain $\hat { f } _ { N , D }$ as we typically perform a single epoch over the dataset of size ??. Instead, be obtain $\bar { f } _ { N , D } ,$ , which is the result of applying a certain number of gradient steps based on the ?? datapoints—the number of steps to perform depends on the gradient batch size, for which we use well-tested heuristics.

Using the Bayes-classifier $f ^ { \star }$ , the expected-risk minimizer $f _ { N }$ and the “single-epoch empirical-risk minimizer” $\bar { f } _ { N , D } ,$ we can finally decompose the loss $L ( N , D )$ into

$$
L (N, D) \triangleq L (\bar {f} _ {N, D}) = L (f ^ {\star}) + \left(L (f _ {N}) - L (f ^ {\star})\right) + \left(L (\bar {f} _ {N, D}) - L (f _ {N})\right). \tag {9}
$$

The loss comprises three terms: the Bayes risk, i.e. the minimal loss achievable for next-token prediction on the full distribution ??, a.k.a the “entropy of natural text.”; a functional approximation term that depends on the size of the hypothesis space; finally, a stochastic approximation term that captures the suboptimality of minimizing $\hat { L } _ { D }$ instead of $L ,$ and of making a single epoch on the provided dataset.

Expected forms of the loss terms. In the decomposition (9), the second term depends entirely on the number of parameters ?? that defines the size of the functional approximation space. On the set of two-layer neural networks, it is expected to be proportional to $\scriptstyle { \frac { 1 } { N ^ { 1 / 2 } } }$ (Siegel and Xu, 2020). Finally, given that it corresponds to early stopping in stochastic first order methods, the third term should scale as the convergence rate of these methods, which is lower-bounded by $\scriptstyle { \frac { 1 } { D ^ { 1 / 2 } } }$ (Robbins and Monro, 1951) (and may attain the bound). This convergence rate is expected to be dimension free (see e.g. Bubeck, 2015, for a review) and depends only on the loss smoothness; hence we assume that the second term only depends on ?? in (2). Empirically, we find after fitting (2) that

$$
L (N, D) = E + \frac {A}{N ^ {0 . 3 4}} + \frac {B}{D ^ {0 . 2 8}}, \tag {10}
$$

with $E = 1 . 6 9 , A = 4 0 6 . 4 , B = 4 1 0 . 7$ . We note that the parameter/data coefficients are both lower than $\begin{array} { l } { { \frac { 1 } { 2 } } . } \end{array}$ ; this is expected for the data-efficiency coefficient (but far from the known lower-bound). Future models and training approaches should endeavor to increase these coefficients.

Fitting the decomposition to data. We effectively minimize the following problem

$$
\min _ {a, b, e, \alpha , \beta} \sum_ {\text { Run } i} \operatorname{Huber} _ {\delta} \left(\mathrm{LSE} (a - \alpha \log N _ {i}, b - \beta \log D _ {i}, e) - \log L _ {i}\right), \tag {11}
$$

where ?????? is the log-sum-exp operator. We then set ??, $B , E = \exp ( a ) , \exp ( b ) , \exp ( e )$ .

We use the LBFGS algorithm to find local minima of the objective above, started on a grid of initialisation given by: $\alpha \in \{ 0 , , 0 . 5 , \ldots , 2 . \} , \beta \in \{ 0 . , 0 . 5 , \ldots , 2 . \} , e \in \{ - 1 . , - . 5 , \ldots , 1 . \} , a \in \{ 0 . , 0 . 5 , \ldots , 2 . \} \quad .$ $\{ 0 , 5 , \ldots , 2 5 \}$ , and $b \in \{ 0 , 5 , \ldots , 2 5 \}$ . We find that the optimal initialisation is not on the boundary of our initialisation sweep.

We use $\delta = 1 0 ^ { - 3 }$ for the Huber loss. We find that using larger values of $\delta$ pushes the model to overfit the small compute regime and poorly predict held-out data from larger runs. We find that using a ?? smaller than $1 0 ^ { - 3 }$ does not impact the resulting predictions.

# D.3. Predicted compute optimal frontier for all three methods

For Approaches 2 and 3, we show the estimated model size and number of training tokens for a variety of compute budgets in Table A3. We plot the predicted number of tokens and parameters for a variety of FLOP budgets for the three methods in Figure A3.

<table><tr><td></td><td colspan="2">Approach 2</td><td colspan="2">Approach 3</td></tr><tr><td>Parameters</td><td>FLOPs</td><td>Tokens</td><td>FLOPs</td><td>Tokens</td></tr><tr><td>400 Million</td><td>1.84e+19</td><td>7.7 Billion</td><td>2.21e+19</td><td>9.2 Billion</td></tr><tr><td>1 Billion</td><td>1.20e+20</td><td>20.0 Billion</td><td>1.62e+20</td><td>27.1 Billion</td></tr><tr><td>10 Billion</td><td>1.32e+22</td><td>219.5 Billion</td><td>2.46e+22</td><td>410.1 Billion</td></tr><tr><td>67 Billion</td><td>6.88e+23</td><td>1.7 Trillion</td><td>1.71e+24</td><td>4.1 Trillion</td></tr><tr><td>175 Billion</td><td>4.54e+24</td><td>4.3 Trillion</td><td>1.26e+24</td><td>12.0 Trillion</td></tr><tr><td>280 Billion</td><td>1.18e+25</td><td>7.1 Trillion</td><td>3.52e+25</td><td>20.1 Trillion</td></tr><tr><td>520 Billion</td><td>4.19e+25</td><td>13.4 Trillion</td><td>1.36e+26</td><td>43.5 Trillion</td></tr><tr><td>1 Trillion</td><td>1.59e+26</td><td>26.5 Trillion</td><td>5.65e+26</td><td>94.1 Trillion</td></tr><tr><td>10 Trillion</td><td>1.75e+28</td><td>292.0 Trillion</td><td>8.55e+28</td><td>1425.5 Trillion</td></tr></table>

Table A3 | Estimated optimal training FLOPs and training tokens for various model sizes. Analogous to Table 3, we show the model size/token count projections from Approaches 2 and 3 for various compute budgets.

![](images/aac9f068057b4ed7fa497d6f4032574558e5680b558b54ae17d4926e15b2cf40.jpg)  
Figure A3 | Optimal number of tokens and parameters for a training FLOP budget. For a fixed FLOP budget, we show the optimal number of tokens and parameters as predicted by Approaches 1, 2, and 3. For an alternate representation, see Figure 1.

# D.4. Small-scale comparison to Kaplan et al. (2020)

For $1 0 ^ { 2 1 }$ FLOPs, we perform a head-to-head comparison of a model predicted by Approach 1 and that predicted by Kaplan et al. (2020). For both models, we use a batch size of 0.5M tokens and a maximum learning rate of $1 . 5 \times 1 0 ^ { - 4 }$ that decays by 10×. From Kaplan et al. (2020), we find that the optimal model size should be 4.68 billion parameters. From our approach 1, we estimate a 2.86 billion parameter model should be optimal. We train a 4.74 billion parameter and a 2.80 billion parameter transformer to test this hypothesis, using the same depth-to-width ratio to avoid as many confounding factors as possible. We find that our predicted model outperforms the model predicted by Kaplan et al. (2020) as shown in Figure A4.

![](images/b9382514b889d7bcf0709262695e7ba4b7f849f596439621604abc846d99f045.jpg)

![](images/a1b2e040050aff326f42d79d7a527e6917c95ecbf9b3347b2e97e0da17ee9373.jpg)  
Figure A4 | Comparison to Kaplan et al. (2020) at $1 0 ^ { 2 1 }$ FLOPs. We train 2.80 and 4.74 billion parameter transformers predicted as optimal for $1 0 ^ { 2 1 }$ FLOPs by Approach 1 and by Kaplan et al. (2020). We find that our prediction results in a more performant model at the end of training.

# E. Curvature of the FLOP-loss frontier

We observe that as models increase there is a curvature in the FLOP-minimal loss frontier. This means that projections from very small models lead to different predictions than those from larger models. In Figure A5 we show linear fits using the first, middle, and final third of frontier-points. In this work, we do not take this in to account and we leave this as interesting future work as it suggests that even smaller models may be optimal for large FLOP budgets.

# F. FLOPs computation

We include all training FLOPs, including those contributed to by the embedding matrices, in our analysis. Note that we also count embeddings matrices in the total parameter count. For large models the FLOP and parameter contribution of embedding matrices is small. We use a factor of 2 to describe the multiply accumulate cost. For the forward pass, we consider contributions from:

• Embeddings   
– 2 × seq\_len × vocab\_size × d\_model

• Attention (Single Layer)

– Key, query and value projections: $2 \times 3 \times \mathsf { s e q } .$ \_len × d\_model × (key\_size × num\_heads)

![](images/2b179c8fe3a7d2ec63ec19e5d137ca47cf9e5edbbd14a075b587087aa4a17d58.jpg)  
Figure A5 | Training curve envelopes. We fit to the first third (orange), the middle third (green), and the last third (blue) of all points along the loss frontier. We plot only a subset of the points.

– Key @ Query logits: 2 × seq\_len × seq\_len × (key\_size × num\_heads)   
– Softmax: 3 × num\_heads × seq\_len × seq\_len  
– Softmax @ query reductions: 2 × seq\_len × seq\_len × (key\_size × num\_heads)   
– Final Linear: 2 × seq\_len × (key\_size × num\_heads) × d\_model

• Dense Block (Single Layer)   
– 2 × seq\_len × (d\_model × ffw\_size + d\_model × ffw\_size)

• Final Logits

– 2 × seq\_len × d\_model × vocab\_size

• Total forward pass FLOPs: embeddings+num\_layers× (total\_attention+dense\_block) + logits

As in Kaplan et al. (2020) we assume that the backward pass has twice the FLOPs of the forward pass. We show a comparison between our calculation and that using the common approximation $C = 6 D N$ (Kaplan et al., 2020) where ?? is FLOPs, ?? is the number of training tokens, and ?? is the number of parameters in Table A4. We find the differences in FLOP calculation to be very small and they do not impact our analysis. Compared to the results presented in Rae et al. (2021), we use a slightly more

<table><tr><td>Parameters</td><td>num_layers</td><td>d_model</td><td>ffw_size</td><td>num_heads</td><td>k/q size</td><td>FLOP Ratio (Ours/6ND)</td></tr><tr><td>73M</td><td>10</td><td>640</td><td>2560</td><td>10</td><td>64</td><td>1.03</td></tr><tr><td>305M</td><td>20</td><td>1024</td><td>4096</td><td>16</td><td>64</td><td>1.10</td></tr><tr><td>552M</td><td>24</td><td>1280</td><td>5120</td><td>10</td><td>128</td><td>1.08</td></tr><tr><td>1.1B</td><td>26</td><td>1792</td><td>7168</td><td>14</td><td>128</td><td>1.04</td></tr><tr><td>1.6B</td><td>28</td><td>2048</td><td>8192</td><td>16</td><td>128</td><td>1.03</td></tr><tr><td>6.8B</td><td>40</td><td>3584</td><td>14336</td><td>28</td><td>128</td><td>0.99</td></tr></table>

Table A4 | FLOP comparison. For a variety of different model sizes, we show the ratio of the FLOPs that we compute per sequence to that using the 6?? ?? approximation.

accurate calculation giving a slightly different value $( 6 . 3 \times 1 0 ^ { 2 3 }$ compared to $5 . 7 6 \times 1 0 ^ { 2 3 } )$ .

# G. Other differences between Chinchilla and Gopher

Beyond differences in model size and number of training tokens, there are some additional minor differences between Chinchilla and Gopher. Specifically, Gopher was trained with Adam (Kingma and Ba, 2014) whereas Chinchilla was trained with AdamW (Loshchilov and Hutter, 2019). Furthermore, as discussed in Lessons Learned in Rae et al. (2021), Chinchilla stored a higher-precision copy of the weights in the sharded optimiser state.

We show comparisons of models trained with Adam and AdamW in Figure A6 and Figure A7. We find that, independent of the learning rate schedule, AdamW trained models outperform models trained with Adam. In Figure A6 we show a comparison of an 680 million parameter model trained

![](images/6b5eae7a2f8a0f7ae775c170f3e6d6fea57e7401faa4db2c959816b4adaaa094.jpg)

![](images/59a836ad4c832ce9f726c22413a2c32b7a0eaefe9f3863351896fdc08064143b.jpg)

![](images/7b1f201d5a6b44136adec9be07a3eee6c57c6afd892bcb27ef9b390c375fe725.jpg)  
Figure A6 | Comparison of other differences. Using an 680 million parameter model, we show a comparison between the setup used to train Gopher and Chinchilla— the change in optimiser and using a higher precision copy of the weights in the optimiser state. The setup used for Chinchilla (orange) clearly outperforms the setup used to train Gopher (green).

![](images/aa041135233cfc21ec5be02c95f371ade3905e6d97048b370b7af5b5bd9e8fa3.jpg)

![](images/ef4a8416bf6426f2a70c48a565aec42dcbcd2db6d18de48f1e68035151a6d3fc.jpg)

![](images/6c87803958184446b476f7ea73e90dac2267742b2b77d43e0f07384a2b11ca32.jpg)  
Figure A7 | Adam vs AdamW. For a 417M (blue) and 1.4B model (green), we find that training with AdamW improves performance over training with Adam.

with and without the higher precision copy of the weights and with Adam/AdamW for comparison.

# H. Results

# H.1. The Pile

In Table A5 we show the bits-per-byte (bpb) on The Pile (Gao et al., 2020) of Chinchilla, Gopher, and Jurassic-1. Chinchilla outperforms Gopher on all subsets. Jurassic-1 outperforms Chinchilla on 2 subsets— dm\_mathematics and ubuntu\_irc.

<table><tr><td>Subset</td><td>Chinchilla (70B)</td><td>Gopher (280B)</td><td>Jurassic-1 (170B)</td></tr><tr><td>pile_cc</td><td>0.667</td><td>0.691</td><td>0.669</td></tr><tr><td>pubmed_abstracts</td><td>0.559</td><td>0.578</td><td>0.587</td></tr><tr><td>stackexchange</td><td>0.614</td><td>0.641</td><td>0.655</td></tr><tr><td>github</td><td>0.337</td><td>0.377</td><td>0.358</td></tr><tr><td>openwebtext2</td><td>0.647</td><td>0.677</td><td>-</td></tr><tr><td>arxiv</td><td>0.627</td><td>0.662</td><td>0.680</td></tr><tr><td>uspto_backgrounds</td><td>0.526</td><td>0.546</td><td>0.537</td></tr><tr><td>freelaw</td><td>0.476</td><td>0.513</td><td>0.514</td></tr><tr><td>pubmed_central</td><td>0.504</td><td>0.525</td><td>0.579</td></tr><tr><td>dm_mathematics</td><td>1.111</td><td>1.142</td><td>1.037</td></tr><tr><td>hackernews</td><td>0.859</td><td>0.890</td><td>0.869</td></tr><tr><td>nih_exporter</td><td>0.572</td><td>0.590</td><td>0.590</td></tr><tr><td>opensubtitles</td><td>0.871</td><td>0.900</td><td>0.879</td></tr><tr><td>europarl</td><td>0.833</td><td>0.938</td><td>-</td></tr><tr><td>books3</td><td>0.675</td><td>0.712</td><td>0.835</td></tr><tr><td>philpapers</td><td>0.656</td><td>0.695</td><td>0.742</td></tr><tr><td>gutenberg_pg_19</td><td>0.548</td><td>0.656</td><td>0.890</td></tr><tr><td>bookcorpus2</td><td>0.714</td><td>0.741</td><td>-</td></tr><tr><td>ubuntu_irc</td><td>1.026</td><td>1.090</td><td>0.857</td></tr></table>

Table A5 | Bits-per-Byte on The Pile. We show the bpb on The Pile for Chinchilla compared to Gopher and Jurassic-1.

# H.2. MMLU

In Table A6 we show the performance of Chinchilla and Gopher on each subset of MMLU.

# H.3. Winogender Setup

We follow the same setup as in Rae et al. (2021). To test coreference resolution in Chinchilla, we input a sentence which includes a pronoun reference (e.g., “The librarian helped the child pick out a book because {pronoun} liked to encourage reading.”), then measure the probability of the model completing the sentence “‘{Pronoun}’ refers to the” with different sentence roles (“librarian” and “child” in this example). Each example is annotated with the correct pronoun resolution (the pronoun corresponds to the librarian in this example). Each sentence is tested with a female, male, and gender-neutral pronoun. An unbiased model would correctly predict which word the pronoun refers to regardless of pronoun gender.

# H.4. BIG-bench

In Table A7 we show Chinchilla and Gopher performance on each subset of BIG-bench that we consider.

# I. Model Card

We present the Chinchilla model card in Table A8, following the framework presented by Mitchell et al. (2019).

<table><tr><td>Task</td><td>Chinchilla</td><td>Gopher</td><td>Task</td><td>Chinchilla</td><td>Gopher</td></tr><tr><td>abstract_algebra</td><td>31.0</td><td>25.0</td><td>anatomy</td><td>70.4</td><td>56.3</td></tr><tr><td>astronomy</td><td>73.0</td><td>65.8</td><td>business_ethics</td><td>72.0</td><td>70.0</td></tr><tr><td>clinical_knowledge</td><td>75.1</td><td>67.2</td><td>college_biology</td><td>79.9</td><td>70.8</td></tr><tr><td>college_chemistry</td><td>51.0</td><td>45.0</td><td>college_computer_science</td><td>51.0</td><td>49.0</td></tr><tr><td>college_mathematics</td><td>32.0</td><td>37.0</td><td>college_medicine</td><td>66.5</td><td>60.1</td></tr><tr><td>college_physics</td><td>46.1</td><td>34.3</td><td>computer_security</td><td>76.0</td><td>65.0</td></tr><tr><td>conceptual_physics</td><td>67.2</td><td>49.4</td><td>econometrics</td><td>38.6</td><td>43.0</td></tr><tr><td>electrical_engineering</td><td>62.1</td><td>60.0</td><td>elementary_mathematics</td><td>41.5</td><td>33.6</td></tr><tr><td>formal_logic</td><td>33.3</td><td>35.7</td><td>global_facts</td><td>39.0</td><td>38.0</td></tr><tr><td>high_school_biology</td><td>80.3</td><td>71.3</td><td>high_school_chemistry</td><td>58.1</td><td>47.8</td></tr><tr><td>high_school_computer_science</td><td>58.0</td><td>54.0</td><td>high_school_european_history</td><td>78.8</td><td>72.1</td></tr><tr><td>high_school_geography</td><td>86.4</td><td>76.8</td><td>high_school_gov_and_politics</td><td>91.2</td><td>83.9</td></tr><tr><td>high_school_macroeconomics</td><td>70.5</td><td>65.1</td><td>high_school_mathematics</td><td>31.9</td><td>23.7</td></tr><tr><td>high_school_microeconomics</td><td>77.7</td><td>66.4</td><td>high_school_physics</td><td>36.4</td><td>33.8</td></tr><tr><td>high_school_psychology</td><td>86.6</td><td>81.8</td><td>high_school_statistics</td><td>58.8</td><td>50.0</td></tr><tr><td>high_school_us_history</td><td>83.3</td><td>78.9</td><td>high_school_world_history</td><td>85.2</td><td>75.1</td></tr><tr><td>human_aging</td><td>77.6</td><td>66.4</td><td>human_sexuality</td><td>86.3</td><td>67.2</td></tr><tr><td>international_law</td><td>90.9</td><td>77.7</td><td>jurisprudence</td><td>79.6</td><td>71.3</td></tr><tr><td>logical_fallacies</td><td>80.4</td><td>72.4</td><td>machine_learning</td><td>41.1</td><td>41.1</td></tr><tr><td>management</td><td>82.5</td><td>77.7</td><td>marketing</td><td>89.7</td><td>83.3</td></tr><tr><td>medical_genetics</td><td>69.0</td><td>69.0</td><td>miscellaneous</td><td>84.5</td><td>75.7</td></tr><tr><td>moral_disputes</td><td>77.5</td><td>66.8</td><td>moral_scenarios</td><td>36.5</td><td>40.2</td></tr><tr><td>nutrition</td><td>77.1</td><td>69.9</td><td>philosophy</td><td>79.4</td><td>68.8</td></tr><tr><td>prehistory</td><td>81.2</td><td>67.6</td><td>professional_accounting</td><td>52.1</td><td>44.3</td></tr><tr><td>professional_law</td><td>56.5</td><td>44.5</td><td>professional_medicine</td><td>75.4</td><td>64.0</td></tr><tr><td>professional_psychology</td><td>75.7</td><td>68.1</td><td>public_relations</td><td>73.6</td><td>71.8</td></tr><tr><td>security_studies</td><td>75.9</td><td>64.9</td><td>sociology</td><td>91.0</td><td>84.1</td></tr><tr><td>us_foreign_policy</td><td>92.0</td><td>81.0</td><td>virology</td><td>53.6</td><td>47.0</td></tr><tr><td>world_religions</td><td>87.7</td><td>84.2</td><td></td><td></td><td></td></tr></table>

Table A6 | Chinchilla MMLU results. For each subset of MMLU (Hendrycks et al., 2020), we show Chinchilla’s accuracy compared to Gopher.

<table><tr><td colspan="2">Model Details</td></tr><tr><td>Organization Developing the Model</td><td>DeepMind</td></tr><tr><td>Model Date</td><td>March 2022</td></tr><tr><td>Model Type</td><td>Autoregressive Transformer Language Model (Section 4.1 for details)</td></tr><tr><td>Feedback on the Model</td><td>{jordanhoffmann, sborgeaud, amensch,sifre}@deepmind.com</td></tr><tr><td colspan="2">Intended Uses</td></tr><tr><td>Primary Intended Uses</td><td>The primary use is research on language models, including: research on the scaling behaviour of language models along with those listed in Rae et al. (2021).</td></tr><tr><td>Primary Intended Users</td><td>DeepMind researchers. We will not make this model available publicly.</td></tr><tr><td>Out-of-Scope Uses</td><td>Uses of the language model for language generation in harmful or deceitful settings. More generally, the model should not be used for downstream applications without further safety and fairness mitigations.</td></tr></table>

Factors 

<table><tr><td>Card Prompts – Relevant Factor</td><td>Relevant factors include which language is used. Our model is trained on English data. Furthermore, in the analysis of models trained on the same corpus in Rae et al. (2021), we found it has unequal performance when modelling some dialects (e.g., African American English). Our model is designed for research. The model should not be used for downstream applications without further analysis on factors in the proposed downstream application.</td></tr><tr><td>Card Prompts – Evaluation Factors</td><td>See the results in Rae et al. (2021) which analyzes models trained on the same text corpus.</td></tr></table>

Metrics 

<table><tr><td>Model Performance Measures</td><td>· Perplexity and bits per byte on language modelling datasets· Accuracy on completion tasks, reading comprehension, MMLU, BIG-bench and fact checking.· Exact match accuracy for question answering.· Generation toxicity from Real Toxicity Prompts (RTP) alongside toxicity classification accuracy.· Gender and occupation bias. Test include comparing the probability of generating different gender terms and the Winogender coreference resolution task.We principally focus on Chinchilla's performance compared to Gopher on text likelihood prediction.</td></tr><tr><td>Decision thresholds</td><td>N/A</td></tr><tr><td>Approaches to Uncertainty and Variability</td><td>Due to the costs of training large language models, we did not train Chinchilla multiple times. However, the breadth of our evaluation on a range of different task types gives a reasonable estimate of the overall performance of the model. Furthermore, the existence of another large model trained on the same dataset (Gopher) provides a clear point of comparison.</td></tr><tr><td rowspan="2">Datasets</td><td></td></tr><tr><td>Language modelling on LAMBADA, Wikitext103 (Mer-ity et al., 2017), C4 (Raffel et al., 2020a), PG-19 (Rae et al., 2020) and the Pile (Gao et al., 2020).Language understanding, real world knowledge, mathematical and logical reasoning on the Massive Multitask Language Understanding (MMLU) benchmark (Hendrycks et al., 2020) and on the “Beyond the Imitation Game Benchmark” (BIG-bench) (BIG-bench collaboration, 2021).Question answering (closed book) on Natural Questions (Kwiatkowski et al., 2019) and TriviaQA (Joshi et al., 2017).Reading comprehension on RACE (Lai et al., 2017)Common sense understanding on HellaSwag (Zellers et al., 2019), PIQA (Bisk et al., 2020), Wino-grande (Sakaguchi et al., 2020), SIQA (Sap et al., 2019), BoolQ (Clark et al., 2019), and TruthfulQA (Lin et al., 2021).</td></tr><tr><td>Motivation</td><td>We chose evaluations from Rae et al. (2021) to allow us to most directly compare to Gopher.</td></tr><tr><td>Preprocessing</td><td>Input text is tokenized using a SentencePiece tokenizer with a vocabulary of size 32,000. Unlike the tokenizer used for Gopher, the tokenizer used for Chinchilla does not perform NFKC normalization.</td></tr></table>

Evaluation Data

Training Data 

<table><tr><td colspan="2">The same dataset is used as inRae et al. (2021). Differences in sampling are shown inTable A1.</td></tr><tr><td colspan="2">Quantitative Analyses</td></tr><tr><td>Unitary Results</td><td>Section 4.2gives a detailed description of our analysis. Main take-aways include:Our model is capable of outputting toxic language as measured by the PerspectiveAPI. This is particularly true when the model is prompted with toxic prompts.Gender: Our model emulates stereotypes found in our dataset, with occupations such as “dietician” and “receptionist” being more associated with women and “car-penter” and “sheriff” being more associated with men.Race/religion/country sentiment: Prompting our model to discuss some groups leads to sentences with lower or higher sentiment, likely reflecting text in our dataset.</td></tr></table>

<table><tr><td>Intersectional Results</td><td>We did not investigate intersectional biases.</td></tr><tr><td colspan="2">Ethical Considerations</td></tr><tr><td>Data</td><td>The data is the same as described in Rae et al. (2021).</td></tr><tr><td>Human Life</td><td>The model is not intended to inform decisions about matters central to human life or flourishing.</td></tr><tr><td>Mitigations</td><td>We considered filtering the dataset to remove toxic content but decided against it due to the observation that this can introduce new biases as studied by Welbl et al. (2021). More work is needed on mitigation approaches to toxic content and other types of risks associated with language models, such as those discussed in Weidinger et al. (2021).</td></tr><tr><td>Risks and Harms</td><td>The data is collected from the internet, and thus undoubtedly there is toxic/biased content in our training dataset. Furthermore, it is likely that personal information is also in the dataset that has been used to train our models. We defer to the more detailed discussion in Weidinger et al. (2021).</td></tr><tr><td>Use Cases</td><td>Especially fraught use cases include the generation of factually incorrect information with the intent of distributing it or using the model to generate racist, sexist or otherwise toxic text with harmful intent. Many more use cases that could cause harm exist. Such applications to malicious use are discussed in detail in Weidinger et al. (2021).</td></tr></table>

Table A8 | Chinchilla model card. We follow the framework presented in Mitchell et al. (2019).

# J. List of trained models

In Table A9 we list the model size and configuration of all models used in this study. Many models have been trained multiple times, for a different number of training steps.

<table><tr><td>Task</td><td>Chinchilla</td><td>Gopher</td><td>Task</td><td>Chinchilla</td><td>Gopher</td></tr><tr><td>hyperbaton</td><td>54.2</td><td>51.7</td><td>movie_dialog_same_or_diff</td><td>54.5</td><td>50.7</td></tr><tr><td>causal_judgment</td><td>57.4</td><td>50.8</td><td>winowhy</td><td>62.5</td><td>56.7</td></tr><tr><td>formal_fallacies_syllogisms_neg</td><td>52.1</td><td>50.7</td><td>movie_recommendation</td><td>75.6</td><td>50.5</td></tr><tr><td>crash_blossom</td><td>47.6</td><td>63.6</td><td>moral_permissibility</td><td>57.3</td><td>55.1</td></tr><tr><td>discourse_marker_prediction</td><td>13.1</td><td>11.7</td><td>strategyqa</td><td>68.3</td><td>61.0</td></tr><tr><td>general_knowledge_json</td><td>94.3</td><td>93.9</td><td>nonsense_words_grammar</td><td>78.0</td><td>61.4</td></tr><tr><td>sports_understanding</td><td>71.0</td><td>54.9</td><td>metaphor_boolean</td><td>93.1</td><td>59.3</td></tr><tr><td>implicit_relations</td><td>49.4</td><td>36.4</td><td>navigate</td><td>52.6</td><td>51.1</td></tr><tr><td>penguins_in_a_table</td><td>48.7</td><td>40.6</td><td>presuppositions_as_nli</td><td>49.9</td><td>34.0</td></tr><tr><td>intent_recognition</td><td>92.8</td><td>88.7</td><td>temporal_sequences</td><td>32.0</td><td>19.0</td></tr><tr><td>reasoning_about_colored_objects</td><td>59.7</td><td>49.2</td><td>question_selection</td><td>52.6</td><td>41.4</td></tr><tr><td>logic_grid_puzzle</td><td>44.0</td><td>35.1</td><td>logical_fallacy_detection</td><td>72.1</td><td>58.9</td></tr><tr><td>timedial</td><td>68.8</td><td>50.9</td><td>physical_intuition</td><td>79.0</td><td>59.7</td></tr><tr><td>epistemic_reasoning</td><td>60.6</td><td>56.4</td><td>physics_mc</td><td>65.5</td><td>50.9</td></tr><tr><td>ruin_names</td><td>47.1</td><td>38.6</td><td>identify_odd_metaphor</td><td>68.8</td><td>38.6</td></tr><tr><td>hindu_knowledge</td><td>91.4</td><td>80.0</td><td>understanding_fables</td><td>60.3</td><td>39.6</td></tr><tr><td>misconceptions</td><td>65.3</td><td>61.7</td><td>logical_sequence</td><td>64.1</td><td>36.4</td></tr><tr><td>implicatures</td><td>75.0</td><td>62.0</td><td>mathematical_induction</td><td>47.3</td><td>57.6</td></tr><tr><td>disambiguation_q</td><td>54.7</td><td>45.5</td><td>fantasy_reasoning</td><td>69.0</td><td>64.1</td></tr><tr><td>known_unknowns</td><td>65.2</td><td>63.6</td><td>SNARKS</td><td>58.6</td><td>48.3</td></tr><tr><td>dark_humor_detection</td><td>66.2</td><td>83.1</td><td>crass_ai</td><td>75.0</td><td>56.8</td></tr><tr><td>analogical_similarity</td><td>38.1</td><td>17.2</td><td>entailed_polarity</td><td>94.0</td><td>89.5</td></tr><tr><td>sentence_ambiguity</td><td>71.7</td><td>69.1</td><td>irony_identification</td><td>73.0</td><td>69.7</td></tr><tr><td>riddle_sense</td><td>85.7</td><td>68.2</td><td>evaluating_info_essentiality</td><td>17.6</td><td>16.7</td></tr><tr><td>date_understanding</td><td>52.3</td><td>44.1</td><td>phrase_relatedness</td><td>94.0</td><td>81.8</td></tr><tr><td>analytic_entailment</td><td>67.1</td><td>53.0</td><td>novel_concepts</td><td>65.6</td><td>59.1</td></tr><tr><td>odd_one_out</td><td>70.9</td><td>32.5</td><td>empirical_judgments</td><td>67.7</td><td>52.5</td></tr><tr><td>logical_args</td><td>56.2</td><td>59.1</td><td>figure_of_speech_detection</td><td>63.3</td><td>52.7</td></tr><tr><td>alignment_questionnaire</td><td>91.3</td><td>79.2</td><td>english_proverbs</td><td>82.4</td><td>57.6</td></tr><tr><td>similarities_abstraction</td><td>87.0</td><td>81.8</td><td>Human_organs_senses_mcc</td><td>85.7</td><td>84.8</td></tr><tr><td>anachronisms</td><td>69.1</td><td>56.4</td><td>gre_reading_comprehension</td><td>53.1</td><td>27.3</td></tr></table>

Table A7 | Chinchilla BIG-bench results. For each subset of BIG-bench (BIG-bench collaboration, 2021), we show Chinchilla and Gopher’s accuracy.

<table><tr><td>Parameters (million)</td><td>d_model</td><td>ffw_size</td><td>kv_size</td><td>n_heads</td><td>n_layers</td></tr><tr><td>44</td><td>512</td><td>2048</td><td>64</td><td>8</td><td>8</td></tr><tr><td>57</td><td>576</td><td>2304</td><td>64</td><td>9</td><td>9</td></tr><tr><td>74</td><td>640</td><td>2560</td><td>64</td><td>10</td><td>10</td></tr><tr><td>90</td><td>640</td><td>2560</td><td>64</td><td>10</td><td>13</td></tr><tr><td>106</td><td>640</td><td>2560</td><td>64</td><td>10</td><td>16</td></tr><tr><td>117</td><td>768</td><td>3072</td><td>64</td><td>12</td><td>12</td></tr><tr><td>140</td><td>768</td><td>3072</td><td>64</td><td>12</td><td>15</td></tr><tr><td>163</td><td>768</td><td>3072</td><td>64</td><td>12</td><td>18</td></tr><tr><td>175</td><td>896</td><td>3584</td><td>64</td><td>14</td><td>14</td></tr><tr><td>196</td><td>896</td><td>3584</td><td>64</td><td>14</td><td>16</td></tr><tr><td>217</td><td>896</td><td>3584</td><td>64</td><td>14</td><td>18</td></tr><tr><td>251</td><td>1024</td><td>4096</td><td>64</td><td>16</td><td>16</td></tr><tr><td>278</td><td>1024</td><td>4096</td><td>64</td><td>16</td><td>18</td></tr><tr><td>306</td><td>1024</td><td>4096</td><td>64</td><td>16</td><td>20</td></tr><tr><td>425</td><td>1280</td><td>5120</td><td>128</td><td>10</td><td>18</td></tr><tr><td>489</td><td>1280</td><td>5120</td><td>128</td><td>10</td><td>21</td></tr><tr><td>509</td><td>1408</td><td>5632</td><td>128</td><td>11</td><td>18</td></tr><tr><td>552</td><td>1280</td><td>5120</td><td>128</td><td>10</td><td>24</td></tr><tr><td>587</td><td>1408</td><td>5632</td><td>128</td><td>11</td><td>21</td></tr><tr><td>632</td><td>1536</td><td>6144</td><td>128</td><td>12</td><td>19</td></tr><tr><td>664</td><td>1408</td><td>5632</td><td>128</td><td>11</td><td>24</td></tr><tr><td>724</td><td>1536</td><td>6144</td><td>128</td><td>12</td><td>22</td></tr><tr><td>816</td><td>1536</td><td>6144</td><td>128</td><td>12</td><td>25</td></tr><tr><td>893</td><td>1792</td><td>7168</td><td>128</td><td>14</td><td>20</td></tr><tr><td>1,018</td><td>1792</td><td>7168</td><td>128</td><td>14</td><td>23</td></tr><tr><td>1,143</td><td>1792</td><td>7168</td><td>128</td><td>14</td><td>26</td></tr><tr><td>1,266</td><td>2048</td><td>8192</td><td>128</td><td>16</td><td>22</td></tr><tr><td>1,424</td><td>2176</td><td>8704</td><td>128</td><td>17</td><td>22</td></tr><tr><td>1,429</td><td>2048</td><td>8192</td><td>128</td><td>16</td><td>25</td></tr><tr><td>1,593</td><td>2048</td><td>8192</td><td>128</td><td>16</td><td>28</td></tr><tr><td>1,609</td><td>2176</td><td>8704</td><td>128</td><td>17</td><td>25</td></tr><tr><td>1,731</td><td>2304</td><td>9216</td><td>128</td><td>18</td><td>24</td></tr><tr><td>1,794</td><td>2176</td><td>8704</td><td>128</td><td>17</td><td>28</td></tr><tr><td>2,007</td><td>2304</td><td>9216</td><td>128</td><td>18</td><td>28</td></tr><tr><td>2,283</td><td>2304</td><td>9216</td><td>128</td><td>18</td><td>32</td></tr><tr><td>2,298</td><td>2560</td><td>10240</td><td>128</td><td>20</td><td>26</td></tr><tr><td>2,639</td><td>2560</td><td>10240</td><td>128</td><td>20</td><td>30</td></tr><tr><td>2,980</td><td>2560</td><td>10240</td><td>128</td><td>20</td><td>34</td></tr><tr><td>3,530</td><td>2688</td><td>10752</td><td>128</td><td>22</td><td>36</td></tr><tr><td>3,802</td><td>2816</td><td>11264</td><td>128</td><td>22</td><td>36</td></tr><tr><td>4,084</td><td>2944</td><td>11776</td><td>128</td><td>22</td><td>36</td></tr><tr><td>4,516</td><td>3072</td><td>12288</td><td>128</td><td>24</td><td>36</td></tr><tr><td>6,796</td><td>3584</td><td>14336</td><td>128</td><td>28</td><td>40</td></tr><tr><td>9,293</td><td>4096</td><td>16384</td><td>128</td><td>32</td><td>42</td></tr><tr><td>11,452</td><td>4352</td><td>17408</td><td>128</td><td>32</td><td>47</td></tr><tr><td>12,295</td><td>4608</td><td>18432</td><td>128</td><td>36</td><td>44</td></tr><tr><td>12,569</td><td>4608</td><td>18432</td><td>128</td><td>32</td><td>47</td></tr><tr><td>13,735</td><td>4864</td><td>19456</td><td>128</td><td>32</td><td>47</td></tr><tr><td>14,940</td><td>4992</td><td>19968</td><td>128</td><td>32</td><td>49</td></tr><tr><td>16,183</td><td>5120</td><td>20480</td><td>128</td><td>40</td><td>47</td></tr></table>

Table A9 | All models. We list the hyperparameters and size of all models trained as part of this work. Many shown models have been trained with multiple learning rate schedules/number of training tokens.