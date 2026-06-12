# 图表分析 — DFlash

## Figure 1: Speedup Comparison (DFlash vs EAGLE-3 vs Baseline)

### 客观描述
- **类型**：分组柱状图
- **X 轴**：7 个 benchmark（GSM8K, Math500, AIME25, HumanEval, MBPP, LiveCodeBench, MT-Bench）
- **Y 轴**：加速倍数（speedup over baseline）
- **三组柱**：Baseline（=1.00）、EAGLE-3（蓝色，~1.8-2.2×）、DFlash（橙色，~2.75-6.1×）

### 深度分析
- **独立解读**：DFlash 在所有任务上都远超 EAGLE-3，差距在 Math/Code 任务最大（~3×），Chat 任务最小（~1.5×）
- **对照 caption**：一致。Caption 说 "more than 2.5× higher speedup than EAGLE-3"，实际范围 1.45×-3.04×
- **验证的假设**：支持"block diffusion drafting 比自回归 drafting 更快"的核心 claim
- **批判性评价**：baseline=1.00 的对比公平（同一模型无加速）；但 EAGLE-3 用的是默认配置，未完全调优
- **面试价值**：一张图说明 DFlash 的核心优势——在所有场景下全面超越 EAGLE-3

---

## Figure 2: DFlash Inference Design (推理架构图)

### 客观描述
- **类型**：架构流程图
- **组件**：Target Model → Target Embedding → KV Cache → Draft Layer 1/2/... → Target LM Head
- **颜色编码**：蓝色=Fused Target Context Feature，其他=Target Decode Token / Mask Token
- **关键机制**：Context Feature 注入每一层 draft layer 的 KV Cache

### 深度分析
- **独立解读**：展示了 draft 模型如何利用目标模型的内部表征来条件化生成
- **对照 caption**：一致。"Hidden context features extracted from the target model are fused and injected into each draft layer's KV cache"
- **验证的假设**：KV Injection 使 draft 模型能利用目标模型的深层知识
- **批判性评价**：图很清晰，但未展示 KV Injection 的数学细节（公式在 Appendix A.3）
- **面试价值**：这是理解 DFlash 最重要的图——context feature 通过 KV 持续注入，而非一次性融合

---

## Figure 3: Draft Cost Comparison

### 客观描述
- **类型**：分组柱状图
- **X 轴**：Draft token 数量（4, 8, 16）
- **Y 轴**：Draft 延迟（ms）
- **四组柱**：EAGLE-3、DFlash(1L)、DFlash(3L)、DFlash(5L)

### 关键数据
| Draft Tokens | EAGLE-3 | DFlash(1L) | DFlash(3L) | DFlash(5L) |
|-------------|---------|------------|------------|------------|
| 4 | ~7ms | ~2ms | ~4ms | ~5ms |
| 8 | ~12ms | ~2ms | ~4ms | ~6ms |
| 16 | ~25ms | ~2ms | ~4ms | ~6ms |

### 深度分析
- **独立解读**：EAGLE-3 延迟随 token 数线性增长，DFlash 几乎恒定
- **对照 caption**：一致。"Draft cost of 1, 3, 5-layer DFlash and 1-layer EAGLE-3"
- **验证的假设**：支持"block diffusion 的并行生成使 draft 延迟与 token 数无关"的 claim
- **批判性评价**：EAGLE-3 只有 1 层 vs DFlash 最多 5 层，对比不完全公平（但 EAGLE-3 受限于延迟只能用 1 层，这正是论文要证明的）
- **面试价值**：这张图直观展示了串行 vs 并行 drafting 的本质差异——线性增长 vs 恒定

---

## Figure 4: Training Attention Mask (训练注意力掩码)

### 客观描述
- **类型**：注意力掩码热力图（heatmap）
- **X/Y 轴**：token 位置
- **颜色**：蓝色=Target Context Feature，黄色=Clean Token（anchor），绿色=Mask Token，白色=Invisible Token
- **结构**：Prompt tokens → Response tokens（包含多个 masked blocks）

### 深度分析
- **独立解读**：展示了训练时不同类型 token 之间的注意力可见性规则
- **对照 caption**：一致。准确描述了 anchor token、mask token、invisible token 的角色
- **验证的假设**：块内双向注意力 + 块间隔离，匹配推理时的行为
- **批判性评价**：图较复杂，需要结合 caption 才能完全理解。但设计合理——causal consistency + 防 inter-block 信息泄漏
- **面试价值**：理解 DFlash 训练的核心——每个 block 独立 denoise，通过 anchor token 与 clean context 连接

---

## Figure 5: Loss Decay Ablation (Appendix)

### 客观描述
- **类型**：折线图
- **X 轴**：Training epoch
- **Y 轴**：Acceptance length (τ)
- **两条线**：With loss decay（实线）、Without loss decay（虚线）

### 深度分析
- **独立解读**：有 loss decay 的训练收敛更快，最终 acceptance length 更高
- **验证的假设**：位置加权损失能加速收敛并提升接受长度
- **面试价值**：证明"早期 token 更重要"这个直觉在训练中确实有效
