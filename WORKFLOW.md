# 🤖 论文教程自动化生成流程

本仓库中的每篇论文教程都遵循以下流程生成。添加新论文时，按步骤执行即可。

## 前置条件

- **MinerU CLI**：`npm install -g mineru-open-api`（v0.5.9+）
- **MINERU_TOKEN**：存储在 `~/.openclaw/secrets.json` 的 `MINERU_API_KEY`
- **OpenClaw**：用于调用 mimo-v2.5（图表分析）和 glm-5.1（教程生成）

## 每篇论文的四份文件

| # | 文件名 | 内容 | 生成方式 |
|---|--------|------|---------|
| 1 | `paper.pdf` | 论文原始 PDF | 下载 |
| 2 | `raw-extract.md` + `images/` | MinerU 转换的 Markdown + 提取的图片 | MinerU VLM |
| 3 | `README.md` | 独立中文讲解教程 | glm-5.1 + 教学 prompt |
| 4 | `merged-tutorial.md` | 论文原文 + 讲解融合版（边看边学） | glm-5.1 融合 |

## 流程步骤

### Step 1: 创建论文目录

```bash
mkdir -p papers/XX-paper-name/images
```

### Step 2: 下载论文 PDF

```bash
# 从 arXiv 下载
curl -L -o papers/XX-paper-name/paper.pdf "https://arxiv.org/pdf/XXXX.XXXXXvN"
```

### Step 3: MinerU 精细化转换

```bash
export MINERU_TOKEN=$(cat ~/.openclaw/secrets.json | python3 -c "import json,sys; print(json.load(sys.stdin).get('MINERU_API_KEY',''))")

# 精细化提取（VLM 模型，保留图片）
mineru-open-api extract papers/XX-paper-name/paper.pdf -o papers/XX-paper-name/ -f md --model vlm

# 重命名提取结果
mv papers/XX-paper-name/paper.md papers/XX-paper-name/raw-extract.md
```

**验证**：
```bash
ls papers/XX-paper-name/images/ | wc -l
grep -c '!\[' papers/XX-paper-name/raw-extract.md
```

### Step 4: LaTeX 源码验证（推荐）

MinerU 转换可能引入 typo（公式、表格数据、专业术语等）。用 arXiv 上的 LaTeX 源码进行交叉验证。

#### 4.1 下载并解压 LaTeX 源码

```bash
# arXiv 论文 ID 格式：XXXX.XXXXX
# 源码下载地址：https://arxiv.org/e-print/XXXX.XXXXX
# 例如 BERT 论文：https://arxiv.org/e-print/1810.04805

PAPER_ID="XXXX.XXXXX"
mkdir -p /tmp/paper-tex-source

# 下载源码（通常是 .tar.gz 或 .gz 格式）
curl -L -o /tmp/paper-tex-source/source.tar.gz "https://arxiv.org/e-print/$PAPER_ID"

# 解压
cd /tmp/paper-tex-source && tar xzf source.tar.gz
```

#### 4.2 对比验证

```bash
# 1. 查看论文结构（各 .tex 文件对应的章节）
grep 'section\|subsection' /tmp/paper-tex-source/*.tex

# 2. 查看图片引用（确保 MinerU 提取的图片名对应正确）
grep -r 'includegraphics' /tmp/paper-tex-source/*.tex

# 3. 查看 caption 文字（对比 MinerU 提取的图/表标题）
grep -E '(\\caption|\\label\{fig)' /tmp/paper-tex-source/*.tex

# 4. 检查表格数据（对比 tex 中的数值与 raw-extract.md 中的数值）
grep -r 'tabular' /tmp/paper-tex-source/*.tex
```

#### 4.3 常见问题与修复

| 问题类型 | 检查方法 | 修复方式 |
|---------|---------|--------|
| 图片文件名 typo | 对比 `includegraphics` 与 `images/` 目录 | 重命名图片文件 |
| 公式错误 | 对比 tex 源码中的数学公式 | 手动修正 raw-extract.md |
| 表格数据偏差 | 对比 tex 中的数值 | 手动修正 raw-extract.md |
| 术语拼写错误 | 对比 tex 中的专业术语 | 手动修正 raw-extract.md |

#### 4.4 清理临时文件

```bash
rm -rf /tmp/paper-tex-source/
```

> **注意**：tex 源码目录不要提交到 git 仓库，只保留在本地临时目录用于验证。

#### 4.5 验证结论记录

验证完成后，在论文目录创建 `tex-verification.md`（可选），记录发现的 typo 和修复情况：
```markdown
# TeX 源码验证记录

## 论文信息
- arXiv ID: XXXX.XXXXX
- 验证日期: YYYY-MM-DD

## 发现的问题
- (无 / 列出发现的 typo)

## 修复内容
- (无 / 列出修复)
```

### Step 5: 图表分析（两阶段）

#### Step 5A：客观提取（mimo-v2.5 多模态）

用 mimo-v2.5 逐个读取论文中的图表，提取基础信息。

```
Prompt: 读取图片 [路径]，这是论文 [标题] 中的 Figure X。
请用中文详细描述：
1. 图的整体结构和类型（折线图/柱状图/架构图/表格等）
2. 每个组件的名称、颜色编码、图例含义
3. 横轴和纵轴的含义及取值范围
4. 关键数值、标注、误差线
5. 数据的趋势和转折点
```

#### Step 5B：深度分析（融入 Step 6 教程生成）

深度分析**不单独做**，而是在 Step 6 生成教程时融入。每张关键图表需要回答以下五个问题：

| 维度 | 问题 |
|------|------|
| **独立解读** | 先不看 caption，这张图在说什么？趋势是什么？关键转折点在哪？ |
| **对照 caption** | 原文 caption 说的和图里显示的一致吗？有没有被选择性展示的数据？ |
| **验证的假设** | 这张图验证了论文的什么假设/claim？如果去掉这个结果，核心论点还站得住吗？ |
| **批判性评价** | 坐标轴刻度是否合理？对比的 baseline 是否公平？有没有遗漏的对比？横轴是否从零开始（是否夸大差异）？ |
| **面试价值** | 如果面试官让你「画一张图说明这篇论文的核心发现」，你能根据这张图复述吗？ |

#### figures-analysis.md 结构模板

```markdown
## Figure X: [标题]

### 客观描述（5A 输出）
[图的类型、结构、组件、数值、趋势]

### 深度分析（5B，教程生成时完成）
- **独立解读**：[不看 caption 的理解]
- **对照 caption**：[原文 caption + 一致性验证]
- **验证的假设**：[这张图支持了什么 claim]
- **批判性评价**：[设计是否合理、baseline 是否公平、刻度是否有误导]
- **面试价值**：[一句话总结这张图的核心洞察]
```

### Step 6: 生成独立讲解教程（README.md）—— 四层递进法

教程必须按以下 **四层递进法** 组织，每层都有明确的「必须做」「不能做」「需要思考的问题」。

---

#### 📖 第一层：鸟瞰（论文在说什么？）

**必须做**：
- ✅ 一句话总结论文的核心贡献
- ✅ 列出 3-5 个关键贡献点
- ✅ 说明这篇论文在知识网络中的位置（之前有什么、之后有什么）
- ✅ 与本系列中其他论文的关系（特别是前后论文的对比）

**不能做**：
- ❌ 不要在鸟瞰阶段就深入技术细节
- ❌ 不要只抄摘要，用自己的话重新组织

**需要思考的问题**：
- 这篇论文解决的核心问题是什么？为什么这个问题重要？
- 这篇论文属于哪个「时代」？它开启了什么新方向？
- 如果只用一句话给面试官介绍这篇论文，你会怎么说？

---

#### 📖 第二层：精读（论文怎么做的？）—— 教程的核心部分

**必须做**：
- ✅ **引言逐段精读**：每一段回答「为什么现有方法不够好 → 本文怎么改进」
- ✅ **方法逐节深入**：每个组件的**直觉解释**（为什么这样设计）+ **公式推导不跳步** + **代码验证**
- ✅ **输入→输出追踪**：数据怎么流入模型、每一步怎么变换、最终输出什么。用文字或代码画出完整数据流
- ✅ **实验精读**：每个实验验证了什么假设？消融实验的每个变体改了什么？结果说明了什么？
- ✅ **图表精读**：先不看描述自己解读图表的趋势和含义，再对照 caption 验证
- ✅ **与已有知识关联**：关联到 llm-math-foundations 的哪些章节？关联到之前读过的哪些论文？

**不能做**：
- ❌ 不能跳过公式推导——每个公式都要解释每个符号的含义和直觉
- ❌ 不能只罗列结果而不解释为什么是这个结果
- ❌ 不能把论文的方法部分当「翻译」——必须是「教学」，有提问、有类比、有引导
- ❌ 不能忽略消融实验——消融实验是理解设计决策的最佳途径

**需要思考的问题**：
- 这个方法的**核心直觉**是什么？能不能用一个生活中的类比来解释？
- 为什么作者选择这个方法而不是其他替代方案？（比如 BERT 为什么用 MLM 而不是双向 LM？）
- 每个设计选择的 trade-off 是什么？（比如 GPT-2 为什么用单向注意力而不是双向？）
- 如果让你用一句话解释这个组件的作用，你会怎么说？
- 公式中每个变量的物理含义是什么？

---

#### 📖 第三层：批判性思考（如果是我，我会怎么做？）

**必须做**：
- ✅ 分析每个**设计决策的理由和替代方案**（为什么用 A 而不是 B？如果用 B 会怎样？）
- ✅ 评估**实验是否充分**：有没有遗漏的对比？结果是否 convincingly 支持了 claim？
- ✅ 分析**局限性**：论文自己承认的 + 你自己发现的
- ✅ 提供**面试视角**：面试官会怎么问这篇论文？标准回答是什么？常见追问怎么应对？

**不能做**：
- ❌ 不能只说「论文很好」——必须指出不足和改进空间
- ❌ 不能把局限性当作「凑字数」——必须认真思考真正的弱点
- ❌ 不能只列面试题不给答案——每道面试题都要有结构化的回答

**需要思考的问题**：
- 这个问题本身重要吗？是真正的问题还是伪需求？
- 如果你是作者，你会做出不同的选择吗？
- 实验中有哪些 confounding factors（混淆因素）被忽略了？
- 这篇论文的结果在什么条件下会失效？
- 后来有没有论文证明了本文的某些假设是错的？（比如 NSP 被 RoBERTa 证明不必要）
- 如果面试官问「你觉得这篇论文最大的问题是什么？」你怎么回答？

---

#### 📖 第四层：知识网络（在更大图景中的位置）

**必须做**：
- ✅ **纵向时间线**：之前有什么 → 本文做了什么 → 之后有什么改进（列出具体的论文名和年份）
- ✅ **横向同期对比**：同一时期其他团队怎么解决类似问题？各自的优劣？
- ✅ **与 llm-math-foundations 关联**：具体到哪一章哪一节的知识点
- ✅ **面试高频问题**：从这篇论文出发，面试常考的知识点 + 回答模板

**不能做**：
- ❌ 不能只列论文名不解释关系——每篇关联论文都要说明「它和本文的关系是什么」
- ❌ 不能忽略反面教材——有些论文是「反面」，说明某条路走不通

**需要思考的问题**：
- 这篇论文开启了什么新方向？后来有哪些重要的 follow-up？
- 如果把这篇论文的方法用到另一个领域，会怎样？
- 这篇论文的哪些思想被后来的工作继承了？哪些被抛弃了？为什么？

---

#### 📖 README.md 最终结构模板

```markdown
# 📖 [论文标题]

> **论文**：[作者], [年份] | [会议/期刊]
> **一句话总结**：[30字以内]

---

## 第一层：鸟瞰

### 🎯 核心贡献
[3-5 个要点]

### 📍 知识网络定位
[之前 → 本文 → 之后]

---

## 第二层：精读

### 1. 引言：为什么需要这篇论文？
[逐段精读，每段回答：现有方法有什么不足？本文怎么改进？]

### 2. 方法：逐节深入

#### 2.1 [组件名称]
- **直觉解释**：[用类比说清楚]
- **公式推导**：[不跳步，每个符号解释]
- **代码验证**：[可运行的代码]
- **为什么这样设计？**：[设计决策的理由]

#### 2.2 [下一个组件]
...

### 3. 数据流：从输入到输出
[完整的输入→变换→输出追踪]

### 4. 实验：每个实验验证了什么？
#### 4.1 主实验
[结果 + 为什么是这个结果]
#### 4.2 消融实验
[每个变体改了什么 + 结论]

### 5. 图表精读
[每张关键图表：先自己解读 + 对照 caption]

---

## 第三层：批判性思考

### 🤔 设计决策分析
[为什么选 A 不选 B？替代方案？]

### ⚠️ 局限性
[论文承认的 + 自己发现的]

### 🎯 面试视角
#### 面试高频问题
[Q&A 格式，每题有结构化回答]

---

## 第四层：知识网络

### 📅 时间线
[之前] → **[本文]** → [之后]

### ↔️ 同期对比
[与其他方法的横向对比]

### 🔗 知识关联
- llm-math-foundations: [具体章节]
- 本系列其他论文: [关联]

---

## ❓ 深度思考题
[5-8 题，包含概念题 + 设计题 + 批判题]

## 📚 延伸阅读
[5+ 篇关联论文，说明每篇和本文的关系]
```

**教学 Prompt**（必须严格遵守）：
```
You are the lead teacher of this classroom. You teach with clarity, warmth, and genuine enthusiasm for the subject matter.

Your teaching style:
- Explain concepts step by step, building from what students already know
- Use vivid analogies, real-world examples, and visual aids to make abstract ideas concrete
- Pause to check understanding — ask questions, not just lecture
- Adapt your pace: slow down for difficult parts, move briskly through familiar ground
- Encourage students by name when they contribute, and gently correct mistakes without embarrassment
- ALWAYS go deeper than the paper itself: explain WHY, not just WHAT
- For every design decision, discuss alternatives and trade-offs

Tone: Professional yet approachable. Patient. Encouraging. You genuinely care about whether students understand.
```

### Step 7: glm-5.1 生成融合版（merged-tutorial.md）

将论文原文（raw-extract.md）和讲解（README.md）融合：

- **以论文原文为骨架**，按原文章节顺序
- **每个章节后插入讲解块**：
  ```
  > 📖 **讲解**
  > 
  > [从 README.md 提取的对应讲解内容]
  ```
- **论文原文保持不动**（不修改、不翻译）
- **节奏**：读一段原文 → 听一段讲解 → 读下一段

写入 `merged-tutorial.md`。

### Step 8: 更新论文列表并推送

```bash
git add .
git commit -m "feat: add XX-paper-name 论文精读教程"
git push origin main
```

## 代码执行与输出要求

教程中的代码块必须包含可执行的测试代码和预期输出：

1. **代码块结构**：函数/类定义 → 测试代码 → 输出结果
2. **输出格式**：在代码块之后添加一个无语言标记的代码块，包含预期输出
3. **生成图片的代码**：保存图片到 `images/` 目录，在教程中插入图片引用 `![描述](./images/xxx.png)`
4. **执行验证**：生成教程后，用 Python 实际执行所有代码块，确认输出与文档一致

示例：
```markdown
```python
def add(a, b):
    return a + b

print(f"1 + 2 = {add(1, 2)}")
```
```
1 + 2 = 3
```

生成图片的代码示例：
```python
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [1, 4, 9])
plt.savefig('images/xxx.png')
```

![xxx 图](./images/xxx.png)
```

## 质量检查清单

- [ ] `paper.pdf` 完整
- [ ] `raw-extract.md` 提取完整，图片引用正常
- [ ] `README.md` 公式推导不跳步、有代码验证、有 llm-math 关联
- [ ] `merged-tutorial.md` 原文未修改、讲解穿插在对应章节后
- [ ] 图片路径无 typo（已用 tex 源码验证）
- [ ] 代码块有测试代码和预期输出
- [ ] 生成图片的代码已执行，图片已保存到 images/
- [ ] 四份文件都存在于论文目录中
- [ ] 主 README.md 论文状态已更新

## 快速一键命令（添加新论文）

```bash
PAPER_NAME="XX-paper-name"
PDF_URL="https://arxiv.org/pdf/XXXX.XXXXX"

mkdir -p papers/$PAPER_NAME/images
curl -L -o papers/$PAPER_NAME/paper.pdf "$PDF_URL"

export MINERU_TOKEN=$(cat ~/.openclaw/secrets.json | python3 -c "import json,sys; print(json.load(sys.stdin).get('MINERU_API_KEY',''))")
mineru-open-api extract papers/$PAPER_NAME/paper.pdf -o papers/$PAPER_NAME/ -f md --model vlm
mv papers/$PAPER_NAME/paper.md papers/$PAPER_NAME/raw-extract.md

# 然后通过 OpenClaw 执行 Step 4-7
```
