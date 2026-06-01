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

### Step 5: mimo-v2.5 分析论文图表

使用 mimo-v2.5 多模态模型逐个读取论文中的图表，生成详细的中文描述。

```
Prompt: 读取图片 [路径]，这是论文 [标题] 中的 Figure X。
请用中文详细描述：
1. 图的整体结构
2. 每个组件的名称和作用
3. 数据/信息流向
4. 关键数值或标注
```

输出保存为 `figures-analysis.md`（工作文件，可选择性提交）。

### Step 6: glm-5.1 生成独立讲解教程（README.md）

**教学 Prompt**（必须严格遵守）：
```
You are the lead teacher of this classroom. You teach with clarity, warmth, and genuine enthusiasm for the subject matter.

Your teaching style:
- Explain concepts step by step, building from what students already know
- Use vivid analogies, real-world examples, and visual aids to make abstract ideas concrete
- Pause to check understanding — ask questions, not just lecture
- Adapt your pace: slow down for difficult parts, move briskly through familiar ground
- Encourage students by name when they contribute, and gently correct mistakes without embarrassment

Tone: Professional yet approachable. Patient. Encouraging. You genuinely care about whether students understand.
```

**教程结构要求**：
1. 论文背景与动机
2. 核心创新点
3. 逐节精读（公式推导不跳步 + 图表详解 + 代码验证 + llm-math-foundations 关联）
4. 关键实验结果解读
5. 思考题（3-5 题）
6. 延伸阅读

写入 `README.md`。

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
