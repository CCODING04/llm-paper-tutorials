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
| 3 | `lecture.md` | 独立中文讲解教程 | glm-5.1 + 教学 prompt |
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

# 图片已在 papers/XX-paper-name/images/ 中
```

**验证**：
```bash
ls papers/XX-paper-name/images/ | wc -l   # 检查图片数量
grep -c '!\[' papers/XX-paper-name/raw-extract.md  # 检查图片引用数
```

### Step 4: mimo-v2.5 分析论文图表

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

### Step 5: glm-5.1 生成独立讲解教程（lecture.md）

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

写入 `lecture.md`。

### Step 6: glm-5.1 生成融合版（merged-tutorial.md）

将论文原文（raw-extract.md）和讲解（lecture.md）融合：

- **以论文原文为骨架**，按原文章节顺序
- **每个章节后插入讲解块**：
  ```
  > 📖 **讲解**
  > 
  > [从 lecture.md 提取的对应讲解内容]
  ```
- **论文原文保持不动**（不修改、不翻译）
- **讲解内容**：公式推导、图表解读、代码验证、类比、llm-math-foundations 关联
- **节奏**：读一段原文 → 听一段讲解 → 读下一段

写入 `merged-tutorial.md`。

### Step 7: 更新 README.md 论文列表并推送

```bash
# 更新 README.md 中的论文状态
git add .
git commit -m "feat: add XX-paper-name 论文精读教程"
git push origin main
```

## 质量检查清单

- [ ] `paper.pdf` 完整
- [ ] `raw-extract.md` 提取完整，图片引用正常
- [ ] `lecture.md` 公式推导不跳步、有代码验证、有 llm-math 关联
- [ ] `merged-tutorial.md` 原文未修改、讲解穿插在对应章节后
- [ ] 四份文件都存在于论文目录中
- [ ] README.md 论文状态已更新

## 快速一键命令（添加新论文）

```bash
PAPER_NAME="XX-paper-name"
PDF_URL="https://arxiv.org/pdf/XXXX.XXXXX"

mkdir -p papers/$PAPER_NAME/images
curl -L -o papers/$PAPER_NAME/paper.pdf "$PDF_URL"

export MINERU_TOKEN=$(cat ~/.openclaw/secrets.json | python3 -c "import json,sys; print(json.load(sys.stdin).get('MINERU_API_KEY',''))")
mineru-open-api extract papers/$PAPER_NAME/paper.pdf -o papers/$PAPER_NAME/ -f md --model vlm
mv papers/$PAPER_NAME/paper.md papers/$PAPER_NAME/raw-extract.md

# 然后通过 OpenClaw 执行 Step 4-6
```
