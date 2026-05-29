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

### Step 4: 验证图片与文本（可选但推荐）

MinerU 转换可能引入 typo（尤其是图片文件名中的 hash）。如果发现可疑错误：

1. 从 arXiv 下载论文 LaTeX 源码：
   ```bash
   # arXiv 论文页面 → "Download Source" → .tar.gz 或 .7z
   # 解压到临时目录，不要提交到仓库
   mkdir -p /tmp/paper-tex-source && tar xzf source.tar.gz -C /tmp/paper-tex-source/
   ```

2. 对比 tex 源码中的图片文件名与 MinerU 生成的文件名：
   ```bash
   # 在 tex 文件中搜索图片引用
   grep -r 'includegraphics' /tmp/paper-tex-source/
   # 对比 images/ 目录中的实际文件名
   ls papers/XX-paper-name/images/
   ```

3. 修复发现的 typo（通常是 hash 中多/少字符）

4. 验证完成后删除临时目录：
   ```bash
   rm -rf /tmp/paper-tex-source/
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

## 质量检查清单

- [ ] `paper.pdf` 完整
- [ ] `raw-extract.md` 提取完整，图片引用正常
- [ ] `README.md` 公式推导不跳步、有代码验证、有 llm-math 关联
- [ ] `merged-tutorial.md` 原文未修改、讲解穿插在对应章节后
- [ ] 图片路径无 typo（如有怀疑，下载 tex 源码验证）
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
