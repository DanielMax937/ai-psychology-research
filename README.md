# AI 心理学研究框架

研究"人类心理学现象是否会在大语言模型输出行为中出现"的实验工具包。不预设 AI 具有主观体验，而是将 LLM 作为可观察行为系统，用心理学实验范式检验其是否表现出类人行为模式。

## 项目结构

```
├── ai_psychology_research/     # 核心代码
│   ├── framework/              # 框架层
│   │   ├── models.py           # 模型调用封装（OpenAI / Anthropic / DashScope）
│   │   ├── experiment_runner.py# 实验执行器
│   │   ├── dataset_generator.py# 数据集生成器
│   │   └── analyzer.py         # 结果分析器
│   ├── experiments/            # 实验实现
│   │   ├── conformity.py       # 从众效应
│   │   ├── authority.py        # 权威服从
│   │   ├── cognitive_dissonance.py # 认知失调
│   │   ├── face_culture.py     # 面子效应
│   │   └── zhongyong.py        # 中庸思维
│   ├── examples/               # 使用示例
│   ├── run_experiment.py       # 单模型实验入口
│   ├── run_multimodel.py       # 多模型对比实验入口
│   └── .env.example            # 环境变量模板
├── notes/                      # 文献笔记与研究规划
├── papers_pdf/                 # 参考论文 PDF
├── web_sources/                # 在线文献存档
└── references.bib              # BibTeX 引用
```

## 快速开始

### 1. 环境准备

```bash
cd ai_psychology_research
pip install openai anthropic pyyaml pandas scipy tqdm numpy
```

### 2. 配置 API 密钥

```bash
cp .env.example .env
# 编辑 .env，填入各模型的 API Key
```

每个模型需要 `*_API_KEY`，`*_BASE_URL` 有默认值可不改。详情见 `.env.example`。

### 3. 运行实验

#### 单模型实验（run_experiment.py）

```bash
# 生成数据集
python run_experiment.py generate -e face -r 10

# 运行指定模型
python run_experiment.py run -e face -m gpt-4o

# 一键完整流程（生成 → 运行 → 分析）
python run_experiment.py full -e conformity -m gpt-4o -r 5
```

支持的实验：`conformity`（从众）、`authority`（权威服从）、`dissonance`（认知失调）、`face`（面子）、`zhongyong`（中庸）

#### 多模型对比实验（run_multimodel.py）

```bash
# 6 模型 × 面子 + 中庸实验，一键批量运行
python run_multimodel.py
```

自动对 MiMo、Claude、GPT、Gemini、DeepSeek、豆包 6 个模型运行面子效应和中庸思维实验，生成对比报告。

## 支持的实验

| 实验 | 文件 | 心理学现象 | 自变量 |
|------|------|-----------|--------|
| 从众效应 | `conformity.py` | Asch 从众范式 | 多数派规模、信息来源、任务难度 |
| 权威服从 | `authority.py` | Milgram 服从范式 | 权威来源、证据质量 |
| 认知失调 | `cognitive_dissonance.py` | Festinger 认知失调 | 承诺水平、角色分配 |
| 面子效应 | `face_culture.py` | 中国文化面子心理 | 关系类型、公开度 |
| 中庸思维 | `zhongyong.py` | 中国文化中庸倾向 | 文化框架、回答格式 |

## 支持的模型

通过 `framework/models.py` 统一封装，支持三种 API 协议：

| Provider | 协议 | 环境变量 |
|----------|------|---------|
| OpenAI 兼容 | Chat Completions | `OPENAI_API_KEY`, `OPENAI_BASE_URL` |
| Anthropic | Messages API | `ANTHROPIC_API_KEY` |
| DashScope | OpenAI 兼容 | `DASHSCOPE_API_KEY` |

多模型实验额外支持：`MIMO_*`、`CLAUDE_*`、`GEMINI_*`、`DEEPSEEK_*`、`DOUBAO_*` 系列环境变量。

## 输出

实验结果保存在 `ai_psychology_research/output/`（已 gitignore）：

- `datasets/` — 生成的实验数据集（JSONL）
- `multimodel/` — 多模型实验原始结果 + 分析 CSV
- `reports/` — 分析报告
- `figures/` — 可视化图表

## 研究笔记

- `notes/01_literature_matrix.md` — 文献梳理
- `notes/02_research_gaps_and_topics.md` — 选题方向
- `notes/03_experiment_templates.md` — 实验设计模板
- `notes/04_conversation_notes.md` — 讨论记录

## 推荐课题方向

> 大语言模型输出中的类社会心理现象：从众、权威服从与认知失调的实验检验

> 中文大语言模型中的文化心理现象：面子、关系取向与中庸思维的行为复现研究
