# AI 心理学研究框架

用于研究大语言模型是否表现出类人心理行为模式的实验框架。

## 核心功能

1. **数据集生成** - 根据实验设计自动生成全因素 prompt 数据集
2. **实验执行** - 批量调用多个 LLM API，记录完整原始数据
3. **结果分析** - 自动提取因变量、统计检验、生成报告和可视化

## 支持的实验

| 实验 | 命令名 | 研究问题 |
|------|--------|----------|
| 从众效应 | `conformity` | LLM 是否跟随多数意见？ |
| 权威服从 | `authority` | LLM 是否受权威来源影响？ |
| 认知失调 | `dissonance` | 承诺后面对反证是否合理化？ |
| 面子/文化心理 | `face` | 是否表现出面子维护行为？ |

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置 API 密钥
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"       # 可选
export DASHSCOPE_API_KEY="your-key"       # 可选

# 3. 查看可用实验
python run_experiment.py list

# 4. 一键运行（生成 → 调用 → 分析）
python run_experiment.py full -e face -m gpt-4o-mini -r 5

# 或分步执行：
python run_experiment.py generate -e face -r 10
python run_experiment.py run -e face -m gpt-4o-mini
python run_experiment.py analyze -r output/face_culture_XXXXXX.jsonl
```

## 项目结构

```
ai_psychology_research/
├── run_experiment.py          # 主入口，一键运行
├── config.yaml                # 模型和实验配置
├── requirements.txt
├── framework/
│   ├── models.py              # 模型 API 统一封装 (OpenAI/Anthropic/DashScope)
│   ├── dataset_generator.py   # 全因素数据集生成器
│   ├── experiment_runner.py   # 批量实验执行器
│   └── analyzer.py            # 结果分析 (提取DV + 统计 + 报告)
├── experiments/
│   ├── conformity.py          # 从众效应
│   ├── authority.py           # 权威服从
│   ├── cognitive_dissonance.py # 认知失调
│   └── face_culture.py        # 面子/文化心理
├── analysis/
│   └── __init__.py            # 可视化函数
├── examples/
│   ├── face_study.py          # 面子研究完整示例
│   └── custom_experiment.py   # 自定义实验模板
└── output/                    # 输出目录
    ├── datasets/              # 生成的数据集
    ├── reports/               # 分析报告
    └── figures/               # 可视化图表
```

## 如何自定义新实验

参考 `examples/custom_experiment.py`，三步即可：

1. **定义因素和水平** - 你的自变量是什么
2. **编写情境材料** - 尽量自然，避免暴露实验目的
3. **定义提取规则** - 如何从模型回复中提取因变量

## 配置说明

编辑 `config.yaml` 设置：
- 使用哪些模型
- temperature / top_p 等参数
- 每个条件重复几次
- 输出目录

也支持 `OPENAI_BASE_URL` 环境变量用于代理/兼容接口。

## 分析输出

- **JSONL** - 原始响应数据
- **CSV** - 结构化数据(可直接导入 SPSS/R)
- **TXT** - 文本分析报告
- **PNG** - 可视化图表

## 注意事项

1. Prompt 中不要出现实验名称（如"从众""权威服从"）
2. 每个条件重复多次（建议 ≥10），因为 LLM 输出有随机性
3. 设置合理的 temperature（推荐 0.7）以获得行为变异
4. 同一题目准备反向版本，避免材料偏向
5. 比较多个模型以增强结论的外部效度
