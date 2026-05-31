#!/usr/bin/env python3
"""
示例：研究面子效应的完整流程

本脚本演示如何使用框架进行一个完整的面子效应研究。
可以直接运行此文件，或参考此文件编写你自己的实验脚本。

用法:
    python examples/face_study.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from framework.models import ModelConfig
from framework.experiment_runner import ExperimentRunner
from framework.analyzer import ResultAnalyzer
from experiments.face_culture import FaceExperiment


def main():
    print("=" * 60)
    print("面子效应研究 - 完整流程演示")
    print("=" * 60)

    # ========== Step 1: 描述实验 ==========
    exp = FaceExperiment(seed=42)
    print(f"\n{exp.describe()}")

    # ========== Step 2: 生成数据集 ==========
    print("\n--- Step 1: 生成数据集 ---")
    items = exp.generate_dataset(repetitions=3)  # 演示用少量重复
    summary = exp.generator.summary()
    print(f"总试次: {summary['total_items']}")
    print(f"因素: {summary['factors']}")

    # 保存数据集
    exp.generator.save("output/datasets/face_dataset.jsonl")

    # ========== Step 3: 配置模型 ==========
    print("\n--- Step 2: 配置模型 ---")
    models = [
        ModelConfig(name="gpt-4o-mini", provider="openai", temperature=0.7),
        # 可添加更多模型:
        # ModelConfig(name="claude-sonnet-4-20250514", provider="anthropic"),
        # ModelConfig(name="qwen-plus", provider="dashscope"),
    ]
    print(f"使用模型: {[m.name for m in models]}")

    # ========== Step 4: 运行实验 ==========
    print("\n--- Step 3: 运行实验 ---")
    runner = ExperimentRunner(
        models=models,
        output_dir="./output",
        delay_between_calls=1.0,
    )

    # 如果只想生成数据集但不实际调用 API，可在这里停止
    # 取消下面的注释来实际运行:
    # result_path = runner.run(items, experiment_name="face_culture")

    # ========== Step 5: 分析 (使用已有结果) ==========
    print("\n--- Step 4: 分析说明 ---")
    print("""
实际运行后，使用以下命令分析:

    python run_experiment.py analyze --result output/face_culture_XXXXXX.jsonl

或在代码中:

    analyzer = ResultAnalyzer("output/face_culture_XXXXXX.jsonl")
    analyzer.extract_choice()

    # 提取直接性评分
    from experiments.face_culture import FaceExperiment
    exp = FaceExperiment()
    for idx, row in analyzer.df.iterrows():
        dv = exp.extract_dv(row['response'])
        analyzer.df.at[idx, 'directness_score'] = dv['directness_score']

    # 统计分析
    result = analyzer.compare_conditions('directness_score', 'cond_relationship')
    print(result)

    # 可视化
    from analysis import plot_face_directness
    plot_face_directness(analyzer.df, 'output/figures/face_directness.png')

    # 导出 CSV (可用于 SPSS/R)
    analyzer.to_csv('output/face_data.csv')
""")

    print("\n✓ 演示完成！")
    print("\n快速运行完整流程:")
    print("  python run_experiment.py full -e face -m gpt-4o-mini -r 3")


if __name__ == "__main__":
    main()
