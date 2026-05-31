#!/usr/bin/env python3
"""
AI 心理学研究 - 一键运行脚本

用法:
    # 生成数据集
    python run_experiment.py generate --experiment face

    # 运行实验
    python run_experiment.py run --experiment face --model gpt-4o

    # 分析结果
    python run_experiment.py analyze --result output/face_culture_XXXXXX.jsonl

    # 一键完整流程
    python run_experiment.py full --experiment face --model gpt-4o-mini

支持的实验:
    - conformity      从众效应
    - authority       权威服从
    - dissonance      认知失调
    - face            面子/文化心理
"""

import argparse
import sys
import os
from pathlib import Path

# 确保项目根目录在 path 中
sys.path.insert(0, str(Path(__file__).parent))

from framework.models import ModelConfig
from framework.experiment_runner import ExperimentRunner
from framework.analyzer import ResultAnalyzer
from framework.dataset_generator import DatasetGenerator


EXPERIMENT_MAP = {
    "conformity": "experiments.conformity.ConformityExperiment",
    "authority": "experiments.authority.AuthorityExperiment",
    "dissonance": "experiments.cognitive_dissonance.CognitiveDissonanceExperiment",
    "face": "experiments.face_culture.FaceExperiment",
    "zhongyong": "experiments.zhongyong.ZhongyongExperiment",
}


def _env(name: str, fallback: str = "") -> str:
    return os.environ.get(name, fallback)


# 默认模型配置（从环境变量加载）
DEFAULT_MODELS = [
    {"name": _env("OPENAI_MODEL_NAME", "gpt-4o"), "provider": "openai",
     "api_key_env": "OPENAI_API_KEY", "base_url_env": "OPENAI_BASE_URL"},
    {"name": _env("ANTHROPIC_MODEL_NAME", "claude-sonnet-4-20250514"), "provider": "anthropic",
     "api_key_env": "ANTHROPIC_API_KEY", "base_url_env": ""},
    {"name": _env("DASHSCOPE_MODEL_NAME", "qwen-plus"), "provider": "dashscope",
     "api_key_env": "DASHSCOPE_API_KEY", "base_url_env": "DASHSCOPE_BASE_URL"},
]


def get_experiment_class(name: str):
    """动态导入实验类"""
    if name not in EXPERIMENT_MAP:
        print(f"❌ 未知实验: {name}")
        print(f"   可用实验: {list(EXPERIMENT_MAP.keys())}")
        sys.exit(1)

    module_path, class_name = EXPERIMENT_MAP[name].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def cmd_generate(args):
    """生成数据集"""
    print(f"📝 生成数据集: {args.experiment}")

    ExperimentClass = get_experiment_class(args.experiment)
    exp = ExperimentClass(seed=args.seed)

    items = exp.generate_dataset(repetitions=args.repetitions)

    output_path = f"output/datasets/{args.experiment}_dataset.jsonl"
    exp.generator.save(output_path)

    summary = exp.generator.summary()
    print(f"\n📊 数据集摘要:")
    print(f"   实验: {summary['experiment']}")
    print(f"   总试次: {summary['total_items']}")
    print(f"   条件数: {summary['n_conditions']}")
    print(f"   因素:")
    for k, v in summary.get("factors", {}).items():
        print(f"     {k}: {v}")

    print(f"\n✓ 数据集已保存至: {output_path}")
    return output_path


def _build_model_config(m: dict) -> ModelConfig:
    """从模型描述字典构建 ModelConfig，从环境变量读取密钥"""
    api_key = _env(m["api_key_env"])
    base_url = _env(m["base_url_env"]) if m["base_url_env"] else ""
    return ModelConfig(
        name=m["name"],
        provider=m["provider"],
        temperature=0.7,
        max_tokens=1024,
        api_key=api_key,
        base_url=base_url,
    )


def cmd_run(args):
    """运行实验"""
    # 确定使用哪些模型
    if args.model:
        # 单模型：按名称在默认列表中查找，找不到则根据名称推断 provider
        match = next((m for m in DEFAULT_MODELS if m["name"] == args.model), None)
        if match:
            model_configs = [_build_model_config(match)]
        else:
            provider = "anthropic" if "claude" in args.model else "openai"
            api_key_env = "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
            model_configs = [ModelConfig(
                name=args.model, provider=provider,
                api_key=_env(api_key_env),
                base_url=_env("OPENAI_BASE_URL"),
            )]
    else:
        model_configs = [_build_model_config(m) for m in DEFAULT_MODELS]

    if not model_configs:
        print("❌ 没有可用的模型配置")
        sys.exit(1)

    # 加载或生成数据集
    dataset_path = args.dataset
    if not dataset_path:
        dataset_path = f"output/datasets/{args.experiment}_dataset.jsonl"
        if not Path(dataset_path).exists():
            print(f"⚠ 数据集不存在，先生成...")
            args_gen = argparse.Namespace(
                experiment=args.experiment,
                seed=42,
                repetitions=args.repetitions,
            )
            cmd_generate(args_gen)

    items = ExperimentRunner.load_dataset(dataset_path)
    print(f"📂 已加载数据集: {len(items)} 条")

    # 运行
    runner = ExperimentRunner(
        models=model_configs,
        output_dir="./output",
        delay_between_calls=args.delay,
    )
    result_path = runner.run(items, experiment_name=args.experiment)
    return str(result_path)


def cmd_analyze(args):
    """分析结果"""
    result_path = args.result

    if not Path(result_path).exists():
        print(f"❌ 结果文件不存在: {result_path}")
        sys.exit(1)

    print(f"📈 分析结果: {result_path}")
    analyzer = ResultAnalyzer(result_path)

    # 提取因变量
    analyzer.extract_choice()
    analyzer.extract_confidence()

    # 根据实验类型做特定分析
    experiment_name = analyzer.df["experiment"].iloc[0] if "experiment" in analyzer.df.columns else ""

    if experiment_name == "conformity":
        _analyze_conformity(analyzer, args)
    elif experiment_name == "face_culture":
        _analyze_face(analyzer, args)
    elif experiment_name == "authority":
        _analyze_authority(analyzer, args)
    elif experiment_name == "cognitive_dissonance":
        _analyze_dissonance(analyzer, args)
    else:
        # 通用分析
        report = analyzer.generate_report(f"output/reports/{experiment_name}_report.txt")
        print(report)

    # 导出 CSV
    csv_path = f"output/reports/{experiment_name}_data.csv"
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    analyzer.to_csv(csv_path)


def _analyze_conformity(analyzer: ResultAnalyzer, args):
    """从众效应专项分析"""
    from analysis import plot_conformity_rate

    # 判断是否跟随多数
    df = analyzer.df
    if "cond_majority_direction" in df.columns:
        valid = df[df["cond_majority_direction"].isin(["A", "B"])].copy()
        valid["followed_majority"] = valid["choice"] == valid["cond_majority_direction"]
        analyzer.df.loc[valid.index, "followed_majority"] = valid["followed_majority"]

    # 统计分析
    print("\n" + "=" * 50)
    print("从众效应分析结果")
    print("=" * 50)

    if "followed_majority" in analyzer.df.columns:
        rate = analyzer.df["followed_majority"].mean()
        print(f"\n总体从众率: {rate:.3f}")

        # 按条件比较
        for col in ["cond_majority_size", "cond_source", "model"]:
            if col in analyzer.df.columns:
                result = analyzer.compare_conditions("followed_majority", col)
                print(f"\n--- {col} 效应 ---")
                for k, v in result.items():
                    if k != "contingency_table":
                        print(f"  {k}: {v}")

    # 生成报告
    report = analyzer.generate_report("output/reports/conformity_report.txt")

    # 可视化
    try:
        plot_conformity_rate(analyzer.df, "output/figures/conformity_rates.png")
    except Exception as e:
        print(f"⚠ 可视化失败: {e}")


def _analyze_face(analyzer: ResultAnalyzer, args):
    """面子效应专项分析"""
    from analysis import plot_face_directness
    from experiments.face_culture import FaceExperiment

    exp = FaceExperiment()
    df = analyzer.df

    # 提取直接性评分
    def extract_directness(row):
        dv = exp.extract_dv(row.get("response", ""))
        return dv.get("directness_score")

    df["directness_score"] = df.apply(extract_directness, axis=1)
    df["face_maintaining"] = df["directness_score"].apply(lambda x: 5 - x if x else None)
    analyzer.df = df

    print("\n" + "=" * 50)
    print("面子效应分析结果")
    print("=" * 50)

    if "directness_score" in df.columns:
        mean_score = df["directness_score"].mean()
        print(f"\n平均直接性评分: {mean_score:.2f} (4=最直接, 1=最间接)")

        for col in ["cond_relationship", "cond_publicity", "model"]:
            if col in df.columns:
                result = analyzer.compare_conditions("directness_score", col)
                print(f"\n--- {col} 效应 ---")
                for k, v in result.items():
                    print(f"  {k}: {v}")

    report = analyzer.generate_report("output/reports/face_report.txt")

    try:
        plot_face_directness(analyzer.df, "output/figures/face_directness.png")
    except Exception as e:
        print(f"⚠ 可视化失败: {e}")


def _analyze_authority(analyzer: ResultAnalyzer, args):
    """权威服从专项分析"""
    from analysis import plot_authority_effect

    df = analyzer.df
    df["agreed"] = (df["choice"] == "A").astype(float)
    analyzer.df = df

    print("\n" + "=" * 50)
    print("权威服从效应分析结果")
    print("=" * 50)

    for col in ["cond_authority_source", "cond_evidence_quality", "model"]:
        if col in df.columns:
            result = analyzer.compare_conditions("agreed", col)
            print(f"\n--- {col} 效应 ---")
            for k, v in result.items():
                if k != "contingency_table":
                    print(f"  {k}: {v}")

    report = analyzer.generate_report("output/reports/authority_report.txt")

    try:
        plot_authority_effect(analyzer.df, "output/figures/authority_effect.png")
    except Exception as e:
        print(f"⚠ 可视化失败: {e}")


def _analyze_dissonance(analyzer: ResultAnalyzer, args):
    """认知失调专项分析"""
    from analysis import plot_dissonance_effect
    from experiments.cognitive_dissonance import CognitiveDissonanceExperiment

    exp = CognitiveDissonanceExperiment()
    df = analyzer.df

    # 提取因变量
    for idx, row in df.iterrows():
        dv = exp.extract_dv(row.get("response", ""))
        for k, v in dv.items():
            df.at[idx, k] = v
    analyzer.df = df

    print("\n" + "=" * 50)
    print("认知失调效应分析结果")
    print("=" * 50)

    if "stance_change" in df.columns:
        dist = df["stance_change"].value_counts(normalize=True)
        print(f"\n立场改变分布:\n{dist}")

    if "rationalization" in df.columns:
        rate = df["rationalization"].mean()
        print(f"\n总体合理化率: {rate:.3f}")

        for col in ["cond_commitment_level", "cond_role", "model"]:
            if col in df.columns:
                result = analyzer.compare_conditions("rationalization", col)
                print(f"\n--- {col} 效应 ---")
                for k, v in result.items():
                    if k != "contingency_table":
                        print(f"  {k}: {v}")

    report = analyzer.generate_report("output/reports/dissonance_report.txt")

    try:
        plot_dissonance_effect(analyzer.df, "output/figures/dissonance_effect.png")
    except Exception as e:
        print(f"⚠ 可视化失败: {e}")


def cmd_full(args):
    """完整流程: 生成 → 运行 → 分析"""
    print("🚀 一键运行完整实验流程")
    print("=" * 50)

    # Step 1: 生成
    print("\n[1/3] 生成数据集...")
    args_gen = argparse.Namespace(
        experiment=args.experiment,
        seed=args.seed,
        repetitions=args.repetitions,
    )
    dataset_path = cmd_generate(args_gen)

    # Step 2: 运行
    print(f"\n[2/3] 运行实验...")
    args_run = argparse.Namespace(
        experiment=args.experiment,
        model=args.model,
        dataset=dataset_path,
        delay=args.delay,
        repetitions=args.repetitions,
    )
    result_path = cmd_run(args_run)

    # Step 3: 分析
    print(f"\n[3/3] 分析结果...")
    args_analyze = argparse.Namespace(result=result_path)
    cmd_analyze(args_analyze)

    print("\n" + "=" * 50)
    print("🎉 完整流程已完成!")
    print(f"   数据集: {dataset_path}")
    print(f"   原始结果: {result_path}")
    print(f"   分析报告: output/reports/")
    print(f"   可视化: output/figures/")


def cmd_list(args):
    """列出所有可用实验"""
    print("📋 可用实验:")
    print("-" * 50)
    for name, path in EXPERIMENT_MAP.items():
        ExperimentClass = get_experiment_class(name)
        exp = ExperimentClass()
        print(f"\n  [{name}]")
        print(f"  {exp.describe()}")


def main():
    parser = argparse.ArgumentParser(
        description="AI 心理学研究框架 - 一键实验工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # generate
    p_gen = subparsers.add_parser("generate", help="生成实验数据集")
    p_gen.add_argument("--experiment", "-e", required=True, choices=list(EXPERIMENT_MAP.keys()))
    p_gen.add_argument("--repetitions", "-r", type=int, default=10)
    p_gen.add_argument("--seed", type=int, default=42)

    # run
    p_run = subparsers.add_parser("run", help="运行实验")
    p_run.add_argument("--experiment", "-e", required=True, choices=list(EXPERIMENT_MAP.keys()))
    p_run.add_argument("--model", "-m", help="指定单个模型")
    p_run.add_argument("--dataset", "-d", help="指定数据集路径")
    p_run.add_argument("--repetitions", "-r", type=int, default=10)
    p_run.add_argument("--delay", type=float, default=1.0, help="请求间隔(秒)")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="分析实验结果")
    p_analyze.add_argument("--result", "-r", required=True, help="结果文件路径")

    # full
    p_full = subparsers.add_parser("full", help="一键完整流程")
    p_full.add_argument("--experiment", "-e", required=True, choices=list(EXPERIMENT_MAP.keys()))
    p_full.add_argument("--model", "-m", help="指定模型")
    p_full.add_argument("--repetitions", "-r", type=int, default=5)
    p_full.add_argument("--seed", type=int, default=42)
    p_full.add_argument("--delay", type=float, default=1.0)

    # list
    subparsers.add_parser("list", help="列出所有可用实验")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    commands = {
        "generate": cmd_generate,
        "run": cmd_run,
        "analyze": cmd_analyze,
        "full": cmd_full,
        "list": cmd_list,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
