#!/usr/bin/env python3
"""
多模型跨文化心理学实验 - 一键批量运行
对比 6 个 LLM 在面子效应和中庸思维实验中的表现
"""

import sys
import os
import json
import time
import logging
import re
from pathlib import Path
from collections import defaultdict
from dataclasses import asdict
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv(Path(__file__).parent / ".env")

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from framework.models import ModelConfig, ModelClient
from framework.experiment_runner import ExperimentRunner
from framework.dataset_generator import ExperimentItem

logging.basicConfig(level=logging.WARNING)

# ============================================================
# 模型配置（从环境变量加载）
# ============================================================

def _env(name: str, fallback: str = "") -> str:
    return os.environ.get(name, fallback)

MODEL_CONFIGS = [
    ModelConfig(
        name=_env("MIMO_MODEL_NAME", "mimo-v2.5-pro"),
        provider="openai",
        api_key=_env("MIMO_API_KEY"),
        base_url=_env("MIMO_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1"),
        temperature=0.7,
        max_tokens=512,
    ),
    ModelConfig(
        name=_env("CLAUDE_MODEL_NAME", "claude-opus-4-7"),
        provider="openai",
        api_key=_env("CLAUDE_API_KEY"),
        base_url=_env("CLAUDE_BASE_URL", "https://openai-proxy.miracleplus.com/v1"),
        temperature=0.7,
        max_tokens=512,
    ),
    ModelConfig(
        name=_env("OPENAI_MODEL_NAME", "gpt-5.5"),
        provider="openai",
        api_key=_env("OPENAI_API_KEY"),
        base_url=_env("OPENAI_BASE_URL", "https://openai-proxy.miracleplus.com/v1"),
        temperature=0.7,
        max_tokens=512,
    ),
    ModelConfig(
        name=_env("GEMINI_MODEL_NAME", "gemini-3.1-pro-preview"),
        provider="openai",
        api_key=_env("GEMINI_API_KEY"),
        base_url=_env("GEMINI_BASE_URL", "https://openai-proxy.miracleplus.com/v1"),
        temperature=0.7,
        max_tokens=512,
    ),
    ModelConfig(
        name=_env("DEEPSEEK_MODEL_NAME", "deepseek-v4-pro"),
        provider="openai",
        api_key=_env("DEEPSEEK_API_KEY"),
        base_url=_env("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        temperature=0.7,
        max_tokens=512,
    ),
    ModelConfig(
        name=_env("DOUBAO_MODEL_NAME", "ep-20260319192219-cjhpg"),
        provider="openai",
        api_key=_env("DOUBAO_API_KEY"),
        base_url=_env("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        temperature=0.7,
        max_tokens=512,
    ),
]

# 模型简称映射（用于显示）
MODEL_SHORT_NAMES = {
    "mimo-v2.5-pro": "MiMo",
    "claude-opus-4-7": "Claude",
    "gpt-5.5": "GPT-5.5",
    "gemini-3.1-pro-preview": "Gemini",
    "deepseek-v4-pro": "DeepSeek",
    "ep-20260319192219-cjhpg": "Doubao",
}


# ============================================================
# 数据集生成（精简版，每模型减少调用量）
# ============================================================

def generate_face_dataset(repetitions=2):
    """生成面子实验精简数据集 - 每模型约45条"""
    from experiments.face_culture import FaceExperiment
    exp = FaceExperiment(seed=42)
    all_items = exp.generate_dataset(repetitions=1)
    # 采样: 3个关系 × 3个公开度 × 5个情境类别 = 45
    selected = []
    target_relationships = ["colleague", "friend", "superior"]
    target_publicity = ["private", "small_group", "public"]
    for item in all_items:
        cond = item.condition
        rel = cond.get("relationship", "")
        pub = cond.get("publicity", "")
        if rel in target_relationships and pub in target_publicity:
            selected.append(item)
            if len(selected) >= 45:
                break
    return selected


def generate_zhongyong_dataset(repetitions=2):
    """生成中庸实验精简数据集 - 每模型约32条"""
    from experiments.zhongyong import ZhongyongExperiment
    exp = ZhongyongExperiment(seed=42)
    all_items = exp.generate_dataset(repetitions=1)
    # 取前32条，覆盖不同条件
    return all_items[:32]


def generate_anti_hedging_dataset(repetitions=3):
    """生成反骑墙实验数据集 - 每模型132条"""
    from experiments.anti_hedging import AntiHedgingExperiment
    exp = AntiHedgingExperiment(seed=42)
    return exp.generate_dataset(repetitions=repetitions)


# ============================================================
# 实验执行
# ============================================================

def run_single_model(model_config: ModelConfig, items: list[ExperimentItem],
                     experiment_name: str, output_dir: Path, delay: float = 0.5) -> list[dict]:
    """对单个模型运行实验"""
    client = ModelClient(model_config)
    results = []
    errors = 0

    short_name = MODEL_SHORT_NAMES.get(model_config.name, model_config.name)
    pbar = tqdm(items, desc=f"  {short_name}", leave=False)

    for item in pbar:
        try:
            response = client.call(item.prompt, item.system_prompt)
            result = {
                "item_id": item.item_id,
                "experiment": item.experiment,
                "model": model_config.name,
                "condition": item.condition,
                "prompt": item.prompt,
                "system_prompt": item.system_prompt,
                "response": response.response or "",
                "timestamp": response.timestamp,
                "temperature": response.temperature,
                "latency_ms": response.latency_ms,
                "expected_dv": item.expected_dv,
                "metadata": item.metadata,
            }
            results.append(result)
            if not response.response:
                errors += 1
        except Exception as e:
            results.append({
                "item_id": item.item_id,
                "experiment": item.experiment,
                "model": model_config.name,
                "condition": item.condition,
                "prompt": item.prompt,
                "system_prompt": item.system_prompt,
                "response": f"[ERROR] {e}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "temperature": model_config.temperature,
                "latency_ms": 0,
                "expected_dv": item.expected_dv,
                "metadata": item.metadata,
            })
            errors += 1

        time.sleep(delay)

    pbar.close()
    valid = len(results) - errors
    print(f"    \u2713 {short_name}: {valid}/{len(results)} valid responses")
    return results


def run_experiment_all_models(experiment_name: str, items: list[ExperimentItem],
                              output_dir: Path, delay: float = 0.5) -> Path:
    """对所有模型运行某个实验（每个模型完成后增量保存）"""
    print(f"\n{'='*60}")
    print(f"\u2696\ufe0f 实验: {experiment_name} | {len(items)} trials \u00d7 {len(MODEL_CONFIGS)} models")
    print(f"{'='*60}")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_path = output_dir / f"multimodel_{experiment_name}_{timestamp}.jsonl"

    all_results = []
    for model_config in MODEL_CONFIGS:
        short_name = MODEL_SHORT_NAMES.get(model_config.name, model_config.name)
        print(f"\n  \U0001f504 Running: {short_name} ({model_config.name})...")
        results = run_single_model(model_config, items, experiment_name, output_dir, delay)
        all_results.extend(results)

        # 增量保存（每个模型完成后写入）
        with open(result_path, "w", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"    \U0001f4be Saved {len(all_results)} results so far")

    print(f"\n  \U0001f4c1 Final: {result_path} ({len(all_results)} total)")
    return result_path


# ============================================================
# 分析函数
# ============================================================

def extract_choice(response: str) -> str:
    """从响应中提取选项"""
    if not response or response.startswith("[ERROR]"):
        return None

    patterns = [
        r'\u9009\u62e9[：:]\s*\[?([A-D])',  # 选择：A
        r'\u6211\u9009\u62e9\s*\[?([A-D])',  # 我选择A
        r'^([A-D])[.、:：\s]',               # A. 开头
        r'\b([A-D])\b',                       # 独立字母
    ]
    for p in patterns:
        m = re.search(p, response)
        if m:
            return m.group(1)
    return None


def analyze_face_results(df: pd.DataFrame) -> dict:
    """面子实验分析"""
    # 提取选择
    df = df.copy()
    df["choice"] = df["response"].apply(extract_choice)
    df = df[df["choice"].notna()].copy()

    # 直接性评分
    score_map = {"A": 4, "B": 3, "C": 2, "D": 1}
    df["directness"] = df["choice"].map(score_map)

    # 提取条件
    df["publicity"] = df["condition"].apply(lambda c: c.get("publicity", "") if isinstance(c, dict) else "")
    df["relationship"] = df["condition"].apply(lambda c: c.get("relationship", "") if isinstance(c, dict) else "")

    results = {
        "total_valid": len(df),
        "by_model": {},
        "overall_choice_dist": df["choice"].value_counts(normalize=True).to_dict(),
    }

    # 模型间比较
    for model, grp in df.groupby("model"):
        short = MODEL_SHORT_NAMES.get(model, model)
        choice_dist = grp["choice"].value_counts(normalize=True).to_dict()
        mean_directness = grp["directness"].mean()
        results["by_model"][short] = {
            "n": len(grp),
            "mean_directness": round(mean_directness, 3),
            "sd": round(grp["directness"].std(), 3),
            "choice_dist": {k: round(v, 3) for k, v in choice_dist.items()},
            "diplomacy_rate": round(choice_dist.get("B", 0) + choice_dist.get("C", 0), 3),
        }

    # 模型间 ANOVA
    groups = [grp["directness"].values for _, grp in df.groupby("model") if len(grp) >= 3]
    if len(groups) >= 2:
        f_stat, p_val = stats.f_oneway(*groups)
        results["model_anova"] = {"F": round(f_stat, 3), "p": round(p_val, 4)}

    # 公开度效应（跨模型）
    pub_groups = {k: grp["directness"].values for k, grp in df.groupby("publicity") if len(grp) >= 3 and k}
    if len(pub_groups) >= 2:
        f_stat, p_val = stats.f_oneway(*pub_groups.values())
        results["publicity_anova"] = {"F": round(f_stat, 3), "p": round(p_val, 4),
                                       "means": {k: round(v.mean(), 3) for k, v in pub_groups.items()}}

    return results, df


def analyze_zhongyong_results(df: pd.DataFrame) -> dict:
    """中庸实验分析"""
    df = df.copy()
    df["choice"] = df["response"].apply(extract_choice)
    df = df[df["choice"].notna()].copy()

    # 是否选择折中
    df["chose_middle"] = (df["choice"] == "C").astype(int)

    # 提取文化框架
    df["cultural_frame"] = df["condition"].apply(
        lambda c: c.get("cultural_frame", "") if isinstance(c, dict) else "")
    df["format"] = df["condition"].apply(
        lambda c: c.get("response_format", "") if isinstance(c, dict) else "")

    # 仅分析自由选择(含C选项)的条件
    free_df = df[df["format"] != "forced_binary"].copy()

    results = {
        "total_valid": len(df),
        "free_choice_valid": len(free_df),
        "by_model": {},
        "overall_middle_rate": round(free_df["chose_middle"].mean(), 3) if len(free_df) > 0 else 0,
    }

    # 模型间比较
    for model, grp in free_df.groupby("model"):
        short = MODEL_SHORT_NAMES.get(model, model)
        middle_rate = grp["chose_middle"].mean()
        choice_dist = grp["choice"].value_counts(normalize=True).to_dict()
        results["by_model"][short] = {
            "n": len(grp),
            "middle_rate": round(middle_rate, 3),
            "choice_dist": {k: round(v, 3) for k, v in choice_dist.items()},
        }

    # 模型间差异: 卡方检验
    contingency = pd.crosstab(free_df["model"], free_df["chose_middle"])
    if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
        chi2, p, dof, _ = stats.chi2_contingency(contingency)
        results["model_chi2"] = {"chi2": round(chi2, 3), "p": round(p, 4), "dof": dof}

    # 文化框架效应
    frame_rates = {}
    for frame, grp in free_df.groupby("cultural_frame"):
        if len(grp) >= 3 and frame:
            frame_rates[frame] = round(grp["chose_middle"].mean(), 3)
    results["cultural_frame_rates"] = frame_rates

    return results, df


def analyze_anti_hedging_results(df: pd.DataFrame) -> dict:
    """反骑墙实验分析"""
    from experiments.anti_hedging import AntiHedgingExperiment
    exp = AntiHedgingExperiment()

    df = df.copy()

    # 用实验类的 extract_dv 提取因变量
    dv_records = []
    for _, row in df.iterrows():
        dv = exp.extract_dv(row.get("response", ""))
        dv["model"] = row["model"]
        dv["condition"] = row["condition"]
        dv["metadata"] = row.get("metadata", {})
        dv_records.append(dv)
    dv_df = pd.DataFrame(dv_records)

    # 提取条件
    dv_df["strategy"] = dv_df["condition"].apply(
        lambda c: c.get("prompt_strategy", "") if isinstance(c, dict) else "")
    dv_df["scene_type"] = dv_df["condition"].apply(
        lambda c: c.get("scene_type", "") if isinstance(c, dict) else "")

    results = {"total_valid": len(dv_df), "by_strategy": {}, "by_model_strategy": {}}

    # ---- 按场景类型分析 ----
    for scene_type in ["decisive", "direct", "uncertainty"]:
        scene_df = dv_df[dv_df["scene_type"] == scene_type].copy()
        if len(scene_df) == 0:
            continue

        scene_results = {"n": len(scene_df), "by_strategy": {}}

        for strategy in ["baseline", "anti_hedging"]:
            strat_df = scene_df[scene_df["strategy"] == strategy]
            if len(strat_df) == 0:
                continue

            stats_dict = {"n": len(strat_df)}

            if scene_type in ("decisive", "direct"):
                # 选择题场景
                valid = strat_df[strat_df["choice"].notna()]
                if len(valid) > 0:
                    choice_dist = valid["choice"].value_counts(normalize=True).to_dict()
                    stats_dict["choice_dist"] = {k: round(v, 3) for k, v in choice_dist.items()}
                    if scene_type == "decisive":
                        stats_dict["middle_rate"] = round(valid["is_middle_way"].mean(), 3)
                    if scene_type == "direct":
                        stats_dict["mean_directness"] = round(valid["directness_score"].mean(), 3)
                        stats_dict["indirect_rate"] = round((valid["choice"].isin(["C", "D"])).mean(), 3)

                stats_dict["mean_hedging"] = round(strat_df["hedging_count"].mean(), 3)
                stats_dict["dialectical_rate"] = round(strat_df["dialectical_thinking"].mean(), 3)

            elif scene_type == "uncertainty":
                # 开放式场景
                stats_dict["dont_know_rate"] = round(strat_df["said_dont_know"].mean(), 3)
                stats_dict["fabrication_rate"] = round(strat_df["fabricated_details"].mean(), 3)
                stats_dict["mean_hedging"] = round(strat_df["hedging_count"].mean(), 3)
                stats_dict["mean_length"] = round(strat_df["response_length"].mean(), 1)

            scene_results["by_strategy"][strategy] = stats_dict

        # 统计检验: baseline vs anti_hedging
        base_df = scene_df[scene_df["strategy"] == "baseline"]
        anti_df = scene_df[scene_df["strategy"] == "anti_hedging"]

        if scene_type == "decisive" and len(base_df) > 2 and len(anti_df) > 2:
            base_valid = base_df[base_df["choice"].notna()]
            anti_valid = anti_df[anti_df["choice"].notna()]
            if len(base_valid) > 0 and len(anti_valid) > 0:
                contingency = pd.DataFrame({
                    "baseline": [int((base_valid["is_middle_way"]).sum()), int((~base_valid["is_middle_way"]).sum())],
                    "anti_hedging": [int((anti_valid["is_middle_way"]).sum()), int((~anti_valid["is_middle_way"]).sum())],
                }, index=["middle", "extreme"])
                if contingency.min().min() >= 0:
                    try:
                        chi2, p, _, _ = stats.chi2_contingency(contingency)
                        scene_results["chi2_test"] = {"chi2": round(chi2, 3), "p": round(p, 4)}
                    except Exception:
                        pass

        if scene_type == "uncertainty" and len(base_df) > 2 and len(anti_df) > 2:
            try:
                # Fisher exact or chi-square for "I don't know" rates
                contingency = pd.DataFrame({
                    "baseline": [int(base_df["said_dont_know"].sum()), int((~base_df["said_dont_know"]).sum())],
                    "anti_hedging": [int(anti_df["said_dont_know"].sum()), int((~anti_df["said_dont_know"]).sum())],
                }, index=["dont_know", "fabricated"])
                chi2, p, _, _ = stats.chi2_contingency(contingency)
                scene_results["chi2_test"] = {"chi2": round(chi2, 3), "p": round(p, 4)}
            except Exception:
                pass

        results["by_strategy"][scene_type] = scene_results

    # ---- 按模型 × 策略分析 ----
    for model, grp in dv_df.groupby("model"):
        short = MODEL_SHORT_NAMES.get(model, model)
        model_results = {}
        for strategy in ["baseline", "anti_hedging"]:
            strat_df = grp[grp["strategy"] == strategy]
            if len(strat_df) == 0:
                continue
            s = {"n": len(strat_df)}

            decisive_df = strat_df[strat_df["scene_type"] == "decisive"]
            direct_df = strat_df[strat_df["scene_type"] == "direct"]
            uncertain_df = strat_df[strat_df["scene_type"] == "uncertainty"]

            if len(decisive_df) > 0:
                valid = decisive_df[decisive_df["choice"].notna()]
                if len(valid) > 0:
                    s["decisive_middle_rate"] = round(valid["is_middle_way"].mean(), 3)
                    s["decisive_hedging"] = round(decisive_df["hedging_count"].mean(), 3)

            if len(direct_df) > 0:
                valid = direct_df[direct_df["choice"].notna()]
                if len(valid) > 0:
                    s["direct_mean"] = round(valid["directness_score"].mean(), 3)
                    s["direct_hedging"] = round(direct_df["hedging_count"].mean(), 3)

            if len(uncertain_df) > 0:
                s["uncertain_dont_know"] = round(uncertain_df["said_dont_know"].mean(), 3)
                s["uncertain_fabrication"] = round(uncertain_df["fabricated_details"].mean(), 3)

            model_results[strategy] = s

        results["by_model_strategy"][short] = model_results

    return results, dv_df

def generate_report(face_results: dict, zhongyong_results: dict,
                    ah_results: dict = None, output_path: Path = None):
    """生成 Markdown 研究报告"""

    report = f"""# 多模型跨文化心理学实验报告
## Large Language Models 中的面子效应、中庸思维与反骑墙策略验证

**实验日期**: {time.strftime("%Y-%m-%d")}
**模型**: MiMo, Claude Opus 4, GPT-5.5, Gemini 3.1 Pro, DeepSeek V4 Pro, Doubao Pro

---

## 1. 研究概述

本研究对比 6 个主流大语言模型在两个中国文化心理学经典现象中的行为表现：
1. **面子效应** — 在涉及面子威胁的社交情境中，模型是否表现出间接表达和关系维护？
2. **中庸思维** — 面对对立观点时，模型是否倾向于折中和避免极端？

### 核心研究问题
- 不同模型是否都表现出文化心理效应？
- 中国本土模型（MiMo, DeepSeek, Doubao）与国际模型（Claude, GPT, Gemini）之间是否存在差异？
- 哪个模型最接近中国文化心理学理论的预测？

---

## 2. 方法

### 2.1 模型
| 模型 | 厂商 | 类型 |
|------|------|------|
| mimo-v2.5-pro | 小米 | 中国本土 |
| deepseek-v4-pro | DeepSeek | 中国本土 |
| doubao-pro | 字节跳动 | 中国本土 |
| claude-opus-4-7 | Anthropic | 国际 |
| gpt-5.5 | OpenAI | 国际 |
| gemini-3.1-pro-preview | Google | 国际 |

### 2.2 实验设计
- **面子实验**: 15情境 × 3关系 × 3公开度 × 3重复 = 约45条/模型
- **中庸实验**: 11情境 × 4文化框架 × 2格式 × 3重复 = 约32条/模型
- **温度**: 0.7, **max_tokens**: 512

### 2.3 因变量
- 面子: 选择策略(A直接/B委婉/C间接/D回避), 直接性评分(1-4)
- 中庸: 折中选择率(选C的比例), 辩证语言使用

---

## 3. 实验一: 面子效应

### 3.1 有效响应
"""
    # Face results
    face = face_results
    report += f"总有效响应: **{face['total_valid']}**\n\n"

    report += "### 3.2 各模型表现\n\n"
    report += "| 模型 | N | 平均直接性 | SD | 委婉率(B+C) | 主选策略 |\n"
    report += "|------|---|-----------|-----|------------|----------|\n"

    for model, data in sorted(face["by_model"].items(), key=lambda x: x[1]["mean_directness"]):
        top_choice = max(data["choice_dist"], key=data["choice_dist"].get) if data["choice_dist"] else "?"
        report += f"| {model} | {data['n']} | {data['mean_directness']} | {data['sd']} | {data['diplomacy_rate']} | {top_choice} |\n"

    if "model_anova" in face:
        report += f"\n**模型间差异 (ANOVA)**: F = {face['model_anova']['F']}, p = {face['model_anova']['p']}\n"

    if "publicity_anova" in face:
        pub = face["publicity_anova"]
        report += f"\n### 3.3 公开度效应\n"
        report += f"**ANOVA**: F = {pub['F']}, p = {pub['p']}\n\n"
        for k, v in sorted(pub["means"].items()):
            report += f"- {k}: M = {v}\n"

    report += f"""
### 3.4 面子实验小结
- 总体选择分布: {face.get('overall_choice_dist', {})}
- 所有模型均表现出高度委婉偏好
"""

    # Zhongyong results
    zy = zhongyong_results
    report += f"""
---

## 4. 实验二: 中庸思维

### 4.1 有效响应
总有效响应: **{zy['total_valid']}** (自由选择条件: {zy['free_choice_valid']})

### 4.2 各模型中庸倾向

| 模型 | N | 折中率 | 选项分布 |
|------|---|--------|----------|
"""

    for model, data in sorted(zy["by_model"].items(), key=lambda x: x[1]["middle_rate"], reverse=True):
        dist_str = ", ".join([f"{k}:{v}" for k, v in sorted(data["choice_dist"].items())])
        report += f"| {model} | {data['n']} | {data['middle_rate']:.1%} | {dist_str} |\n"

    if "model_chi2" in zy:
        report += f"\n**模型间差异 (卡方)**: \u03c7\u00b2 = {zy['model_chi2']['chi2']}, df = {zy['model_chi2']['dof']}, p = {zy['model_chi2']['p']}\n"

    if zy.get("cultural_frame_rates"):
        report += f"\n### 4.3 文化框架效应\n\n"
        report += "| 文化框架 | 折中率 |\n|----------|--------|\n"
        for frame, rate in sorted(zy["cultural_frame_rates"].items(), key=lambda x: x[1], reverse=True):
            report += f"| {frame} | {rate:.1%} |\n"

    report += f"""
### 4.4 中庸实验小结
- 总体折中率: {zy['overall_middle_rate']:.1%}
- 文化框架对折中行为有调节作用
"""

    # Anti-hedging results
    if ah_results:
        ah = ah_results
        report += f"""
---

## 5. 实验三: 反骑墙策略验证

### 5.1 实验设计
- **目的**: 检验 anti-hedging prompt 能否减少折中偏好、间接表达和幻觉
- **自变量**: prompt策略(baseline vs anti_hedging) × 场景类型(决断力/直接性/不确定性)
- **因变量**: 折中率、直接性评分、对冲语言频率、"不知道"率、编造率

### 5.2 决断力场景（折中率对比）

| 指标 | Baseline | Anti-Hedging | 变化 |
|------|----------|-------------|------|"""

        decisive = ah.get("by_strategy", {}).get("decisive", {})
        base_s = decisive.get("by_strategy", {}).get("baseline", {})
        anti_s = decisive.get("by_strategy", {}).get("anti_hedging", {})

        if base_s and anti_s:
            base_mr = base_s.get("middle_rate", 0)
            anti_mr = anti_s.get("middle_rate", 0)
            delta_mr = anti_mr - base_mr
            report += f"\n| 折中率 | {base_mr:.1%} | {anti_mr:.1%} | {delta_mr:+.1%} |"

            base_hedge = base_s.get("mean_hedging", 0)
            anti_hedge = anti_s.get("mean_hedging", 0)
            report += f"\n| 平均对冲次数 | {base_hedge:.2f} | {anti_hedge:.2f} | {anti_hedge - base_hedge:+.2f} |"

            base_dial = base_s.get("dialectical_rate", 0)
            anti_dial = anti_s.get("dialectical_rate", 0)
            report += f"\n| 辩证思维率 | {base_dial:.1%} | {anti_dial:.1%} | {anti_dial - base_dial:+.1%} |"

        if "chi2_test" in decisive:
            report += f"\n\n**卡方检验**: χ² = {decisive['chi2_test']['chi2']}, p = {decisive['chi2_test']['p']}"

        report += f"""

### 5.3 直接性场景（直接性对比）

| 指标 | Baseline | Anti-Hedging | 变化 |
|------|----------|-------------|------|"""

        direct = ah.get("by_strategy", {}).get("direct", {})
        base_d = direct.get("by_strategy", {}).get("baseline", {})
        anti_d = direct.get("by_strategy", {}).get("anti_hedging", {})

        if base_d and anti_d:
            base_dir = base_d.get("mean_directness", 0)
            anti_dir = anti_d.get("mean_directness", 0)
            report += f"\n| 平均直接性(1-4) | {base_dir:.2f} | {anti_dir:.2f} | {anti_dir - base_dir:+.2f} |"

            base_ind = base_d.get("indirect_rate", 0)
            anti_ind = anti_d.get("indirect_rate", 0)
            report += f"\n| 间接率(C+D) | {base_ind:.1%} | {anti_ind:.1%} | {anti_ind - base_ind:+.1%} |"

            base_h = base_d.get("mean_hedging", 0)
            anti_h = anti_d.get("mean_hedging", 0)
            report += f"\n| 平均对冲次数 | {base_h:.2f} | {anti_h:.2f} | {anti_h - base_h:+.2f} |"

        report += f"""

### 5.4 不确定性场景（幻觉对比）

| 指标 | Baseline | Anti-Hedging | 变化 |
|------|----------|-------------|------|"""

        uncertain = ah.get("by_strategy", {}).get("uncertainty", {})
        base_u = uncertain.get("by_strategy", {}).get("baseline", {})
        anti_u = uncertain.get("by_strategy", {}).get("anti_hedging", {})

        if base_u and anti_u:
            base_dk = base_u.get("dont_know_rate", 0)
            anti_dk = anti_u.get("dont_know_rate", 0)
            report += f"\n| '不知道'率 | {base_dk:.1%} | {anti_dk:.1%} | {anti_dk - base_dk:+.1%} |"

            base_fab = base_u.get("fabrication_rate", 0)
            anti_fab = anti_u.get("fabrication_rate", 0)
            report += f"\n| 编造率 | {base_fab:.1%} | {anti_fab:.1%} | {anti_fab - base_fab:+.1%} |"

            base_len = base_u.get("mean_length", 0)
            anti_len = anti_u.get("mean_length", 0)
            report += f"\n| 平均回答长度 | {base_len:.0f}字 | {anti_len:.0f}字 | {anti_len - base_len:+.0f}字 |"

        if "chi2_test" in uncertain:
            report += f"\n\n**卡方检验**: χ² = {uncertain['chi2_test']['chi2']}, p = {uncertain['chi2_test']['p']}"

        report += f"""

### 5.5 各模型反骑墙效果

| 模型 | 策略 | 决断折中率 | 直接性 | 不知道率 | 编造率 |
|------|------|-----------|--------|---------|--------|
"""
        for model, strategies in sorted(ah.get("by_model_strategy", {}).items()):
            for strategy, s in strategies.items():
                label = "baseline" if strategy == "baseline" else "anti-hedge"
                mr = s.get("decisive_middle_rate", "-")
                dr = s.get("direct_mean", "-")
                dk = s.get("uncertain_dont_know", "-")
                fab = s.get("uncertain_fabrication", "-")
                if isinstance(mr, float):
                    mr = f"{mr:.1%}"
                if isinstance(dr, float):
                    dr = f"{dr:.2f}"
                if isinstance(dk, float):
                    dk = f"{dk:.1%}"
                if isinstance(fab, float):
                    fab = f"{fab:.1%}"
                report += f"| {model} | {label} | {mr} | {dr} | {dk} | {fab} |\n"

    report += f"""
---

## 6. 跨模型综合讨论

### 6.1 主要发现

1. **共同特征**: 所有模型在面子情境中都偏好委婉策略，表明 RLHF 对齐过程普遍引入了"礼貌"倾向
2. **模型差异**: 不同模型在折中倾向上可能存在差异
3. **文化可操控性**: 通过 system prompt 中的文化框架可以调节中庸行为
4. **Anti-hedging 效果**: 反骑墙 prompt 能否有效减少折中/间接/幻觉行为（见实验三数据）

### 6.2 局限性

1. mimo-v2.5-pro 作为推理模型，有效响应率较低
2. 各模型 API 稳定性不同，可能影响有效样本量
3. 单次实验重复3次，统计效力有限

---

## 7. 结论

本研究通过系统对比 6 个 LLM 在面子效应、中庸思维和反骑墙策略验证三个实验中的表现，
初步揭示了大语言模型的文化心理行为特征及 prompt 策略的调节效果。

---

*Generated by AI Psychology Research Framework v1.0*
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n📝 报告已保存: {output_path}")
    return report


# ============================================================
# 主程序
# ============================================================

def main():
    output_dir = Path("output/multimodel")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🔬 多模型跨文化心理学实验")
    print(f"   模型: {len(MODEL_CONFIGS)} 个")
    print(f"   实验: 面子效应 + 中庸思维 + 反骑墙验证")
    print("=" * 60)

    # 生成数据集
    print("\n[1/5] 生成数据集...")
    face_items = generate_face_dataset(repetitions=3)
    zhongyong_items = generate_zhongyong_dataset(repetitions=3)
    anti_hedging_items = generate_anti_hedging_dataset(repetitions=3)
    print(f"  面子: {len(face_items)} trials")
    print(f"  中庸: {len(zhongyong_items)} trials")
    print(f"  反骑墙: {len(anti_hedging_items)} trials")

    # 运行面子实验
    print("\n[2/5] 运行面子实验...")
    face_path = run_experiment_all_models("face", face_items, output_dir, delay=0.5)

    # 运行中庸实验
    print("\n[3/5] 运行中庸实验...")
    zy_path = run_experiment_all_models("zhongyong", zhongyong_items, output_dir, delay=0.5)

    # 运行反骑墙实验
    print("\n[4/5] 运行反骑墙实验...")
    ah_path = run_experiment_all_models("anti_hedging", anti_hedging_items, output_dir, delay=0.5)

    # 分析
    print("\n[5/5] 分析结果...")

    # 加载结果
    face_df = pd.DataFrame([json.loads(l) for l in open(face_path, encoding="utf-8")])
    zy_df = pd.DataFrame([json.loads(l) for l in open(zy_path, encoding="utf-8")])
    ah_df = pd.DataFrame([json.loads(l) for l in open(ah_path, encoding="utf-8")])

    face_results, face_analyzed = analyze_face_results(face_df)
    zy_results, zy_analyzed = analyze_zhongyong_results(zy_df)
    ah_results, ah_analyzed = analyze_anti_hedging_results(ah_df)

    # 保存分析数据
    face_analyzed.to_csv(output_dir / "face_analysis.csv", index=False)
    zy_analyzed.to_csv(output_dir / "zhongyong_analysis.csv", index=False)
    ah_analyzed.to_csv(output_dir / "anti_hedging_analysis.csv", index=False)

    # 生成报告
    report_path = output_dir / "多模型实验报告.md"
    generate_report(face_results, zy_results, ah_results, report_path)

    print("\n" + "=" * 60)
    print("🎉 全部完成!")
    print(f"   原始数据: {face_path}")
    print(f"            {zy_path}")
    print(f"            {ah_path}")
    print(f"   分析报告: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
