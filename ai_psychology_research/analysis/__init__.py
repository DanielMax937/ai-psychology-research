"""可视化模块 - 生成实验结果图表"""

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np


def plot_conformity_rate(df: pd.DataFrame, output_path: Optional[str] = None):
    """绘制从众率图"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. 按多数规模
    if "cond_majority_size" in df.columns and "followed_majority" in df.columns:
        valid = df[df["cond_majority_direction"] != "none"].copy()
        ax = axes[0]
        sns.barplot(data=valid, x="cond_majority_size", y="followed_majority", ax=ax)
        ax.set_title("从众率 × 多数规模")
        ax.set_xlabel("多数规模")
        ax.set_ylabel("从众率")
        ax.set_ylim(0, 1)

    # 2. 按来源
    if "cond_source" in df.columns and "followed_majority" in df.columns:
        valid = df[df["cond_majority_direction"] != "none"].copy()
        ax = axes[1]
        sns.barplot(data=valid, x="cond_source", y="followed_majority", ax=ax)
        ax.set_title("从众率 × 信息来源")
        ax.set_xlabel("来源")
        ax.set_ylabel("从众率")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)

    # 3. 按模型
    if "followed_majority" in df.columns:
        valid = df[df["cond_majority_direction"] != "none"].copy()
        ax = axes[2]
        sns.barplot(data=valid, x="model", y="followed_majority", ax=ax)
        ax.set_title("从众率 × 模型")
        ax.set_xlabel("模型")
        ax.set_ylabel("从众率")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✓ 图表已保存: {output_path}")
    plt.close()


def plot_face_directness(df: pd.DataFrame, output_path: Optional[str] = None):
    """绘制面子实验的直接性评分图"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. 按关系亲疏
    if "cond_relationship" in df.columns and "directness_score" in df.columns:
        ax = axes[0]
        order = ["stranger", "colleague", "friend", "superior", "elder"]
        valid_order = [o for o in order if o in df["cond_relationship"].values]
        sns.barplot(data=df, x="cond_relationship", y="directness_score",
                    order=valid_order, ax=ax)
        ax.set_title("直接性 × 关系类型")
        ax.set_xlabel("关系")
        ax.set_ylabel("直接性评分 (4=最直接)")
        ax.tick_params(axis="x", rotation=45)

    # 2. 按公开程度
    if "cond_publicity" in df.columns and "directness_score" in df.columns:
        ax = axes[1]
        order = ["private", "small_group", "public"]
        valid_order = [o for o in order if o in df["cond_publicity"].values]
        sns.barplot(data=df, x="cond_publicity", y="directness_score",
                    order=valid_order, ax=ax)
        ax.set_title("直接性 × 公开程度")
        ax.set_xlabel("公开程度")
        ax.set_ylabel("直接性评分")

    # 3. 交互效应热力图
    if all(c in df.columns for c in ["cond_relationship", "cond_publicity", "directness_score"]):
        ax = axes[2]
        pivot = df.pivot_table(
            values="directness_score",
            index="cond_relationship",
            columns="cond_publicity",
            aggfunc="mean"
        )
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", ax=ax)
        ax.set_title("关系 × 公开程度 交互效应")

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✓ 图表已保存: {output_path}")
    plt.close()


def plot_authority_effect(df: pd.DataFrame, output_path: Optional[str] = None):
    """绘制权威服从效应图"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. 同意率 × 权威来源
    if "cond_authority_source" in df.columns and "choice" in df.columns:
        ax = axes[0]
        df_copy = df.copy()
        df_copy["agreed"] = (df_copy["choice"] == "A").astype(int)
        order = ["none", "anonymous", "student", "professor", "official", "nobel"]
        valid_order = [o for o in order if o in df_copy["cond_authority_source"].values]
        sns.barplot(data=df_copy, x="cond_authority_source", y="agreed",
                    order=valid_order, ax=ax)
        ax.set_title("同意率 × 权威来源")
        ax.set_xlabel("来源")
        ax.set_ylabel("同意率")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)

    # 2. 置信度 × 权威来源
    if "cond_authority_source" in df.columns and "confidence" in df.columns:
        ax = axes[1]
        valid = df.dropna(subset=["confidence"])
        order = ["none", "anonymous", "student", "professor", "official", "nobel"]
        valid_order = [o for o in order if o in valid["cond_authority_source"].values]
        sns.barplot(data=valid, x="cond_authority_source", y="confidence",
                    order=valid_order, ax=ax)
        ax.set_title("置信度 × 权威来源")
        ax.set_xlabel("来源")
        ax.set_ylabel("置信度 (1-10)")
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✓ 图表已保存: {output_path}")
    plt.close()


def plot_dissonance_effect(df: pd.DataFrame, output_path: Optional[str] = None):
    """绘制认知失调效应图"""
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.rcParams["font.sans-serif"] = ["SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. 立场改变 × 承诺强度
    if "cond_commitment_level" in df.columns and "stance_change" in df.columns:
        ax = axes[0]
        ct = pd.crosstab(df["cond_commitment_level"], df["stance_change"], normalize="index")
        ct.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("立场改变分布 × 承诺强度")
        ax.set_xlabel("承诺强度")
        ax.set_ylabel("比例")
        ax.legend(title="改变程度")
        ax.tick_params(axis="x", rotation=45)

    # 2. 合理化率 × 承诺强度
    if "cond_commitment_level" in df.columns and "rationalization" in df.columns:
        ax = axes[1]
        order = ["no_commitment", "mild_commitment", "strong_commitment", "public_commitment"]
        valid_order = [o for o in order if o in df["cond_commitment_level"].values]
        sns.barplot(data=df, x="cond_commitment_level", y="rationalization",
                    order=valid_order, ax=ax)
        ax.set_title("合理化率 × 承诺强度")
        ax.set_xlabel("承诺强度")
        ax.set_ylabel("合理化出现率")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"✓ 图表已保存: {output_path}")
    plt.close()
