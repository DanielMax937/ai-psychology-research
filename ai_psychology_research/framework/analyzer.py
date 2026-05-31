"""结果分析器 - 对实验结果进行统计分析和可视化"""

import json
import re
from pathlib import Path
from typing import Optional, Callable

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency


class ResultAnalyzer:
    """实验结果分析器"""

    def __init__(self, result_path: str):
        self.result_path = Path(result_path)
        self.df = self._load_results()

    def _load_results(self) -> pd.DataFrame:
        """加载 JSONL 结果为 DataFrame"""
        records = []
        with open(self.result_path, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

        df = pd.DataFrame(records)

        # 展开 condition 字典为独立列
        if "condition" in df.columns:
            condition_df = pd.json_normalize(df["condition"])
            condition_df.columns = [f"cond_{c}" for c in condition_df.columns]
            df = pd.concat([df, condition_df], axis=1)

        return df

    def extract_choice(
        self,
        pattern: str = r"[选择我的选择是|我选择|选择：?]\s*([A-D])",
        column: str = "response",
        new_column: str = "choice",
    ) -> pd.DataFrame:
        """从响应文本中提取选择"""
        def _extract(text):
            if not isinstance(text, str):
                return None
            # 尝试多种模式
            patterns = [
                pattern,
                r"^([A-D])[.\s、：:]",
                r"选择\s*([A-D])",
                r"答案[是为：:]\s*([A-D])",
                r"我[的]?选择[是为：:]\s*([A-D])",
            ]
            for p in patterns:
                match = re.search(p, text, re.MULTILINE)
                if match:
                    return match.group(1)
            # 如果开头就是单个字母
            if text.strip() and text.strip()[0] in "ABCD":
                return text.strip()[0]
            return None

        self.df[new_column] = self.df[column].apply(_extract)
        extracted = self.df[new_column].notna().sum()
        total = len(self.df)
        print(f"✓ 选择提取: {extracted}/{total} ({extracted/total*100:.1f}%)")
        return self.df

    def extract_confidence(
        self,
        column: str = "response",
        new_column: str = "confidence",
    ) -> pd.DataFrame:
        """从响应中提取置信度评分"""
        def _extract_conf(text):
            if not isinstance(text, str):
                return None
            patterns = [
                r"置信度[：:]\s*(\d+)",
                r"信心[：:]\s*(\d+)",
                r"confidence[：:]\s*(\d+)",
                r"(\d+)[/／]10",
                r"(\d+)\s*分",
            ]
            for p in patterns:
                match = re.search(p, text, re.IGNORECASE)
                if match:
                    val = int(match.group(1))
                    if 0 <= val <= 100:
                        return val
                    elif 0 <= val <= 10:
                        return val * 10
            return None

        self.df[new_column] = self.df[column].apply(_extract_conf)
        return self.df

    def custom_extract(
        self,
        extractor: Callable[[str], any],
        column: str = "response",
        new_column: str = "extracted",
    ) -> pd.DataFrame:
        """自定义提取函数"""
        self.df[new_column] = self.df[column].apply(extractor)
        return self.df

    def conformity_rate(self, choice_col: str = "choice", majority_col: str = "cond_majority_direction") -> dict:
        """计算从众率"""
        valid = self.df.dropna(subset=[choice_col])
        if majority_col not in valid.columns:
            print(f"⚠ 未找到列 {majority_col}")
            return {}

        # 判断是否跟随多数
        valid = valid.copy()
        valid["followed_majority"] = valid[choice_col] == valid[majority_col]

        # 按条件分组统计
        results = {}
        condition_cols = [c for c in valid.columns if c.startswith("cond_")]
        for col in condition_cols:
            group_stats = valid.groupby(col)["followed_majority"].agg(["mean", "count", "std"])
            results[col] = group_stats.to_dict("index")

        # 总体从众率
        results["overall_conformity_rate"] = valid["followed_majority"].mean()
        return results

    def chi_square_test(self, row_var: str, col_var: str) -> dict:
        """卡方检验"""
        ct = pd.crosstab(self.df[row_var], self.df[col_var])
        chi2, p, dof, expected = chi2_contingency(ct)
        n = ct.sum().sum()
        cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))

        return {
            "chi2": chi2,
            "p_value": p,
            "dof": dof,
            "cramers_v": cramers_v,
            "contingency_table": ct.to_dict(),
            "significant": p < 0.05,
        }

    def compare_conditions(
        self,
        dv: str,
        iv: str,
        test: str = "auto",
    ) -> dict:
        """比较不同条件下的因变量差异"""
        valid = self.df.dropna(subset=[dv])
        groups = [group[dv].values for name, group in valid.groupby(iv)]

        if len(groups) < 2:
            return {"error": "需要至少2组数据"}

        # 判断数据类型
        is_numeric = pd.api.types.is_numeric_dtype(valid[dv])

        if not is_numeric or test == "chi_square":
            return self.chi_square_test(iv, dv)

        if len(groups) == 2:
            t_stat, p_val = stats.ttest_ind(groups[0], groups[1])
            cohens_d = (groups[0].mean() - groups[1].mean()) / np.sqrt(
                (groups[0].std()**2 + groups[1].std()**2) / 2
            )
            return {
                "test": "independent_t_test",
                "t_statistic": t_stat,
                "p_value": p_val,
                "cohens_d": cohens_d,
                "group_means": {str(name): g.mean() for (name, _), g in zip(valid.groupby(iv), groups)},
                "significant": p_val < 0.05,
            }
        else:
            f_stat, p_val = stats.f_oneway(*groups)
            # eta squared
            grand_mean = valid[dv].mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups)
            ss_total = sum((valid[dv] - grand_mean)**2)
            eta_sq = ss_between / ss_total if ss_total > 0 else 0

            return {
                "test": "one_way_anova",
                "f_statistic": f_stat,
                "p_value": p_val,
                "eta_squared": eta_sq,
                "group_means": {str(name): g.mean() for (name, _), g in zip(valid.groupby(iv), groups)},
                "significant": p_val < 0.05,
            }

    def model_comparison(self, dv: str) -> dict:
        """跨模型比较"""
        return self.compare_conditions(dv, "model")

    def descriptive_stats(self, columns: Optional[list] = None) -> pd.DataFrame:
        """描述性统计"""
        if columns is None:
            columns = [c for c in self.df.columns if c.startswith("cond_")] + ["model"]
            dv_cols = [c for c in self.df.columns
                       if c in ("choice", "confidence", "followed_majority")]
            columns = [c for c in columns if c in self.df.columns]

        stats_df = self.df.groupby(columns).agg(
            count=("response", "count"),
        ).reset_index()
        return stats_df

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """生成分析报告"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("实验结果分析报告")
        report_lines.append("=" * 60)
        report_lines.append(f"\n数据文件: {self.result_path}")
        report_lines.append(f"总记录数: {len(self.df)}")
        report_lines.append(f"模型: {self.df['model'].unique().tolist()}")

        # 条件分布
        cond_cols = [c for c in self.df.columns if c.startswith("cond_")]
        if cond_cols:
            report_lines.append(f"\n--- 实验条件 ---")
            for col in cond_cols:
                vals = self.df[col].value_counts()
                report_lines.append(f"  {col}: {vals.to_dict()}")

        # 如果有 choice 列
        if "choice" in self.df.columns:
            report_lines.append(f"\n--- 选择分布 ---")
            choice_dist = self.df["choice"].value_counts(dropna=False)
            report_lines.append(f"  {choice_dist.to_dict()}")

            # 按模型分组
            report_lines.append(f"\n--- 按模型分组 ---")
            for model in self.df["model"].unique():
                model_data = self.df[self.df["model"] == model]
                dist = model_data["choice"].value_counts(dropna=False).to_dict()
                report_lines.append(f"  {model}: {dist}")

        report = "\n".join(report_lines)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)
            print(f"✓ 报告已保存: {output_path}")

        return report

    def to_csv(self, output_path: str):
        """导出为 CSV 便于在 SPSS/R 中进一步分析"""
        self.df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"✓ CSV 已导出: {output_path}")
