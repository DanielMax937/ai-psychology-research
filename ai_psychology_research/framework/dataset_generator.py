"""数据集生成器 - 根据实验设计自动生成 prompt 数据集"""

import json
import itertools
import random
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field, asdict


@dataclass
class ExperimentItem:
    """单个实验试次"""
    item_id: str
    experiment: str
    condition: dict          # 自变量条件组合
    prompt: str              # 发送给模型的 prompt
    system_prompt: str = ""
    expected_dv: list = field(default_factory=list)  # 预期因变量名称
    metadata: dict = field(default_factory=dict)


class DatasetGenerator:
    """根据实验设计生成完整数据集"""

    def __init__(self, experiment_name: str, seed: int = 42):
        self.experiment_name = experiment_name
        self.seed = seed
        self.items: list[ExperimentItem] = []
        random.seed(seed)

    def generate_factorial(
        self,
        factors: dict[str, list],
        prompt_template: str,
        system_prompt: str = "",
        scenarios: list[dict] = None,
        repetitions: int = 1,
        expected_dv: list[str] = None,
    ) -> list[ExperimentItem]:
        """
        全因素设计生成器

        Args:
            factors: 自变量及其水平, e.g. {"majority_size": ["60%", "80%"], "source": ["expert", "peer"]}
            prompt_template: prompt 模板, 用 {variable_name} 占位
            system_prompt: system prompt
            scenarios: 情境材料列表, 每个是 dict, 模板中可引用
            repetitions: 每个条件重复次数
            expected_dv: 因变量名称列表
        """
        if scenarios is None:
            scenarios = [{}]
        if expected_dv is None:
            expected_dv = ["choice"]

        factor_names = list(factors.keys())
        factor_levels = list(factors.values())

        items = []
        item_count = 0

        for rep in range(repetitions):
            for scenario in scenarios:
                for combination in itertools.product(*factor_levels):
                    condition = dict(zip(factor_names, combination))

                    # 合并 scenario 和 condition 用于模板填充
                    template_vars = {**scenario, **condition}
                    try:
                        prompt = prompt_template.format(**template_vars)
                    except KeyError as e:
                        raise ValueError(f"模板变量缺失: {e}. 可用变量: {list(template_vars.keys())}")

                    sys_prompt = system_prompt.format(**template_vars) if system_prompt else ""

                    item = ExperimentItem(
                        item_id=f"{self.experiment_name}_{item_count:04d}",
                        experiment=self.experiment_name,
                        condition=condition,
                        prompt=prompt,
                        system_prompt=sys_prompt,
                        expected_dv=expected_dv,
                        metadata={"scenario": scenario, "repetition": rep},
                    )
                    items.append(item)
                    item_count += 1

        # 随机化顺序
        random.shuffle(items)
        self.items = items
        return items

    def generate_multi_turn(
        self,
        factors: dict[str, list],
        turn_templates: list[str],
        system_prompt: str = "",
        scenarios: list[dict] = None,
        repetitions: int = 1,
        expected_dv: list[str] = None,
    ) -> list[ExperimentItem]:
        """
        多轮对话实验生成器（用于认知失调等实验）

        turn_templates: 多轮模板列表, 按顺序发送
        """
        if scenarios is None:
            scenarios = [{}]
        if expected_dv is None:
            expected_dv = ["choice_change", "rationalization"]

        factor_names = list(factors.keys())
        factor_levels = list(factors.values())

        items = []
        item_count = 0

        for rep in range(repetitions):
            for scenario in scenarios:
                for combination in itertools.product(*factor_levels):
                    condition = dict(zip(factor_names, combination))
                    template_vars = {**scenario, **condition}

                    turns = []
                    for tmpl in turn_templates:
                        turns.append(tmpl.format(**template_vars))

                    # 将多轮信息编码在 prompt 中, 用 separator 分隔
                    prompt = "\n===TURN_SEPARATOR===\n".join(turns)

                    item = ExperimentItem(
                        item_id=f"{self.experiment_name}_{item_count:04d}",
                        experiment=self.experiment_name,
                        condition=condition,
                        prompt=prompt,
                        system_prompt=system_prompt.format(**template_vars) if system_prompt else "",
                        expected_dv=expected_dv,
                        metadata={"scenario": scenario, "repetition": rep, "is_multi_turn": True},
                    )
                    items.append(item)
                    item_count += 1

        random.shuffle(items)
        self.items = items
        return items

    def save(self, output_path: str):
        """保存数据集为 JSON Lines 格式"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            for item in self.items:
                f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

        print(f"✓ 数据集已保存: {path} ({len(self.items)} 条)")
        return path

    def summary(self) -> dict:
        """返回数据集摘要"""
        if not self.items:
            return {"total": 0}

        conditions = {}
        for item in self.items:
            for k, v in item.condition.items():
                conditions.setdefault(k, set()).add(v)

        return {
            "experiment": self.experiment_name,
            "total_items": len(self.items),
            "factors": {k: list(v) for k, v in conditions.items()},
            "n_conditions": len(set(
                tuple(sorted(item.condition.items())) for item in self.items
            )),
        }
