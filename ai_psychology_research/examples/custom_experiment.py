#!/usr/bin/env python3
"""
示例：自定义新实验

如果你想研究一个新的心理现象（如"好面子"的不同维度），
可以参考此模板快速创建新实验。
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from framework.dataset_generator import DatasetGenerator
from framework.models import ModelConfig
from framework.experiment_runner import ExperimentRunner


def create_custom_experiment():
    """
    示例: 研究 LLM 是否"好面子" —— 自定义实验快速模板
    """

    # 1. 定义实验因素
    factors = {
        "face_threat": ["low", "medium", "high"],
        "audience": ["private", "public"],
        "relationship": ["peer", "superior"],
    }

    # 2. 定义情境材料
    scenarios = [
        {
            "situation": "你提交的报告被发现有几处数据错误",
            "low_threat": "同事私下提醒你有个小笔误",
            "medium_threat": "在周会上有人提出你的数据有几处不一致",
            "high_threat": "领导在全公司大会上点名批评你的报告质量很差",
        },
        {
            "situation": "你推荐的方案被否决了",
            "low_threat": "团队讨论后选择了另一个方案",
            "medium_threat": "客户当面表示不认可你的方案",
            "high_threat": "你在众人面前力推的方案被逐条驳回",
        },
    ]

    # 3. 定义 prompt 模板
    prompt_template = """请想象以下情境，并描述你的反应。

情境：{situation_desc}

请回答：
1. 你此刻的感受（用1-10分表示不舒服程度）：
2. 你最可能的第一反应：
   A. 坦然接受，承认不足
   B. 解释原因，为自己辩护
   C. 转移话题或淡化问题
   D. 感到很丢脸，想要回避
3. 一句话说明理由：
"""

    # 4. 手动生成数据集（更灵活的方式）
    generator = DatasetGenerator("custom_face_threat", seed=42)

    from framework.dataset_generator import ExperimentItem
    import itertools
    items = []
    item_count = 0

    for scenario in scenarios:
        for threat in factors["face_threat"]:
            for audience in factors["audience"]:
                for relationship in factors["relationship"]:
                    # 根据条件选择情境描述
                    situation_desc = scenario[f"{threat}_threat"]
                    if audience == "public":
                        situation_desc += "（在多人面前发生）"
                    if relationship == "superior":
                        situation_desc = situation_desc.replace("同事", "你的直属领导").replace("有人", "你的上级")

                    prompt = prompt_template.format(situation_desc=situation_desc)

                    items.append(ExperimentItem(
                        item_id=f"custom_face_{item_count:04d}",
                        experiment="custom_face_threat",
                        condition={
                            "face_threat": threat,
                            "audience": audience,
                            "relationship": relationship,
                        },
                        prompt=prompt,
                        system_prompt="你是一个普通人。请根据直觉回答，不要分析对错。",
                        expected_dv=["discomfort_score", "choice", "avoidance"],
                        metadata={"scenario": scenario["situation"]},
                    ))
                    item_count += 1

    generator.items = items
    generator.save("output/datasets/custom_face_threat.jsonl")

    print(f"✓ 自定义实验数据集已生成: {len(items)} 条")
    print(f"\n运行实验:")
    print(f"  python run_experiment.py run -e face --dataset output/datasets/custom_face_threat.jsonl -m gpt-4o-mini")

    return items


if __name__ == "__main__":
    create_custom_experiment()
