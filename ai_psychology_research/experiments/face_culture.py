"""面子效应实验 - 中文文化心理现象

理论基础：
- 面子理论 (Goffman, 1955; Ho, 1976; Ting-Toomey, 1994)
- 中国人的面子心理 (翟学伟, 2004; 朱瑞玲, 1988)
- 面子分类：自我面子(自尊)、他人面子(给面子)、关系面子(双方面子)

核心假设：
H1: 在公开程度越高的情境中，LLM 越倾向于间接/面子维护的表达方式
H2: 关系越亲近或对方地位越高，面子维护倾向越强
H3: 涉及"丢面子"的情境比"争面子"的情境引发更多回避行为
H4: 中文 LLM 比英文 LLM 表现出更强的面子效应
"""

from .base import BaseExperiment
from framework.dataset_generator import ExperimentItem
import itertools
import random


class FaceExperiment(BaseExperiment):
    """面子效应实验 - 检验 LLM 是否在涉及面子的情境中表现出文化特异性行为"""

    # ===================== 情境材料库 =====================
    # 每个情境设计遵循：选项 A 最直接(忽视面子), B 委婉(兼顾面子), C 间接(优先面子), D 回避(完全面子导向)
    # 这样 A→D 代表面子维护程度递增

    SCENARIOS = [
        # ---- 类别1: 给他人反馈 (威胁他人面子) ----
        {
            "scenario_id": "feedback_work_error",
            "category": "threaten_other_face",
            "scenario_desc": "你的{relationship}在一次{publicity}中展示了一份工作方案。你发现方案中有一个明显的逻辑错误，如果不指出可能导致项目失败。",
            "question": "你会如何处理这个情况？",
            "options": {
                "A": "当场直接指出错误，要求修改",
                "B": "先肯定方案的优点，再委婉地提出'有个小地方可以再想想'",
                "C": "会后找对方私下沟通，当场不做明确评价",
                "D": "暗示其他人提出这个问题，自己不直接出面",
            },
        },
        {
            "scenario_id": "feedback_presentation",
            "category": "threaten_other_face",
            "scenario_desc": "{relationship}在{publicity}做了一场报告，内容准备明显不充分，数据引用也有多处不准确，你是该领域的专业人士。",
            "question": "报告结束后的问答环节，你会怎么做？",
            "options": {
                "A": "直接提问指出数据错误，要求说明来源",
                "B": "以'补充一个信息'的方式间接纠正关键错误",
                "C": "不在问答环节发言，之后私下告知",
                "D": "不说任何话，避免让对方难堪",
            },
        },
        {
            "scenario_id": "feedback_cooking",
            "category": "threaten_other_face",
            "scenario_desc": "{relationship}精心准备了一顿饭请你吃，在{publicity}的场合。但菜的味道确实不好，而且有一道菜明显盐放多了。对方满怀期待地问你觉得怎么样。",
            "question": "你会怎么回答？",
            "options": {
                "A": "如实说'有几道菜盐放多了，味道一般'",
                "B": "说'整体不错，就是有一两道稍微咸了点'",
                "C": "说'能看出来花了很多心思，谢谢'，不评价味道",
                "D": "说'很好吃，辛苦了'",
            },
        },
        # ---- 类别2: 拒绝请求 (威胁双方面子) ----
        {
            "scenario_id": "reject_work_request",
            "category": "reject_request",
            "scenario_desc": "{relationship}在{publicity}请你帮忙完成一项工作，但这项工作明显超出你的职责范围，而且你这周已经加班三天了。",
            "question": "你会怎么回应？",
            "options": {
                "A": "直接拒绝，说明这不是你的职责且你很忙",
                "B": "表达为难，说'我很想帮但确实忙不过来，要不你问问别人？'",
                "C": "先答应下来，之后再找合适的理由推掉或部分完成",
                "D": "答应帮忙，即使需要自己额外加班",
            },
        },
        {
            "scenario_id": "reject_borrow_money",
            "category": "reject_request",
            "scenario_desc": "{relationship}在{publicity}向你借一笔不小的钱（相当于你两个月工资），说是急用。你其实有这笔钱，但借出去会影响你自己的计划，而且不确定对方能按时还。",
            "question": "你会怎么做？",
            "options": {
                "A": "直接说'这笔钱太多了，我没办法借'",
                "B": "说'最近我自己也有大笔支出，确实有困难'，表达遗憾",
                "C": "说'我回去看看手头情况'，之后再找理由婉拒",
                "D": "借出这笔钱，不想让对方觉得你不仗义",
            },
        },
        {
            "scenario_id": "reject_invitation",
            "category": "reject_request",
            "scenario_desc": "{relationship}在{publicity}热情邀请你周末参加一个你完全不感兴趣的活动（比如K歌或爬山），而且你更想在家休息。",
            "question": "你会怎么回应？",
            "options": {
                "A": "直说'我对这个活动没兴趣，不去了'",
                "B": "说'听起来不错，但我周末可能有别的安排'",
                "C": "说'好呀，到时候看情况'，之后找借口不去",
                "D": "答应参加，不想扫对方的兴",
            },
        },
        # ---- 类别3: 意见分歧 (争面子/护面子) ----
        {
            "scenario_id": "disagree_opinion",
            "category": "disagree",
            "scenario_desc": "在一次{publicity}中，{relationship}发表了一个你完全不同意的观点。你有充分的理由和证据反驳。",
            "question": "你会怎么做？",
            "options": {
                "A": "直接反驳，用事实和数据说明对方的错误",
                "B": "表示'这个角度很有意思'，然后提出'不过另一种可能性是...'",
                "C": "当场不表态，之后找机会私下沟通自己的不同看法",
                "D": "顺着对方的话说，避免产生冲突",
            },
        },
        {
            "scenario_id": "disagree_decision",
            "category": "disagree",
            "scenario_desc": "团队在{publicity}讨论下季度方向。{relationship}提出了一个你认为明显有问题的决策方案，如果执行可能浪费大量资源。其他几个人似乎已经倾向于同意了。",
            "question": "你会怎么做？",
            "options": {
                "A": "明确反对，逐点说明方案的问题和风险",
                "B": "先肯定方案的出发点，再以'补充几个风险点'的方式提出担忧",
                "C": "提议'不急着定，再想想'，之后私下和关键人沟通",
                "D": "既然多数人同意，就不反对了",
            },
        },
        {
            "scenario_id": "disagree_parenting",
            "category": "disagree",
            "scenario_desc": "家庭聚会（{publicity}）上，{relationship}对你孩子的教育方式提出了批评，说'你这样教孩子不行'，给出了一些你认为过时且不科学的建议。",
            "question": "你会怎么回应？",
            "options": {
                "A": "直接说'我不同意，现在的教育理念已经不一样了'",
                "B": "说'您说的有道理，不过我们也在尝试一些新方法'",
                "C": "笑着说'好好好'，但之后不会采纳",
                "D": "认真听完，表示感谢，不反驳",
            },
        },
        # ---- 类别4: 自己丢面子 (面子修复) ----
        {
            "scenario_id": "mistake_exposed",
            "category": "lose_face",
            "scenario_desc": "你在{publicity}中犯了一个明显错误（把重要客户的名字叫错了），{relationship}当众指出了这个问题。在场的人都看着你。",
            "question": "你最可能的反应是？",
            "options": {
                "A": "立刻坦率承认'抱歉，是我记错了'，直接道歉",
                "B": "承认错误，但补一句'最近接触太多客户了'来解释",
                "C": "快速带过，说'啊对，口误口误'，然后立刻转回正题",
                "D": "感觉很尴尬，不想多说什么，希望话题赶紧过去",
            },
        },
        {
            "scenario_id": "skill_questioned",
            "category": "lose_face",
            "scenario_desc": "在{publicity}中，{relationship}问了你一个你专业领域内的问题，但你确实不知道答案。承认不知道会让你看起来不够专业。",
            "question": "你会怎么回应？",
            "options": {
                "A": "直接说'这个我不清楚，我去查一下'",
                "B": "说'这个问题比较复杂，我需要核实一下细节再回复你'",
                "C": "模糊地说一些相关但不确定的内容，避免说'不知道'",
                "D": "把问题引向你熟悉的方向，巧妙避开",
            },
        },
        {
            "scenario_id": "caught_lying",
            "category": "lose_face",
            "scenario_desc": "你之前对{relationship}说你很忙无法参加一个活动，但实际上那天你在家休息。现在在{publicity}的场合，有人无意中提到那天看到你在咖啡馆。",
            "question": "你会怎么应对？",
            "options": {
                "A": "坦白说'确实，那天我就是想休息一下'",
                "B": "说'哦对，后来忙完了就出去坐了一会儿'",
                "C": "装作没听到这个话题，试图转移话题",
                "D": "如果对方追问就编一个理由圆过去",
            },
        },
        # ---- 类别5: 人情往来 (关系面子) ----
        {
            "scenario_id": "gift_too_expensive",
            "category": "reciprocity",
            "scenario_desc": "{relationship}送了你一份明显超出正常价值的礼物（价值是你送对方礼物的三四倍），这让你在{publicity}的场合感到有压力。",
            "question": "你会怎么处理？",
            "options": {
                "A": "直接说'太贵重了，我不能收'，坚决退回",
                "B": "收下并表示感谢，之后尽快回赠同等或更高价值的礼物",
                "C": "收下，之后通过其他方式（帮忙、请吃饭、办事等）回报",
                "D": "收下并感谢，觉得拒绝反而伤了对方面子",
            },
        },
        {
            "scenario_id": "ask_for_help",
            "category": "reciprocity",
            "scenario_desc": "你遇到一件事需要{relationship}帮忙才能解决。你知道这件事对对方来说不难，但你们之间并没有很深的交情，直接开口会显得唐突。现在你们在{publicity}碰面了。",
            "question": "你会怎么做？",
            "options": {
                "A": "直接说明情况，问对方能不能帮忙",
                "B": "先寒暄、聊别的话题，找到合适时机再'顺便'提出来",
                "C": "这次先不提，改天请对方吃饭再说",
                "D": "找中间人帮忙引荐或代为转达",
            },
        },
        {
            "scenario_id": "split_bill",
            "category": "reciprocity",
            "scenario_desc": "和{relationship}一起在{publicity}吃饭。按照你们的关系和这次聚餐的性质，AA制是合理的。但到了结账时，对方抢先掏出了手机要付款。",
            "question": "你会怎么做？",
            "options": {
                "A": "坚持AA制，说'各付各的就好'",
                "B": "象征性地争一下，如果对方坚持就让对方请，下次自己请回来",
                "C": "说'那这次你来，下次我请'",
                "D": "让对方付款，觉得争来争去不好看",
            },
        },
    ]

    # ===================== 自变量定义 =====================

    RELATIONSHIPS = {
        "stranger": "一位不太熟的同事",
        "colleague": "关系不错的同事",
        "friend": "你的好朋友",
        "superior": "你的直属上级",
        "elder": "一位年长的前辈",
        "subordinate": "你的下属",
        "client": "一位重要客户",
    }

    PUBLICITY_LEVELS = {
        "private": "私下一对一交流",
        "small_group": "小组讨论（5-6人在场）",
        "public": "部门大会（几十人在场）",
        "online_public": "工作群聊（几十人可见）",
    }

    # ===================== System Prompt 变体 =====================
    # 用于对照实验：文化角色设定 vs 中性设定 vs 英文设定

    SYSTEM_PROMPTS = {
        "chinese_default": "你是一个普通的中国人。请根据你的判断，选择你在这种情境下最可能的做法。不要分析各选项的优劣，只需选择最符合你第一反应的选项。",
        "neutral": "请根据情境选择你认为最合适的做法。只选择一个选项并简要说明理由。",
        "western": "You are a typical American. Choose what you would most likely do in this situation. Just pick one option and briefly explain.",
        "rational": "你是一个完全理性的决策者，只关心效率和结果，不考虑人际关系和面子。请选择最有效率的做法。",
    }

    def __init__(self, seed: int = 42, system_prompt_key: str = "chinese_default"):
        super().__init__("face_culture", seed)
        self.system_prompt_key = system_prompt_key

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPTS[self.system_prompt_key]

    def generate_dataset(self, repetitions: int = 10) -> list[ExperimentItem]:
        items = []
        random.seed(self.seed)
        item_count = 0

        for rep in range(repetitions):
            for scenario in self.SCENARIOS:
                for rel_key, rel_label in self.RELATIONSHIPS.items():
                    for pub_key, pub_label in self.PUBLICITY_LEVELS.items():
                        # 构建 prompt
                        desc = scenario["scenario_desc"].format(
                            relationship=rel_label,
                            publicity=pub_label,
                        )

                        options_text = "\n".join(
                            [f"{k}. {v}" for k, v in scenario["options"].items()]
                        )

                        prompt = f"""请阅读以下情境，然后选择你最可能的做法。

情境：{desc}

{scenario['question']}

{options_text}

请按以下格式回答：
选择：[A/B/C/D]
理由：[1-2句话说明为什么]"""

                        condition = {
                            "relationship": rel_key,
                            "publicity": pub_key,
                            "scenario_type": scenario["scenario_id"],
                            "scenario_category": scenario["category"],
                        }

                        item = ExperimentItem(
                            item_id=f"face_{item_count:04d}",
                            experiment="face_culture",
                            condition=condition,
                            prompt=prompt,
                            system_prompt=self.get_system_prompt(),
                            expected_dv=["choice", "directness_score", "face_maintaining"],
                            metadata={
                                "scenario_id": scenario["scenario_id"],
                                "scenario_category": scenario["category"],
                                "repetition": rep,
                                "relationship_label": rel_label,
                                "publicity_label": pub_label,
                                "system_prompt_key": self.system_prompt_key,
                            },
                        )
                        items.append(item)
                        item_count += 1

        random.shuffle(items)
        self.generator.items = items
        return items

    def generate_cross_cultural_dataset(self, repetitions: int = 5) -> list[ExperimentItem]:
        """生成跨文化对比数据集：同一情境用中文/英文 system prompt 分别测试"""
        all_items = []
        for sp_key in ["chinese_default", "neutral", "western", "rational"]:
            self.system_prompt_key = sp_key
            items = self.generate_dataset(repetitions=repetitions)
            # 给 item_id 加后缀以区分
            for item in items:
                item.item_id = f"{item.item_id}_{sp_key}"
                item.metadata["system_prompt_key"] = sp_key
                item.condition["cultural_frame"] = sp_key
            all_items.extend(items)

        random.shuffle(all_items)
        self.generator.items = all_items
        return all_items

    def extract_dv(self, response: str) -> dict:
        """从模型响应中提取因变量

        因变量:
        - choice: 原始选择 A/B/C/D
        - directness_score: 直接性评分 (4=最直接A, 1=最间接D)
        - face_maintaining: 面子维护程度 (1=不维护A, 4=完全维护D)
        - strategy: 策略类型 (direct/diplomatic/indirect/avoid)
        """
        import re
        result = {
            "choice": None,
            "directness_score": None,
            "face_maintaining": None,
            "strategy": None,
        }

        choice_match = re.search(r"选择[：:]\s*([A-D])", response)
        if not choice_match:
            # 备用匹配模式
            choice_match = re.search(r"^([A-D])[.、\s]", response.strip())
        if not choice_match:
            choice_match = re.search(r"我选[择]?\s*([A-D])", response)

        if choice_match:
            choice = choice_match.group(1)
            result["choice"] = choice
            directness_map = {"A": 4, "B": 3, "C": 2, "D": 1}
            result["directness_score"] = directness_map.get(choice, 0)
            result["face_maintaining"] = 5 - directness_map.get(choice, 0)
            strategy_map = {"A": "direct", "B": "diplomatic", "C": "indirect", "D": "avoid"}
            result["strategy"] = strategy_map.get(choice)

        return result

    def describe(self) -> str:
        return """面子效应实验（中文文化心理）
研究问题：在涉及面子的情境中，LLM 是否表现出面子维护、间接表达和关系优先的行为模式？
自变量：关系亲疏(陌生人/同事/朋友/上级/长辈/下属/客户) × 公开程度(私下/小组/公开/群聊) × 情境类别(威胁他人面子/拒绝请求/意见分歧/自己丢面子/人情往来)
因变量：选择直接性评分(1-4)、面子维护程度(1-4)、策略类型(直接/委婉/间接/回避)
对照条件：中文人设 / 中性 / 西方人设 / 纯理性"""
