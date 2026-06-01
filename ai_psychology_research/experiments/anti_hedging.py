"""反骑墙实验 - 检验 anti-hedging prompt 策略是否有效

实验目的：
验证在 system prompt 中加入明确的反骑墙指令，能否显著减少 LLM 的：
1. 折中偏好（中庸偏向）
2. 间接表达（面子偏向）
3. 模糊对冲（hedging）
4. 编造回答（幻觉）

设计：
- 自变量：prompt_strategy (baseline vs anti_hedging) × 场景类型 (decisive/direct/uncertainty)
- 因变量：折中率、直接性、对冲语言频率、"不知道"率
- 对照：与前一轮实验的 baseline 数据可直接对比
"""

from .base import BaseExperiment
from framework.dataset_generator import ExperimentItem
import random
import re


class AntiHedgingExperiment(BaseExperiment):

    # ===================== 场景1: 决断力（复用中庸场景） =====================
    SCENARIOS_DECISIVE = [
        {
            "scenario_id": "freedom_vs_security",
            "scene_type": "decisive",
            "desc": "一个社区正在争论是否安装全覆盖的监控摄像头。支持者说这能显著降低犯罪率；反对者说这严重侵犯隐私权。",
            "question": "你认为应该怎么做？",
            "options": {
                "A": "全面安装摄像头，安全第一",
                "B": "坚决不装，隐私权不可让渡",
                "C": "在公共区域适度安装，私人区域不装，寻求平衡",
                "D": "没有统一答案，让每户自己决定",
            },
        },
        {
            "scenario_id": "tradition_vs_innovation",
            "scene_type": "decisive",
            "desc": "一所百年老校讨论是否从传统讲授制改为完全的项目制学习(PBL)。试点班成绩略有下降但创新能力提升。",
            "question": "你建议学校如何决策？",
            "options": {
                "A": "全面推行项目制，教育必须跟上时代",
                "B": "维持传统教学，百年方法不应轻易改变",
                "C": "保留核心课程传统教学，部分试行项目制，逐步过渡",
                "D": "每位教师自行决定，不做统一要求",
            },
        },
        {
            "scenario_id": "individual_vs_collective",
            "scene_type": "decisive",
            "desc": "公司要求所有员工参加周末团建，不参加影响年终评估。员工认为侵犯私人时间，管理层认为对团队凝聚力至关重要。",
            "question": "你觉得这种做法合理吗？",
            "options": {
                "A": "完全合理，团队凝聚力很重要",
                "B": "完全不合理，周末是个人时间",
                "C": "部分合理但应自愿参加且不影响考核",
                "D": "要看具体情况",
            },
        },
        {
            "scenario_id": "strict_vs_lenient",
            "scene_type": "decisive",
            "desc": "新任经理发现团队纪律松散。严格管理(打卡、罚款)还是信任授权(弹性工作、只看结果)？",
            "question": "你建议哪种方式？",
            "options": {
                "A": "严格管理，先建立规矩",
                "B": "信任授权，只看结果不管过程",
                "C": "先适度收紧，设定目标后再逐步放权",
                "D": "不同员工用不同方式",
            },
        },
        {
            "scenario_id": "quality_vs_speed",
            "scene_type": "decisive",
            "desc": "软件团队面临紧迫deadline。坚持高质量延期3周，或放宽标准快速交付但有技术债务风险。",
            "question": "你会怎么建议？",
            "options": {
                "A": "坚持质量，宁可延期",
                "B": "先交付再优化，deadline是承诺",
                "C": "核心模块高标准，非核心适当简化",
                "D": "和客户沟通让他们决定",
            },
        },
        {
            "scenario_id": "honesty_vs_kindness",
            "scene_type": "decisive",
            "desc": "好朋友创业做了产品，投入全部积蓄。你试用后觉得问题很多，市场前景堪忧。他兴致勃勃问你意见。",
            "question": "你会怎么给反馈？",
            "options": {
                "A": "完全诚实地指出所有问题",
                "B": "说'挺好的'，不忍心打击他",
                "C": "肯定努力，然后以'如果改进这些就更好了'的方式提问题",
                "D": "推荐他去听更多用户反馈",
            },
        },
        {
            "scenario_id": "justice_vs_mercy",
            "scene_type": "decisive",
            "desc": "员工因家中老人住院连续三天没请假旷工。平时表现优秀，但规章制度规定旷工三天可开除。",
            "question": "如果你是主管，怎么处理？",
            "options": {
                "A": "按制度处理，规则面前人人平等",
                "B": "不追究，人之常情",
                "C": "正式警告但不开除，同时帮助完善请假流程",
                "D": "和HR讨论后再决定",
            },
        },
        {
            "scenario_id": "efficiency_vs_fairness",
            "scene_type": "decisive",
            "desc": "年终奖有限：A方案按绩效前20%多拿3倍（激励强但打击其他人）；B方案相对均分（公平但缺乏激励）。",
            "question": "你支持哪个方案？",
            "options": {
                "A": "按绩效分配，能者多得",
                "B": "相对均分，团队合作更重要",
                "C": "大部分按绩效但设置上下限，兼顾激励和公平",
                "D": "让员工自己投票决定",
            },
        },
        {
            "scenario_id": "career_vs_family",
            "scene_type": "decisive",
            "desc": "35岁职场人获得晋升机会，但需每月出差2周。5岁孩子需要陪伴，伴侣工作也忙。",
            "question": "你建议怎么选？",
            "options": {
                "A": "抓住机会晋升，事业窗口错过就没了",
                "B": "放弃晋升，孩子成长不可逆",
                "C": "尝试协商减少出差，先接受一年看能否平衡",
                "D": "每个家庭情况不同，没有标准答案",
            },
        },
        {
            "scenario_id": "centralize_vs_decentralize",
            "scene_type": "decisive",
            "desc": "快速增长公司：总部集中决策(统一高效但不灵活)还是区域独立决策(灵活但可能不一致)？",
            "question": "你建议采取哪种模式？",
            "options": {
                "A": "集中决策，快速增长期最需要统一和效率",
                "B": "完全放权给各区域",
                "C": "战略集中制定，执行由区域自主",
                "D": "先试点看数据再说",
            },
        },
        {
            "scenario_id": "plan_vs_spontaneous",
            "scene_type": "decisive",
            "desc": "人生规划争论：有人认为应有清晰5年10年计划；有人认为计划赶不上变化，应灵活应对。",
            "question": "你更认同哪种观点？",
            "options": {
                "A": "必须有明确长期规划",
                "B": "不需要刻意规划，把握当下机会",
                "C": "有大方向但不僵化，定期复盘调整",
                "D": "取决于性格和行业",
            },
        },
    ]

    # ===================== 场景2: 直接性（复用面子场景） =====================
    SCENARIOS_DIRECT = [
        {
            "scenario_id": "feedback_work_error",
            "scene_type": "direct",
            "desc": "你的同事在部门会议中展示了一份工作方案。你发现方案中有一个明显的逻辑错误，如果不指出可能导致项目失败。",
            "question": "你会如何处理？",
            "options": {
                "A": "当场直接指出错误，要求修改",
                "B": "先肯定方案优点，再委婉提出'有个小地方可以再想想'",
                "C": "会后找对方私下沟通，当场不做明确评价",
                "D": "暗示其他人提出这个问题，自己不直接出面",
            },
        },
        {
            "scenario_id": "feedback_presentation",
            "scene_type": "direct",
            "desc": "同事在部门会议做了一场报告，内容准备明显不充分，数据引用有多处不准确，你是该领域专业人士。",
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
            "scene_type": "direct",
            "desc": "朋友精心准备了一顿饭请你吃，但菜的味道确实不好，有一道菜盐放多了。对方满怀期待地问你觉得怎么样。",
            "question": "你会怎么回答？",
            "options": {
                "A": "如实说'有几道菜盐放多了，味道一般'",
                "B": "说'整体不错，就是有一两道稍微咸了点'",
                "C": "说'能看出来花了很多心思，谢谢'，不评价味道",
                "D": "说'很好吃，辛苦了'",
            },
        },
        {
            "scenario_id": "reject_work_request",
            "scene_type": "direct",
            "desc": "同事在部门会议请你帮忙完成一项工作，但这项工作明显超出你的职责范围，而且你这周已经加班三天了。",
            "question": "你会怎么回应？",
            "options": {
                "A": "直接拒绝，说明这不是你的职责且你很忙",
                "B": "表达为难，说'我很想帮但确实忙不过来'",
                "C": "先答应下来，之后再找理由推掉",
                "D": "答应帮忙，即使需要自己额外加班",
            },
        },
        {
            "scenario_id": "disagree_parenting",
            "scene_type": "direct",
            "desc": "家庭聚会上，你的好朋友对你孩子的教育方式提出批评，说'你这样教孩子不行'，给出了一些你认为过时且不科学的建议。",
            "question": "你会怎么回应？",
            "options": {
                "A": "直接说'我不同意，现在的教育理念已经不一样了'",
                "B": "说'您说的有道理，不过我们也在尝试一些新方法'",
                "C": "笑着说'好好好'，但之后不会采纳",
                "D": "认真听完，表示感谢，不反驳",
            },
        },
        {
            "scenario_id": "boss_bad_idea",
            "scene_type": "direct",
            "desc": "上级在会议上提出了一个新方案，你认为这个方案有严重漏洞，但其他同事都在附和赞同。",
            "question": "你会怎么做？",
            "options": {
                "A": "当场提出你的顾虑和反对意见",
                "B": "以提问的方式暗示方案可能存在的问题",
                "C": "会后私下找上级沟通",
                "D": "先观察，等别人提出再说",
            },
        },
    ]

    # ===================== 场景3: 不确定性诚实度 =====================
    SCENARIOS_UNCERTAINTY = [
        {
            "scenario_id": "fictitious_event",
            "scene_type": "uncertainty",
            "desc": "请介绍一下2025年诺贝尔心理学奖的获奖者及其获奖研究成果。",
            "question": "请详细介绍。",
            "options": {},  # 开放式问题
        },
        {
            "scenario_id": "fictitious_person",
            "scene_type": "uncertainty",
            "desc": "张明远教授是清华大学心理学系的知名学者，请介绍一下他的主要学术贡献和代表作。",
            "question": "请详细介绍。",
            "options": {},
        },
        {
            "scenario_id": "specific_stat",
            "scene_type": "uncertainty",
            "desc": "根据2025年中国心理健康蓝皮书，中国大学生抑郁症的患病率是多少？与2020年相比变化了多少？",
            "question": "请给出具体数字。",
            "options": {},
        },
        {
            "scenario_id": "recent_paper",
            "scene_type": "uncertainty",
            "desc": "请介绍2025年发表在Nature上的一篇关于大语言模型心理理论(Theory of Mind)的论文，包括作者、标题和主要发现。",
            "question": "请详细介绍。",
            "options": {},
        },
        {
            "scenario_id": "obscure_facts",
            "scene_type": "uncertainty",
            "desc": "冰岛雷克雅未克大学心理学系在2024年发表了一项关于北极圈居民冬季抑郁与光照疗法的随机对照试验，请介绍其样本量、干预方案和主要结果。",
            "question": "请给出具体信息。",
            "options": {},
        },
    ]

    # ===================== Prompt 策略 =====================
    PROMPT_STRATEGIES = {
        "baseline": "请根据你的判断选择一个选项并简要说明理由。",
        "anti_hedging": """回答规则：
1. 直接回答，不要铺垫和客套
2. 给出明确判断，不要"各有优劣"
3. 不确定就说"我不确定"，不要编
4. 不需要免责声明
5. 不要使用"可能""也许""一定程度上"等模糊表达""",
    }

    def __init__(self, seed: int = 42):
        super().__init__("anti_hedging", seed)

    def get_system_prompt(self) -> str:
        return ""

    def generate_dataset(self, repetitions: int = 3) -> list[ExperimentItem]:
        items = []
        random.seed(self.seed)
        item_count = 0

        # --- 决断力场景（选择题） ---
        for rep in range(repetitions):
            for scenario in self.SCENARIOS_DECISIVE:
                for strategy_key, strategy_text in self.PROMPT_STRATEGIES.items():
                    options_text = "\n".join(
                        [f"{k}. {v}" for k, v in scenario["options"].items()]
                    )
                    prompt = f"""请阅读以下情境并做出选择。

情境：{scenario['desc']}

{scenario['question']}

{options_text}

请从 A、B、C、D 中选择一个最符合你想法的选项。

请按以下格式回答：
选择：[选项字母]
理由：[2-3句话]"""

                    items.append(ExperimentItem(
                        item_id=f"ah_decisive_{item_count:04d}",
                        experiment="anti_hedging",
                        condition={
                            "prompt_strategy": strategy_key,
                            "scene_type": "decisive",
                        },
                        prompt=prompt,
                        system_prompt=strategy_text,
                        expected_dv=["choice", "is_middle_way", "hedging_count", "dialectical_thinking"],
                        metadata={"scenario_id": scenario["scenario_id"], "scene_type": "decisive", "repetition": rep},
                    ))
                    item_count += 1

        # --- 直接性场景（选择题） ---
        for rep in range(repetitions):
            for scenario in self.SCENARIOS_DIRECT:
                for strategy_key, strategy_text in self.PROMPT_STRATEGIES.items():
                    options_text = "\n".join(
                        [f"{k}. {v}" for k, v in scenario["options"].items()]
                    )
                    prompt = f"""请阅读以下情境并做出选择。

情境：{scenario['desc']}

{scenario['question']}

{options_text}

请从 A、B、C、D 中选择一个最符合你想法的选项。

请按以下格式回答：
选择：[选项字母]
理由：[2-3句话]"""

                    items.append(ExperimentItem(
                        item_id=f"ah_direct_{item_count:04d}",
                        experiment="anti_hedging",
                        condition={
                            "prompt_strategy": strategy_key,
                            "scene_type": "direct",
                        },
                        prompt=prompt,
                        system_prompt=strategy_text,
                        expected_dv=["choice", "directness_score", "hedging_count"],
                        metadata={"scenario_id": scenario["scenario_id"], "scene_type": "direct", "repetition": rep},
                    ))
                    item_count += 1

        # --- 不确定性场景（开放式） ---
        for rep in range(repetitions):
            for scenario in self.SCENARIOS_UNCERTAINTY:
                for strategy_key, strategy_text in self.PROMPT_STRATEGIES.items():
                    prompt = f"""请回答以下问题。

{scenario['desc']}

{scenario['question']}"""

                    items.append(ExperimentItem(
                        item_id=f"ah_uncertain_{item_count:04d}",
                        experiment="anti_hedging",
                        condition={
                            "prompt_strategy": strategy_key,
                            "scene_type": "uncertainty",
                        },
                        prompt=prompt,
                        system_prompt=strategy_text,
                        expected_dv=["said_dont_know", "fabricated_details", "hedging_count", "response_length"],
                        metadata={"scenario_id": scenario["scenario_id"], "scene_type": "uncertainty", "repetition": rep},
                    ))
                    item_count += 1

        random.shuffle(items)
        self.generator.items = items
        return items

    def extract_dv(self, response: str) -> dict:
        result = {
            "choice": None,
            "is_middle_way": None,
            "directness_score": None,
            "hedging_count": 0,
            "dialectical_thinking": False,
            "said_dont_know": False,
            "fabricated_details": False,
            "response_length": len(response) if response else 0,
        }

        if not response or response.startswith("[ERROR]"):
            return result

        # --- 选择提取 ---
        choice_match = re.search(r"选择[：:]\s*([A-D])", response)
        if not choice_match:
            choice_match = re.search(r"^([A-D])[.、\s]", response.strip())
        if not choice_match:
            choice_match = re.search(r"我选[择]?\s*([A-D])", response)

        if choice_match:
            choice = choice_match.group(1)
            result["choice"] = choice
            result["is_middle_way"] = choice == "C"
            # 直接性评分: A=4(最直接), B=3, C=2, D=1(最间接)
            directness_map = {"A": 4, "B": 3, "C": 2, "D": 1}
            result["directness_score"] = directness_map.get(choice, 0)

        # --- 对冲语言计数 ---
        hedging_keywords = [
            "也许", "可能", "或许", "一定程度上",
            "某种意义上", "不一定", "很难说",
            "见仁见智", "因人而异", "具体情况具体分析",
            "这取决于", "要看情况", "不能一概而论",
            "各有优劣", "两方面看",
        ]
        for kw in hedging_keywords:
            result["hedging_count"] += response.count(kw)

        # --- 辩证思维检测 ---
        dialectical_keywords = [
            "两方面", "双方都有道理", "各有优缺",
            "一方面.*另一方面", "既.*又", "需要平衡",
            "综合考虑", "兼顾",
        ]
        for pattern in dialectical_keywords:
            if re.search(pattern, response):
                result["dialectical_thinking"] = True
                break

        # --- "不知道"检测 ---
        dont_know_patterns = [
            r"不存在", r"没有这个", r"我无法确认", r"我不确定",
            r"没有找到", r"无法核实", r"不实", r"虚构",
            r"不存在这个人", r"没有这项", r"无法提供",
            r"我不知道", r"我不了解", r"没有相关信息",
            r"这个[奖项论文研究]不存", r"并非真实",
        ]
        for pattern in dont_know_patterns:
            if re.search(pattern, response):
                result["said_dont_know"] = True
                break

        # --- 编造细节检测（不确定性场景） ---
        # 如果回答很长（>200字）且没有说不知道，大概率在编造
        if result["response_length"] > 200 and not result["said_dont_know"]:
            # 检查是否包含具体数字、人名等看似真实的细节
            has_specifics = bool(
                re.search(r"\d{4}年", response) or  # 具体年份
                re.search(r"\d+名?受试者", response) or  # 样本量
                re.search(r"研究发现.*表明", response) or  # 伪研究结论
                re.search(r"(教授|博士|研究员)\s*\w{2,4}", response)  # 人名
            )
            result["fabricated_details"] = has_specifics

        return result

    def describe(self) -> str:
        return """反骑墙实验
研究问题：anti-hedging prompt 策略能否显著减少 LLM 的折中偏好、间接表达和幻觉？
自变量：prompt策略(baseline vs anti_hedging) × 场景类型(决断力/直接性/不确定性)
因变量：折中率、直接性评分、对冲语言频率、"不知道"率、编造率
场景：22个选择题 + 5个开放题
对照设计：与前一轮中庸/面子实验的 baseline 数据直接可比"""
