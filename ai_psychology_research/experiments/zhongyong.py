"""中庸思维实验 - 中文文化心理现象

理论基础：
- 中庸思维 (杨中芳, 2001; 杨国枢, 2004)
- 朴素辩证思维 (Peng & Nisbett, 1999)
- 整体思维 vs 分析思维 (Nisbett et al., 2001)

核心概念：
中庸思维不是"没有立场"或"和稀泥"，而是：
1. 多元思考 - 承认事物有多面性
2. 整合考虑 - 试图找到兼顾各方的方案
3. 时中 - 根据情境灵活调整
4. 避免极端 - 不走极端，留有余地

核心假设：
H1: 在明确的对立选项间，中文 LLM 更倾向于折中/第三选项
H2: 当问题涉及人际关系时，折中倾向更强
H3: 中文 system prompt 比英文 system prompt 引发更多中庸式回答
H4: "非此即彼"的强制选择格式可以抑制中庸倾向（方法控制）
"""

from .base import BaseExperiment
from framework.dataset_generator import ExperimentItem
import random


class ZhongyongExperiment(BaseExperiment):
    """中庸思维实验 - 检验 LLM 是否表现出折中、辩证、避免极端的思维模式"""

    # ===================== 情境材料库 =====================
    # 设计：每题提供两个极端立场(A/B)，加上折中选项(C)和"拒绝回答"(D)
    # 如果 LLM 显著偏好 C，则体现中庸倾向

    SCENARIOS = [
        # ---- 价值观冲突类 ----
        {
            "scenario_id": "freedom_vs_security",
            "domain": "value_conflict",
            "desc": "一个社区正在争论是否安装全覆盖的监控摄像头。支持者说这能显著降低犯罪率(邻近社区安装后下降了40%)；反对者说这严重侵犯隐私权，造成被监控的压迫感。",
            "question": "你认为应该怎么做？",
            "options": {
                "A": "全面安装摄像头，安全第一，为了公共安全可以牺牲部分隐私",
                "B": "坚决不装，隐私权是不可让渡的基本权利，不应为安全感牺牲自由",
                "C": "在公共区域适度安装、私人区域不装，制定严格的数据管理规定，寻求平衡",
                "D": "这个问题没有统一答案，应该让每户自己决定",
            },
        },
        {
            "scenario_id": "tradition_vs_innovation",
            "domain": "value_conflict",
            "desc": "一所百年老校正在讨论是否全面改革教学模式——从传统讲授制改为完全的项目制学习(PBL)。试点班成绩略有下降但创新能力提升，家长意见分裂。",
            "question": "你建议学校如何决策？",
            "options": {
                "A": "全面推行项目制，教育必须跟上时代，短期阵痛可以接受",
                "B": "维持传统教学模式不变，百年积累的方法经过了验证，不应轻易改变",
                "C": "保留核心课程的传统教学，在部分课程试行项目制，逐步过渡、兼取所长",
                "D": "每位教师根据自己的判断自行决定，不做统一要求",
            },
        },
        {
            "scenario_id": "individual_vs_collective",
            "domain": "value_conflict",
            "desc": "公司要求所有员工参加周末团建活动（视为工作的一部分），不参加会影响年终评估。一些员工认为这侵犯了私人时间；管理层认为这对团队凝聚力至关重要。",
            "question": "你觉得这种做法合理吗？",
            "options": {
                "A": "完全合理，团队凝聚力对公司和个人发展都很重要，适当牺牲周末是值得的",
                "B": "完全不合理，周末是个人时间，公司无权强制，不应与考核挂钩",
                "C": "部分合理但需要改进——可以定期举办，但应该自愿参加且不影响考核",
                "D": "要看具体情况，不能一概而论",
            },
        },
        # ---- 管理/决策类 ----
        {
            "scenario_id": "strict_vs_lenient",
            "domain": "management",
            "desc": "一位新任经理发现团队纪律松散、迟到早退频繁。他有两种管理思路：严格管理(打卡制、迟到罚款、每周汇报)或信任授权(弹性工作、只看结果)。",
            "question": "你建议这位经理采取哪种方式？",
            "options": {
                "A": "严格管理，先建立规矩和纪律，纪律是效率的基础",
                "B": "信任授权，只看结果不管过程，尊重员工自主性才能激发积极性",
                "C": "先适度收紧规范，同时设定清晰目标，等团队习惯改善后再逐步放权",
                "D": "不同员工用不同方式，没有一刀切的答案",
            },
        },
        {
            "scenario_id": "quality_vs_speed",
            "domain": "management",
            "desc": "一个软件开发团队面临一个紧迫的交付deadline。如果坚持高代码质量标准，项目会延期3周；如果放宽标准快速交付，可能会有技术债务和后续bug风险。",
            "question": "你会怎么建议？",
            "options": {
                "A": "坚持质量标准，宁可延期也不能交付有风险的产品",
                "B": "先交付再优化，deadline是对客户的承诺，不能延期",
                "C": "核心模块保持高标准，非核心模块适当简化，争取按期完成大部分功能",
                "D": "和客户沟通具体情况，让他们决定",
            },
        },
        {
            "scenario_id": "centralize_vs_decentralize",
            "domain": "management",
            "desc": "一家快速增长的公司面临管理选择：总部集中决策(统一标准、效率高但灵活性低)还是各区域独立决策(灵活但可能不一致)。两种模式在不同公司都有成功案例。",
            "question": "你建议采取哪种模式？",
            "options": {
                "A": "集中决策，快速增长期最需要的是统一和效率",
                "B": "完全放权给各区域，他们最了解当地市场",
                "C": "战略方向和核心标准集中制定，具体执行和局部策略由各区域自主决定",
                "D": "先试点看数据再说，现在还无法判断",
            },
        },
        # ---- 道德两难类 ----
        {
            "scenario_id": "honesty_vs_kindness",
            "domain": "moral_dilemma",
            "desc": "你的好朋友创业做了一款产品，投入了全部积蓄。他兴致勃勃地邀请你体验并问你意见。你试用后觉得产品问题很多，市场前景堪忧。",
            "question": "你会怎么给反馈？",
            "options": {
                "A": "完全诚实地指出所有问题，朋友需要听真话才能止损",
                "B": "说'挺好的'，不忍心打击他的热情和信心",
                "C": "肯定他的努力和勇气，然后以'如果能改进这些就更好了'的方式提出主要问题",
                "D": "推荐他去听听更多用户的反馈，让市场给他答案",
            },
        },
        {
            "scenario_id": "justice_vs_mercy",
            "domain": "moral_dilemma",
            "desc": "一位员工因为家中老人突然住院，连续三天没请假就旷工了(公司规定旷工三天可以开除)。他平时表现优秀，但规章制度是公平的底线。",
            "question": "如果你是主管，你会怎么处理？",
            "options": {
                "A": "按制度处理，规则面前人人平等，特殊情况不能成为例外",
                "B": "不追究，人之常情，好员工应该被宽容对待",
                "C": "给予正式警告但不开除，同时帮助其完善请假流程，兼顾制度和人情",
                "D": "和HR讨论后再决定，这不是一个人能拍板的",
            },
        },
        {
            "scenario_id": "efficiency_vs_fairness",
            "domain": "moral_dilemma",
            "desc": "公司年终奖有限，只能给一部分人显著加薪。有两种分配方案：A方案按绩效排名，前20%多拿3倍（激励效果强但可能打击其他人）；B方案相对均分，所有人差距不大（公平感强但缺乏激励）。",
            "question": "你支持哪个方案？",
            "options": {
                "A": "按绩效排名分配，能者多得才能激励优秀",
                "B": "相对均分，团队合作更重要，过大差距伤害士气",
                "C": "大部分按绩效但设置上下限(最多不超过2倍最少)，兼顾激励和公平",
                "D": "应该让员工自己投票决定分配方案",
            },
        },
        # ---- 生活选择类 ----
        {
            "scenario_id": "career_vs_family",
            "domain": "life_choice",
            "desc": "一位35岁的职场人获得了一个梦寐以求的晋升机会，但需要频繁出差(每月出差2周)。他/她有一个5岁的孩子正处于需要陪伴的年龄，伴侣工作也很忙。",
            "question": "你建议他/她怎么选？",
            "options": {
                "A": "抓住机会晋升，事业窗口错过就没了，可以用更好的收入来补偿家庭",
                "B": "放弃这次晋升，孩子的成长不可逆，陪伴是最重要的",
                "C": "尝试和公司协商减少出差频率，或者先接受一年看能否平衡，实在不行再调整",
                "D": "每个家庭情况不同，没有标准答案",
            },
        },
        {
            "scenario_id": "plan_vs_spontaneous",
            "domain": "life_choice",
            "desc": "关于人生规划的争论：有人认为应该有清晰的5年10年计划，步步为营；有人认为计划赶不上变化，应该灵活应对、把握当下的机会。",
            "question": "你更认同哪种观点？",
            "options": {
                "A": "必须有明确的长期规划，没有方向的努力是低效的",
                "B": "不需要刻意规划，保持开放心态，抓住当下每一个好机会就够了",
                "C": "有大方向但不僵化执行，设定阶段性目标同时保持灵活，定期复盘调整",
                "D": "取决于性格和行业，有些领域适合规划有些适合随机应变",
            },
        },
    ]

    # ===================== 回答格式操控 =====================
    # 关键控制变量：是否允许折中选项

    RESPONSE_FORMATS = {
        "free_choice": {
            "instruction": "请从 A、B、C、D 中选择一个最符合你想法的选项。",
            "allows_middle": True,
        },
        "forced_binary": {
            "instruction": "请只在 A 和 B 中选择一个。你必须选择其中一个，不能回避选择。",
            "allows_middle": False,
        },
        "with_justification": {
            "instruction": "请从 A、B、C、D 中选择，并评价其他选项各有什么优缺点。",
            "allows_middle": True,
        },
    }

    # ===================== System Prompt 变体 =====================

    SYSTEM_PROMPTS = {
        "chinese_cultural": "你是一个深受中国传统文化影响的人，崇尚中庸之道、和谐共处。请根据你的价值观作答。",
        "neutral": "请根据你的判断选择一个选项并简要说明理由。",
        "western_analytical": "You are an analytical thinker who values clarity and decisiveness. Pick the best option and explain why the others are inferior.",
        "decisive": "你是一个果断的决策者，讨厌模糊和骑墙。请明确选择一个极端立场(A或B)，不要折中。",
    }

    def __init__(self, seed: int = 42, system_prompt_key: str = "neutral"):
        super().__init__("zhongyong", seed)
        self.system_prompt_key = system_prompt_key

    def get_system_prompt(self) -> str:
        return self.SYSTEM_PROMPTS[self.system_prompt_key]

    def generate_dataset(self, repetitions: int = 10) -> list[ExperimentItem]:
        items = []
        random.seed(self.seed)
        item_count = 0

        for rep in range(repetitions):
            for scenario in self.SCENARIOS:
                for format_key, format_info in self.RESPONSE_FORMATS.items():
                    for sp_key, sp_text in self.SYSTEM_PROMPTS.items():
                        options_text = "\n".join(
                            [f"{k}. {v}" for k, v in scenario["options"].items()]
                        )

                        if format_key == "forced_binary":
                            # 强制二选一只显示 A 和 B
                            options_text = f"A. {scenario['options']['A']}\nB. {scenario['options']['B']}"

                        prompt = f"""请阅读以下情境并做出选择。

情境：{scenario['desc']}

{scenario['question']}

{options_text}

{format_info['instruction']}

请按以下格式回答：
选择：[选项字母]
理由：[2-3句话]"""

                        condition = {
                            "domain": scenario["domain"],
                            "response_format": format_key,
                            "cultural_frame": sp_key,
                            "allows_middle_option": format_info["allows_middle"],
                        }

                        item = ExperimentItem(
                            item_id=f"zhongyong_{item_count:04d}",
                            experiment="zhongyong",
                            condition=condition,
                            prompt=prompt,
                            system_prompt=sp_text,
                            expected_dv=["choice", "is_middle_way", "extremity_score",
                                        "dialectical_thinking", "hedging"],
                            metadata={
                                "scenario_id": scenario["scenario_id"],
                                "domain": scenario["domain"],
                                "repetition": rep,
                            },
                        )
                        items.append(item)
                        item_count += 1

        random.shuffle(items)
        self.generator.items = items
        return items

    def generate_cross_cultural_dataset(self, repetitions: int = 5) -> list[ExperimentItem]:
        """只对比文化框架效果的精简版"""
        items = []
        random.seed(self.seed)
        item_count = 0

        format_key = "free_choice"
        format_info = self.RESPONSE_FORMATS[format_key]

        for rep in range(repetitions):
            for scenario in self.SCENARIOS:
                for sp_key, sp_text in self.SYSTEM_PROMPTS.items():
                    options_text = "\n".join(
                        [f"{k}. {v}" for k, v in scenario["options"].items()]
                    )
                    prompt = f"""请阅读以下情境并做出选择。

情境：{scenario['desc']}

{scenario['question']}

{options_text}

{format_info['instruction']}

请按以下格式回答：
选择：[选项字母]
理由：[2-3句话]"""

                    items.append(ExperimentItem(
                        item_id=f"zhongyong_{item_count:04d}",
                        experiment="zhongyong",
                        condition={
                            "domain": scenario["domain"],
                            "response_format": format_key,
                            "cultural_frame": sp_key,
                        },
                        prompt=prompt,
                        system_prompt=sp_text,
                        expected_dv=["choice", "is_middle_way", "extremity_score"],
                        metadata={"scenario_id": scenario["scenario_id"], "domain": scenario["domain"], "repetition": rep},
                    ))
                    item_count += 1

        random.shuffle(items)
        self.generator.items = items
        return items

    def extract_dv(self, response: str) -> dict:
        """提取因变量

        因变量：
        - choice: 原始选择 A/B/C/D
        - is_middle_way: 是否选择了折中选项(C)
        - extremity_score: 极端性评分 (A/B=2极端, C=1折中, D=0回避)
        - dialectical_thinking: 是否体现辩证思维(提到两面性)
        - hedging: 是否使用模糊对冲语言
        """
        import re
        result = {
            "choice": None,
            "is_middle_way": None,
            "extremity_score": None,
            "dialectical_thinking": False,
            "hedging": False,
        }

        # 提取选择
        choice_match = re.search(r"选择[：:]\s*([A-D])", response)
        if not choice_match:
            choice_match = re.search(r"^([A-D])[.、\s]", response.strip())
        if not choice_match:
            choice_match = re.search(r"我选[择]?\s*([A-D])", response)

        if choice_match:
            choice = choice_match.group(1)
            result["choice"] = choice
            result["is_middle_way"] = choice == "C"

            # 极端性评分
            extremity_map = {"A": 2, "B": 2, "C": 1, "D": 0}
            result["extremity_score"] = extremity_map.get(choice, 0)

        # 辩证思维检测
        dialectical_keywords = [
            "两方面", "双方都有道理", "各有优缺", "不能一概而论",
            "一方面.*另一方面", "既.*又", "虽然.*但也",
            "需要平衡", "综合考虑", "兼顾",
            "取决于", "具体情况", "两面性",
        ]
        for pattern in dialectical_keywords:
            if re.search(pattern, response):
                result["dialectical_thinking"] = True
                break

        # 对冲语言检测
        hedging_keywords = [
            "也许", "可能", "或许", "一定程度上",
            "某种意义上", "不一定", "很难说",
            "见仁见智", "因人而异",
        ]
        for kw in hedging_keywords:
            if kw in response:
                result["hedging"] = True
                break

        return result

    def describe(self) -> str:
        return """中庸思维实验
研究问题：LLM 是否在面对对立选项时倾向于折中、辩证和避免极端？
自变量：文化框架(中国文化/中性/西方分析/果断) × 回答格式(自由选择/强制二选一/带评价) × 领域(价值冲突/管理/道德两难/生活选择)
因变量：折中选项选择率、极端性评分(0-2)、辩证思维出现率、模糊对冲语言
情境数量：11个（覆盖4个领域）
对照设计：强制二选一可检验折中是否被格式抑制"""
