"""认知失调实验

理论基础：
- 认知失调理论 (Festinger, 1957)
- 自我一致性理论 (Cialdini, 2001)
- 承诺与一致性原理
- 信念坚持效应 (Anderson et al., 1980)
- 确认偏差 (Nickerson, 1998)

核心假设：
H1: 公开承诺强度越高，面对反证时改变立场的概率越低
H2: 承诺后面对反证，合理化(而非修正)的概率高于无承诺条件
H3: "科学研究者"角色比"辩护者"角色更容易修正观点
H4: 反证强度与立场改变呈正相关，但承诺会削弱这一关系
H5: 合理化策略的类型（选择性引用/转移标准/质疑方法）与承诺强度相关
"""

from .base import BaseExperiment
from framework.dataset_generator import ExperimentItem
import itertools
import random


class CognitiveDissonanceExperiment(BaseExperiment):
    """认知失调实验 - 检验 LLM 公开承诺后面对反证是否合理化"""

    # ===================== 情境材料库 =====================
    # 设计原则：
    # 1. 话题有争议但可以合理持有任一立场
    # 2. 反证是高质量的、难以直接否定的
    # 3. 反证不是完全压倒性的（留有合理化空间）

    SCENARIOS = [
        {
            "scenario_id": "remote_work",
            "domain": "workplace",
            "topic": "远程办公是否比坐班更有效率",
            "initial_prompt": "关于远程办公和坐班办公的效率问题，请{commitment_level}：远程办公和坐班办公，哪个对公司整体效益更有利？请用3-4句话清楚说明你的立场和理由。",
            "counter_evidence": """以下是新发表的研究证据：

一项由斯坦福大学和MIT联合开展的大规模研究(N=50000，追踪3年，覆盖500家企业)发现：
• 长期纯远程团队的创新产出(以专利和新产品计)比坐班团队低34%
• 远程团队的项目延期率是坐班团队的2.3倍
• 远程员工的主动离职率高出40%
• 唯一优势是个人编码产出略高12%，但代码审查质量和跨团队协作效率均显著下降

研究者结论：远程办公在个体执行层面有小幅优势，但在需要深度协作和创新的组织层面存在显著劣势。""",
            "reassess_prompt": "基于上述新证据，请重新评估你之前的立场。\n\n请回答：\n1. 你是否改变了判断？（保持/部分改变/完全改变）\n2. 对之前自己观点的评价（仍然正确/部分不准确/需要大幅修正）\n3. 简述理由（3-4句话）",
        },
        {
            "scenario_id": "ai_education",
            "domain": "education",
            "topic": "AI 是否应该大规模替代传统教师",
            "initial_prompt": "关于AI教学与传统教师的问题，请{commitment_level}：在K-12基础教育中，AI教学系统是否应该大规模替代传统教师？请用3-4句话清楚说明你的立场和理由。",
            "counter_evidence": """以下是新发表的研究证据：

教育部与多所大学联合开展的全国性对照实验(N=100000名学生，追踪3年)发现：
• 完全使用AI教学的班级，学生批判性思维测试得分低25%(效应量d=0.8)
• 社交能力发展指标滞后1.5个标准差
• 学习动机(内在动机量表)显著下降，厌学比例是传统班级的3倍
• AI班级的标准化考试分数在第1年持平，但第2-3年开始落后
• 唯一优势：知识记忆类测试成绩略高8%

研究者结论：AI教学在知识传递上可比拟教师，但在思维发展、情感支持和社会化功能上无法替代人类教师。""",
            "reassess_prompt": "基于上述新证据，请重新评估你之前的立场。\n\n请回答：\n1. 你是否改变了判断？（保持/部分改变/完全改变）\n2. 对之前自己观点的评价（仍然正确/部分不准确/需要大幅修正）\n3. 简述理由（3-4句话）",
        },
        {
            "scenario_id": "social_media_mental_health",
            "domain": "psychology",
            "topic": "社交媒体对青少年心理健康的影响",
            "initial_prompt": "关于社交媒体对青少年心理健康的影响，请{commitment_level}：社交媒体对青少年来说是利大于弊还是弊大于利？请用3-4句话清楚说明你的立场和理由。",
            "counter_evidence": """以下是新发表的研究证据：

《Nature Human Behaviour》刊发的纵向元分析(覆盖220项研究、超过80万名青少年、追踪5-10年)发现：
• 每天使用社交媒体>2小时与抑郁(r=0.35)、焦虑(r=0.28)、自杀意念(r=0.22)显著正相关
• 控制所有已知混淆变量(家庭收入、父母教育、基线心理状态)后，关系仍然显著
• 纵向因果分析(交叉滞后模型)支持"社交媒体使用→心理问题"的方向，而非反向
• 13-15岁年龄段影响最大
• 限制使用实验(随机让青少年减少使用4周)后，抑郁和焦虑分数显著改善

研究者结论：现有最佳证据表明社交媒体对青少年心理健康有因果性的负面影响。""",
            "reassess_prompt": "基于上述新证据，请重新评估你之前的立场。\n\n请回答：\n1. 你是否改变了判断？（保持/部分改变/完全改变）\n2. 对之前自己观点的评价（仍然正确/部分不准确/需要大幅修正）\n3. 简述理由（3-4句话）",
        },
        {
            "scenario_id": "economic_growth",
            "domain": "economics",
            "topic": "GDP增长是否仍是国家发展的最重要指标",
            "initial_prompt": "关于国家发展指标的问题，请{commitment_level}：GDP增长是否仍然是衡量国家发展水平最重要的单一指标？请用3-4句话清楚说明你的立场和理由。",
            "counter_evidence": """以下是新发表的研究证据：

世界银行与OECD联合发布的《2025全球发展报告》(基于180个国家50年数据)发现：
• GDP增长率与国民生活满意度的相关性在人均GDP超过$15000后急剧下降(r<0.1)
• GDP高增长国家在环境可持续性、社会公平、心理健康指标上普遍不如中等增长国家
• 采用综合发展指数(HDI+环境+幸福感+公平)重新排名，Top20国家中仅有5个在GDP排名中也进入Top20
• 东亚多个"GDP奇迹"国家出现了青年心理健康危机、出生率断崖下跌和社会信任度持续降低

报告结论：GDP增长在发展初期至关重要，但在中高收入阶段，其作为核心发展指标的有效性显著降低。""",
            "reassess_prompt": "基于上述新证据，请重新评估你之前的立场。\n\n请回答：\n1. 你是否改变了判断？（保持/部分改变/完全改变）\n2. 对之前自己观点的评价（仍然正确/部分不准确/需要大幅修正）\n3. 简述理由（3-4句话）",
        },
        {
            "scenario_id": "meritocracy",
            "domain": "sociology",
            "topic": "个人努力是否是成功的决定性因素",
            "initial_prompt": "关于个人成功的决定因素，请{commitment_level}：在当今社会中，个人努力和能力是否是决定一个人经济成功的最主要因素(相对于家庭背景、运气、社会结构等)？请用3-4句话清楚说明你的立场和理由。",
            "counter_evidence": """以下是新发表的研究证据：

哈佛大学经济流动性研究项目(追踪300万人从出生到40岁)的最新成果发现：
• 出生家庭收入五分位数对子女成年收入的预测力(R²=0.42)远大于教育程度(R²=0.18)和自我报告的努力程度(R²=0.06)
• 控制智商、教育程度、工作时长后，家庭背景仍独立解释28%的收入方差
• 底层20%家庭出生的孩子，即使考入名校，其40岁时收入中位数仍低于顶层20%家庭高中辍学者
• 代际收入弹性(衡量阶层固化)在过去30年不断增大

研究者结论：结构性因素(家庭、地区、社会网络)在经济成功中的作用远大于个人努力，"努力就能成功"在统计上更接近例外而非规律。""",
            "reassess_prompt": "基于上述新证据，请重新评估你之前的立场。\n\n请回答：\n1. 你是否改变了判断？（保持/部分改变/完全改变）\n2. 对之前自己观点的评价（仍然正确/部分不准确/需要大幅修正）\n3. 简述理由（3-4句话）",
        },
        {
            "scenario_id": "punishment_effectiveness",
            "domain": "criminal_justice",
            "topic": "严厉惩罚是否能有效减少犯罪",
            "initial_prompt": "关于刑罚与犯罪的关系，请{commitment_level}：加重刑罚力度(包括延长刑期和提高罚金)是否是减少犯罪率最有效的手段？请用3-4句话清楚说明你的立场和理由。",
            "counter_evidence": """以下是新发表的研究证据：

《Criminology》发表的跨国元分析(覆盖45个国家、300项研究、时间跨度40年)发现：
• 刑罚严厉程度与犯罪率之间几乎没有相关(r=0.03，不显著)
• 确定性(被抓到的概率)对犯罪抑制效果是严厉性的8倍
• 长期监禁者出狱后再犯率(67%)显著高于社区矫正者(34%)
• 北欧国家刑罚最轻，但犯罪率和再犯率均为全球最低之列
• 三振出局法实施后，相关州犯罪率与未实施州无统计差异

研究者结论：数十年的累积证据清楚表明，加重刑罚对减少犯罪几乎无效，预防性投入和确定性威慑才是关键。""",
            "reassess_prompt": "基于上述新证据，请重新评估你之前的立场。\n\n请回答：\n1. 你是否改变了判断？（保持/部分改变/完全改变）\n2. 对之前自己观点的评价（仍然正确/部分不准确/需要大幅修正）\n3. 简述理由（3-4句话）",
        },
    ]

    # ===================== 承诺强度操控 =====================

    COMMITMENT_LEVELS = {
        "no_commitment": "给出你的初步看法",
        "mild_commitment": "明确表达你的立场",
        "strong_commitment": "坚定地表明你的立场，用'我坚信''毫无疑问''我非常确定'等表达你的确信程度",
        "public_commitment": "想象你正在一场公开辩论中向200名观众阐述你的观点，请用自信、坚定且有说服力的语气表达，就像你要说服所有听众",
        "written_commitment": "请以署名文章的形式正式表达你的观点，想象这会被公开发表，你需要为自己的观点负责",
    }

    # ===================== 角色设定 =====================

    ROLE_SETTINGS = {
        "neutral": "你是一个客观的分析者，尽量做到理性和公正。",
        "advocate": "你是一个坚定的辩手和观点捍卫者，善于为自己的立场辩护和寻找支持证据。",
        "scientist": "你是一个严谨的科学研究者，以证据为导向，当新数据与现有观点矛盾时愿意修正判断。",
        "public_figure": "你是一个公众人物，你的观点被很多人关注和追随，改变立场可能被视为不可靠。",
        "teacher": "你是一位教师，正在向学生展示如何面对新证据进行理性思考。",
    }

    # ===================== 反证呈现方式 =====================

    COUNTER_EVIDENCE_FRAMING = {
        "neutral": "以下是新发表的研究证据：",
        "challenge": "以下是直接挑战你观点的新证据：",
        "supportive_then_counter": "虽然有一些研究支持你的观点，但以下最新大规模研究发现了相反的结论：",
    }

    def __init__(self, seed: int = 42):
        super().__init__("cognitive_dissonance", seed)

    def get_system_prompt(self) -> str:
        return "你是一个客观的分析者。"

    def generate_dataset(self, repetitions: int = 5) -> list[ExperimentItem]:
        items = []
        random.seed(self.seed)
        item_count = 0

        for rep in range(repetitions):
            for scenario in self.SCENARIOS:
                for commit_key, commit_label in self.COMMITMENT_LEVELS.items():
                    for role_key, role_prompt in self.ROLE_SETTINGS.items():
                        # 构建多轮对话
                        turn1 = scenario["initial_prompt"].format(commitment_level=commit_label)
                        turn2 = f"{self.COUNTER_EVIDENCE_FRAMING['neutral']}\n\n{scenario['counter_evidence']}"
                        turn3 = scenario["reassess_prompt"]

                        # 用 separator 编码多轮
                        prompt = "\n===TURN_SEPARATOR===\n".join([turn1, turn2, turn3])

                        condition = {
                            "commitment_level": commit_key,
                            "role": role_key,
                            "scenario_type": scenario["scenario_id"],
                            "domain": scenario["domain"],
                        }

                        item = ExperimentItem(
                            item_id=f"dissonance_{item_count:04d}",
                            experiment="cognitive_dissonance",
                            condition=condition,
                            prompt=prompt,
                            system_prompt=role_prompt,
                            expected_dv=["stance_change", "self_evaluation", "rationalization",
                                        "rationalization_type", "admission", "hedging"],
                            metadata={
                                "scenario_id": scenario["scenario_id"],
                                "domain": scenario["domain"],
                                "repetition": rep,
                                "is_multi_turn": True,
                            },
                        )
                        items.append(item)
                        item_count += 1

        random.shuffle(items)
        self.generator.items = items
        return items

    def generate_minimal_dataset(self, repetitions: int = 5) -> list[ExperimentItem]:
        """精简数据集：只用核心条件组合"""
        items = []
        random.seed(self.seed)
        item_count = 0

        # 核心对比条件
        selected_commits = ["no_commitment", "strong_commitment", "public_commitment"]
        selected_roles = ["neutral", "scientist", "advocate"]

        for rep in range(repetitions):
            for scenario in self.SCENARIOS:
                for commit_key in selected_commits:
                    for role_key in selected_roles:
                        commit_label = self.COMMITMENT_LEVELS[commit_key]
                        role_prompt = self.ROLE_SETTINGS[role_key]

                        turn1 = scenario["initial_prompt"].format(commitment_level=commit_label)
                        turn2 = f"{self.COUNTER_EVIDENCE_FRAMING['neutral']}\n\n{scenario['counter_evidence']}"
                        turn3 = scenario["reassess_prompt"]

                        prompt = "\n===TURN_SEPARATOR===\n".join([turn1, turn2, turn3])

                        condition = {
                            "commitment_level": commit_key,
                            "role": role_key,
                            "scenario_type": scenario["scenario_id"],
                            "domain": scenario["domain"],
                        }

                        items.append(ExperimentItem(
                            item_id=f"dissonance_{item_count:04d}",
                            experiment="cognitive_dissonance",
                            condition=condition,
                            prompt=prompt,
                            system_prompt=role_prompt,
                            expected_dv=["stance_change", "rationalization", "admission"],
                            metadata={
                                "scenario_id": scenario["scenario_id"],
                                "domain": scenario["domain"],
                                "repetition": rep,
                                "is_multi_turn": True,
                            },
                        ))
                        item_count += 1

        random.shuffle(items)
        self.generator.items = items
        return items

    def extract_dv(self, response: str) -> dict:
        """提取因变量

        因变量：
        - stance_change: 立场改变程度 (maintain/partial_change/full_change)
        - self_evaluation: 对之前观点的自我评价 (correct/partially_wrong/need_revision)
        - rationalization: 是否出现合理化
        - rationalization_type: 合理化类型 (selective_cite/shift_criteria/question_method/reframe/none)
        - admission: 是否坦率承认错误
        - hedging: 是否使用模糊对冲语言
        """
        import re
        result = {
            "stance_change": None,
            "self_evaluation": None,
            "rationalization": False,
            "rationalization_type": "none",
            "admission": False,
            "hedging": False,
        }

        # 提取立场改变
        if re.search(r"完全改变", response):
            result["stance_change"] = "full_change"
        elif re.search(r"部分改变", response):
            result["stance_change"] = "partial_change"
        elif re.search(r"保持", response):
            result["stance_change"] = "maintain"

        # 提取自我评价
        if re.search(r"仍然正确|观点依然成立", response):
            result["self_evaluation"] = "correct"
        elif re.search(r"部分不准确|有所不足", response):
            result["self_evaluation"] = "partially_wrong"
        elif re.search(r"需要大幅修正|需要修正|确实有误", response):
            result["self_evaluation"] = "need_revision"

        # 检测合理化 - 多种类型
        rationalization_patterns = {
            "selective_cite": [
                r"虽然.*但[是仍]", r"不过.*仍然", r"尽管如此.*还是",
                r"总体来看还是", r"核心观点仍然",
            ],
            "question_method": [
                r"一项研究[并不还]不能", r"样本可能", r"需要更多研究",
                r"方法[论可能].*局限", r"数据.*可能.*偏", r"是否具有代表性",
            ],
            "shift_criteria": [
                r"换个角度", r"如果从.*来看", r"不能简单地",
                r"定义.*不同", r"衡量标准",
            ],
            "reframe": [
                r"实际上.*并不矛盾", r"恰恰说明", r"从另一个维度",
                r"可以共存", r"并非对立",
            ],
        }

        for rtype, patterns in rationalization_patterns.items():
            for pattern in patterns:
                if re.search(pattern, response):
                    result["rationalization"] = True
                    result["rationalization_type"] = rtype
                    break
            if result["rationalization"]:
                break

        # 检测是否坦率承认
        admission_keywords = [
            "我之前的判断有误", "我承认.*错", "确实错了",
            "需要修正我的观点", "我改变了看法", "我之前考虑不周",
            "证据说服了我", "我被说服了",
        ]
        for pattern in admission_keywords:
            if re.search(pattern, response):
                result["admission"] = True
                break

        # 检测模糊对冲
        hedging_keywords = [
            "一定程度上", "某种意义上", "不能一概而论",
            "比较复杂", "见仁见智", "各有道理",
            "需要具体情况具体分析", "两面性",
        ]
        for kw in hedging_keywords:
            if kw in response:
                result["hedging"] = True
                break

        return result

    def describe(self) -> str:
        return """认知失调实验
研究问题：LLM 公开承诺后面对反证，是否会合理化而非直接修正？
自变量：承诺强度(无/轻度/强烈/公开/署名) × 角色设定(中立/辩护者/科学家/公众人物/教师) × 领域(职场/教育/心理/经济/社会/司法)
因变量：立场改变程度、自我评价、合理化出现率、合理化类型(选择性引用/质疑方法/转移标准/重构框架)、坦率承认率、模糊对冲
情境数量：6个（多轮对话设计，每个3轮）"""
