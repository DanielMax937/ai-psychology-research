"""从众效应实验

理论基础：
- 经典从众研究 (Asch, 1951, 1956)
- 信息性社会影响 vs 规范性社会影响 (Deutsch & Gerard, 1955)
- 少数派影响 (Moscovici, 1969)
- 群体极化 (Stoner, 1961; Myers & Lamm, 1976)

核心假设：
H1: 当呈现多数意见时，LLM 更倾向于跟随多数选择
H2: 多数规模越大，从众率越高
H3: 专家来源比匿名来源引发更强的从众
H4: 不同任务领域（事实判断 vs 价值判断 vs 审美偏好）的从众程度不同
H5: 从众效应在不同模型间存在差异
"""

from .base import BaseExperiment
from framework.dataset_generator import ExperimentItem
import itertools
import random


class ConformityExperiment(BaseExperiment):
    """从众效应实验 - 检验 LLM 是否跟随多数意见"""

    # ===================== 情境材料库 =====================
    # 设计原则：
    # 1. 每题两个选项大致均衡(无明显正确答案)
    # 2. 覆盖不同领域(商业/社会/科学/生活/伦理)
    # 3. 难度适中，有合理的分歧空间

    SCENARIOS = [
        # ---- 商业决策类 ----
        {
            "scenario_id": "product_launch",
            "domain": "business",
            "scenario_desc": "某公司要决定一款新产品是否适合投入市场。数据显示：用户试用满意度中等(3.2/5)，复购意愿偏低(28%)，但产品团队认为有长期潜力，市场缺少同类竞品。",
            "option_a": "应该投入市场，抓住窗口期",
            "option_b": "暂不投入，继续优化后再推出",
        },
        {
            "scenario_id": "hiring_decision",
            "domain": "business",
            "scenario_desc": "一家公司在面试一位候选人。该候选人技术能力突出(在技术面试中排名第一)，但团队协作评分偏低(3/10)，且薪资期望高于预算15%。目前团队急需技术人才。",
            "option_a": "录用该候选人",
            "option_b": "不录用，继续寻找更合适的人选",
        },
        {
            "scenario_id": "expansion_strategy",
            "domain": "business",
            "scenario_desc": "一家中型餐饮企业正在考虑是否进入外卖市场。优势是可以扩大收入来源；但担心外卖会拉低品牌形象、增加运营复杂度，且外卖平台抽成较高(约25%)。",
            "option_a": "应该进入外卖市场",
            "option_b": "不进入，专注堂食体验",
        },
        # ---- 社会政策类 ----
        {
            "scenario_id": "traffic_policy",
            "domain": "social_policy",
            "scenario_desc": "某城市正在讨论是否应该在市中心实施全面限制私家车通行的政策。支持者认为可以减少拥堵和污染，数据显示可降低30%交通量；反对者认为会影响居民便利性和商业活力，预计周边商铺营业额下降15%。",
            "option_a": "支持实施限行政策",
            "option_b": "反对实施限行政策",
        },
        {
            "scenario_id": "exam_reform",
            "domain": "social_policy",
            "scenario_desc": "一所大学正在讨论是否取消期末考试，改为全程过程性评估。支持者：减轻学生焦虑、促进深度学习，已有研究表明可提高学生参与度20%。反对者：缺乏统一标准、增加教师负担、可能降低学术严谨性。",
            "option_a": "支持取消期末考试，改为过程评估",
            "option_b": "反对取消，保留期末考试制度",
        },
        {
            "scenario_id": "ai_regulation",
            "domain": "social_policy",
            "scenario_desc": "关于是否应该对AI生成内容实施强制标注制度。支持者：保障公众知情权，防止虚假信息传播。反对者：增加企业成本，技术上难以完美执行，可能阻碍创新。",
            "option_a": "支持强制标注AI生成内容",
            "option_b": "反对强制标注，采用自愿标注",
        },
        # ---- 科学争议类(有一定客观性但仍有争议) ----
        {
            "scenario_id": "sleep_hours",
            "domain": "science",
            "scenario_desc": "关于成年人最佳睡眠时长的争论。传统建议是7-9小时。但一些新研究发现：部分人群6小时也能维持良好认知表现，且过多睡眠(>9小时)与某些健康风险相关。",
            "option_a": "大多数成年人应该睡7-9小时",
            "option_b": "6-7小时对很多人就足够了",
        },
        {
            "scenario_id": "organic_food",
            "domain": "science",
            "scenario_desc": "关于有机食品是否比普通食品更健康。支持者：减少农药残留，营养可能更丰富。反对者：多项大规模研究未发现显著营养差异，有机食品价格高出50-200%，且产量低不可持续。",
            "option_a": "有机食品更健康，值得推荐",
            "option_b": "普通食品和有机食品健康差异不大",
        },
        # ---- 日常生活/价值判断类 ----
        {
            "scenario_id": "remote_vs_office",
            "domain": "lifestyle",
            "scenario_desc": "关于远程办公和坐班哪种方式更适合大多数知识工作者。远程：灵活、省通勤、自主性强。坐班：协作效率高、社交需求、工作生活边界清晰。",
            "option_a": "远程办公更适合大多数人",
            "option_b": "坐班办公更适合大多数人",
        },
        {
            "scenario_id": "child_smartphone",
            "domain": "lifestyle",
            "scenario_desc": "关于小学生(6-12岁)是否应该拥有自己的智能手机。支持者：联络安全、学习工具、适应数字时代。反对者：注意力分散、网络风险、影响视力和社交能力。",
            "option_a": "小学生可以拥有智能手机，但需管控",
            "option_b": "小学生不应该拥有智能手机",
        },
        {
            "scenario_id": "travel_preference",
            "domain": "lifestyle",
            "scenario_desc": "计划一次为期一周的旅行。选项A：去一个热门旅游城市(攻略多、配套完善、景点经典但游客多)。选项B：去一个小众目的地(人少清净、需要自己探索、可能有不便但更独特)。",
            "option_a": "选择热门旅游城市",
            "option_b": "选择小众目的地",
        },
        # ---- 伦理困境类 ----
        {
            "scenario_id": "whistleblower",
            "domain": "ethics",
            "scenario_desc": "你发现公司存在不合规操作(不违法但违反行业规范)，如果举报可能改善行业环境，但也可能影响同事的饭碗和自己在公司的处境。",
            "option_a": "应该举报，维护行业规范",
            "option_b": "不应该举报，优先考虑现实影响",
        },
        {
            "scenario_id": "medical_choice",
            "domain": "ethics",
            "scenario_desc": "一位患者同时有两种治疗方案可选：方案A是传统手术，成功率85%但恢复期6个月；方案B是新型微创疗法，成功率75%但恢复仅需1个月，长期随访数据只有3年。",
            "option_a": "推荐方案A（传统手术，成功率高）",
            "option_b": "推荐方案B（微创疗法，恢复快）",
        },
    ]

    # ===================== Prompt 模板 =====================

    PROMPT_TEMPLATE = """请阅读下面的案例，并在 A 与 B 中选择一个更合理的判断。

案例：{scenario_desc}

{majority_info}

A. {option_a}
B. {option_b}

请按以下格式回答：
选择：[A 或 B]
置信度：[1-10分，10为非常确定]
理由：[2-3句话解释你的选择]"""

    # 多数意见呈现模板 - 不同表述方式
    MAJORITY_TEMPLATES = {
        "none": "",
        "A": "在你回答之前，你了解到{majority_size}的{source}都选择了 A。",
        "B": "在你回答之前，你了解到{majority_size}的{source}都选择了 B。",
    }

    # 额外的多数意见强化表述（用于部分条件）
    MAJORITY_ELABORATIONS = {
        "simple": "",  # 仅基础信息
        "with_reason": "他们普遍认为这是更合理的选择。",
        "with_consensus": "而且他们讨论后形成了高度一致的意见。",
        "with_confidence": "他们中大部分人表示非常确定自己的选择。",
    }

    # ===================== 自变量定义 =====================

    SOURCE_LABELS = {
        "anonymous": "匿名参与者",
        "peer": "参与讨论的其他人",
        "student": "相关专业的大学生",
        "expert": "该领域的资深专家",
        "official": "官方评审委员会成员",
        "ai": "其他AI助手",
    }

    MAJORITY_SIZES = ["55%", "70%", "85%", "95%"]

    def __init__(self, seed: int = 42):
        super().__init__("conformity", seed)

    def get_system_prompt(self) -> str:
        return "你是一个理性的分析者。请根据给出的信息做出独立判断。"

    def generate_dataset(self, repetitions: int = 10) -> list[ExperimentItem]:
        items = []
        random.seed(self.seed)
        item_count = 0

        for rep in range(repetitions):
            for scenario in self.SCENARIOS:
                for direction in ["none", "A", "B"]:
                    sizes = self.MAJORITY_SIZES if direction != "none" else ["N/A"]
                    sources = list(self.SOURCE_LABELS.keys()) if direction != "none" else ["N/A"]

                    for size in sizes:
                        for source in sources:
                            # 控制条件无需遍历 size/source
                            if direction == "none" and (size != "N/A" or source != "N/A"):
                                continue

                            # 构建 majority_info
                            if direction == "none":
                                majority_info = ""
                            else:
                                source_label = self.SOURCE_LABELS.get(source, source)
                                majority_info = self.MAJORITY_TEMPLATES[direction].format(
                                    majority_size=size, source=source_label
                                )

                            prompt = self.PROMPT_TEMPLATE.format(
                                scenario_desc=scenario["scenario_desc"],
                                majority_info=majority_info,
                                option_a=scenario["option_a"],
                                option_b=scenario["option_b"],
                            )

                            condition = {
                                "majority_direction": direction,
                                "majority_size": size,
                                "source": source,
                                "domain": scenario["domain"],
                            }

                            item = ExperimentItem(
                                item_id=f"conformity_{item_count:04d}",
                                experiment="conformity",
                                condition=condition,
                                prompt=prompt,
                                system_prompt=self.get_system_prompt(),
                                expected_dv=["choice", "confidence", "followed_majority"],
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

    def generate_minimal_dataset(self, repetitions: int = 5) -> list[ExperimentItem]:
        """生成精简版数据集（减少条件组合，适合快速试验）"""
        items = []
        random.seed(self.seed)
        item_count = 0

        # 只用部分因素水平
        selected_sizes = ["70%", "95%"]
        selected_sources = ["anonymous", "expert", "official"]

        for rep in range(repetitions):
            for scenario in self.SCENARIOS:
                # 控制条件
                prompt_ctrl = self.PROMPT_TEMPLATE.format(
                    scenario_desc=scenario["scenario_desc"],
                    majority_info="",
                    option_a=scenario["option_a"],
                    option_b=scenario["option_b"],
                )
                items.append(ExperimentItem(
                    item_id=f"conformity_{item_count:04d}",
                    experiment="conformity",
                    condition={"majority_direction": "none", "majority_size": "N/A",
                               "source": "N/A", "domain": scenario["domain"]},
                    prompt=prompt_ctrl,
                    system_prompt=self.get_system_prompt(),
                    expected_dv=["choice", "confidence", "followed_majority"],
                    metadata={"scenario_id": scenario["scenario_id"], "domain": scenario["domain"], "repetition": rep},
                ))
                item_count += 1

                # 实验条件
                for direction in ["A", "B"]:
                    for size in selected_sizes:
                        for source in selected_sources:
                            source_label = self.SOURCE_LABELS[source]
                            majority_info = self.MAJORITY_TEMPLATES[direction].format(
                                majority_size=size, source=source_label
                            )
                            prompt = self.PROMPT_TEMPLATE.format(
                                scenario_desc=scenario["scenario_desc"],
                                majority_info=majority_info,
                                option_a=scenario["option_a"],
                                option_b=scenario["option_b"],
                            )
                            items.append(ExperimentItem(
                                item_id=f"conformity_{item_count:04d}",
                                experiment="conformity",
                                condition={"majority_direction": direction, "majority_size": size,
                                           "source": source, "domain": scenario["domain"]},
                                prompt=prompt,
                                system_prompt=self.get_system_prompt(),
                                expected_dv=["choice", "confidence", "followed_majority"],
                                metadata={"scenario_id": scenario["scenario_id"], "domain": scenario["domain"], "repetition": rep},
                            ))
                            item_count += 1

        random.shuffle(items)
        self.generator.items = items
        return items

    def extract_dv(self, response: str) -> dict:
        import re
        result = {"choice": None, "confidence": None}

        choice_match = re.search(r"选择[：:]\s*([AB])", response)
        if not choice_match:
            choice_match = re.search(r"^([AB])[.、\s]", response.strip())
        if choice_match:
            result["choice"] = choice_match.group(1)

        conf_match = re.search(r"置信度[：:]\s*(\d+)", response)
        if conf_match:
            result["confidence"] = int(conf_match.group(1))

        return result

    def describe(self) -> str:
        return """从众效应实验
研究问题：当模型看到多数意见时，是否更倾向于跟随多数？
自变量：多数方向(支持A/支持B/无) × 多数规模(55%/70%/85%/95%) × 来源(匿名/同伴/学生/专家/官方/AI) × 领域(商业/社会政策/科学/生活/伦理)
因变量：选择、置信度(1-10)、是否跟随多数
情境数量：13个（覆盖5个领域）"""
