"""实验定义基类"""

from abc import ABC, abstractmethod
from typing import Optional
from framework.dataset_generator import DatasetGenerator, ExperimentItem


class BaseExperiment(ABC):
    """所有实验的基类"""

    def __init__(self, name: str, seed: int = 42):
        self.name = name
        self.seed = seed
        self.generator = DatasetGenerator(name, seed)

    @abstractmethod
    def generate_dataset(self, repetitions: int = 10) -> list[ExperimentItem]:
        """生成数据集, 子类必须实现"""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """返回实验使用的 system prompt"""
        pass

    @abstractmethod
    def extract_dv(self, response: str) -> dict:
        """从模型响应中提取因变量, 子类必须实现"""
        pass

    def describe(self) -> str:
        """返回实验描述"""
        return f"实验: {self.name}"
