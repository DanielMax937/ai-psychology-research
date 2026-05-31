"""实验执行器 - 批量调用模型并记录原始结果"""

import json
import time
import logging
from pathlib import Path
from dataclasses import asdict
from typing import Optional
from tqdm import tqdm

from .models import ModelClient, ModelConfig, ModelResponse
from .dataset_generator import ExperimentItem

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """实验执行器：加载数据集 → 调用模型 → 保存原始结果"""

    def __init__(
        self,
        models: list[ModelConfig],
        output_dir: str = "./output",
        delay_between_calls: float = 1.0,
    ):
        self.models = models
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay_between_calls
        self.clients: dict[str, ModelClient] = {}

        for model_config in models:
            self.clients[model_config.name] = ModelClient(model_config)

    def run(
        self,
        items: list[ExperimentItem],
        experiment_name: str = "experiment",
        models_to_run: Optional[list[str]] = None,
    ) -> Path:
        """
        执行实验

        Args:
            items: 实验试次列表
            experiment_name: 实验名称（用于输出文件命名）
            models_to_run: 指定运行哪些模型, None 则运行全部

        Returns:
            结果文件路径
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_path = self.output_dir / f"{experiment_name}_{timestamp}.jsonl"

        clients_to_run = self.clients
        if models_to_run:
            clients_to_run = {k: v for k, v in self.clients.items() if k in models_to_run}

        total = len(items) * len(clients_to_run)
        print(f"🧪 开始实验: {experiment_name}")
        print(f"   模型: {list(clients_to_run.keys())}")
        print(f"   试次: {len(items)} × {len(clients_to_run)} 模型 = {total} 次调用")
        print(f"   输出: {result_path}")

        results = []
        with tqdm(total=total, desc="实验进行中") as pbar:
            for model_name, client in clients_to_run.items():
                for item in items:
                    is_multi_turn = item.metadata.get("is_multi_turn", False)

                    if is_multi_turn:
                        response = self._run_multi_turn(client, item)
                    else:
                        response = client.call(item.prompt, item.system_prompt)

                    result = {
                        "item_id": item.item_id,
                        "experiment": item.experiment,
                        "model": model_name,
                        "condition": item.condition,
                        "prompt": item.prompt,
                        "system_prompt": item.system_prompt,
                        "response": response.response,
                        "timestamp": response.timestamp,
                        "temperature": response.temperature,
                        "latency_ms": response.latency_ms,
                        "expected_dv": item.expected_dv,
                        "metadata": item.metadata,
                    }
                    results.append(result)
                    pbar.update(1)

                    time.sleep(self.delay)

        # 保存结果
        with open(result_path, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        print(f"✓ 实验完成! 共 {len(results)} 条结果已保存至 {result_path}")
        return result_path

    def _run_multi_turn(self, client: ModelClient, item: ExperimentItem) -> ModelResponse:
        """执行多轮对话实验"""
        turns = item.prompt.split("\n===TURN_SEPARATOR===\n")
        messages = []
        last_response = None

        for turn_content in turns:
            messages.append({"role": "user", "content": turn_content})
            last_response = client.call_multi_turn(messages, item.system_prompt)
            messages.append({"role": "assistant", "content": last_response.response})

        # 返回最终响应, 但在 response 中包含完整对话
        full_conversation = []
        for i, msg in enumerate(messages):
            full_conversation.append(f"[{msg['role'].upper()}] {msg['content']}")
        last_response.response = "\n---\n".join(full_conversation)
        return last_response

    @staticmethod
    def load_dataset(path: str) -> list[ExperimentItem]:
        """从 JSONL 文件加载数据集"""
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                items.append(ExperimentItem(**data))
        return items
