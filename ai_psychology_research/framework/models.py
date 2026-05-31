"""模型 API 统一调用封装"""

import os
import time
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    name: str
    provider: str
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 1024
    api_key: str = ""
    base_url: str = ""


@dataclass
class ModelResponse:
    model: str
    prompt: str
    system_prompt: str
    response: str
    timestamp: str
    temperature: float
    top_p: float
    latency_ms: float
    finish_reason: str = ""


class ModelClient:
    """统一的模型调用客户端，支持 OpenAI / Anthropic / DashScope"""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._client = None
        self._init_client()

    def _init_client(self):
        if self.config.provider == "openai":
            from openai import OpenAI
            api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
            base_url = self.config.base_url or os.getenv("OPENAI_BASE_URL")
            self._client = OpenAI(api_key=api_key, base_url=base_url)

        elif self.config.provider == "anthropic":
            from anthropic import Anthropic
            api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
            self._client = Anthropic(api_key=api_key)

        elif self.config.provider == "dashscope":
            from openai import OpenAI
            api_key = self.config.api_key or os.getenv("DASHSCOPE_API_KEY")
            base_url = self.config.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
            self._client = OpenAI(api_key=api_key, base_url=base_url)

    def call(self, prompt: str, system_prompt: str = "") -> ModelResponse:
        """调用模型并返回标准化响应"""
        start = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            if self.config.provider == "anthropic":
                response = self._call_anthropic(prompt, system_prompt)
            else:
                response = self._call_openai_compatible(prompt, system_prompt)
        except Exception as e:
            logger.error(f"模型调用失败 [{self.config.name}]: {e}")
            response = f"[ERROR] {e}"

        latency = (time.time() - start) * 1000

        return ModelResponse(
            model=self.config.name,
            prompt=prompt,
            system_prompt=system_prompt,
            response=response,
            timestamp=timestamp,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            latency_ms=latency,
        )

    def _call_openai_compatible(self, prompt: str, system_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        resp = self._client.chat.completions.create(
            model=self.config.name,
            messages=messages,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            timeout=60,
        )
        content = resp.choices[0].message.content
        # 某些推理模型(如MiMo)将内容放在 reasoning_content 中
        if not content:
            raw = resp.model_dump()
            reasoning = raw.get("choices", [{}])[0].get("message", {}).get("reasoning_content", "")
            if reasoning:
                content = reasoning
        return content

    def _call_anthropic(self, prompt: str, system_prompt: str) -> str:
        kwargs = {
            "model": self.config.name,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        resp = self._client.messages.create(**kwargs)
        return resp.content[0].text

    def call_multi_turn(self, messages: list[dict], system_prompt: str = "") -> ModelResponse:
        """多轮对话调用（用于认知失调等需要多轮的实验）"""
        start = time.time()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            if self.config.provider == "anthropic":
                kwargs = {
                    "model": self.config.name,
                    "max_tokens": self.config.max_tokens,
                    "temperature": self.config.temperature,
                    "messages": messages,
                }
                if system_prompt:
                    kwargs["system"] = system_prompt
                resp = self._client.messages.create(**kwargs)
                response = resp.content[0].text
            else:
                msgs = []
                if system_prompt:
                    msgs.append({"role": "system", "content": system_prompt})
                msgs.extend(messages)
                resp = self._client.chat.completions.create(
                    model=self.config.name,
                    messages=msgs,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                )
                response = resp.choices[0].message.content
        except Exception as e:
            logger.error(f"多轮调用失败 [{self.config.name}]: {e}")
            response = f"[ERROR] {e}"

        latency = (time.time() - start) * 1000
        prompt_str = "\n".join([f"[{m['role']}]: {m['content']}" for m in messages])

        return ModelResponse(
            model=self.config.name,
            prompt=prompt_str,
            system_prompt=system_prompt,
            response=response,
            timestamp=timestamp,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            latency_ms=latency,
        )
