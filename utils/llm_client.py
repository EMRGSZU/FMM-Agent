import http.client
import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

from utils.log import get_logger


class LLMClient:
    """Small provider adapter for the LLM APIs used by FMM-Agent."""

    def __init__(self, config: Dict[str, Any]):
        self.model_config = config
        self.model_name = config.get("name", "deepseek-chat")
        self.response_format = config.get("response_format", "openai")
        self.logger = get_logger("FMM_enhanced_llm_client")
        self.logger.info(f"Initialized LLM client for model: {self.model_name}")

    def call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call the configured model and return response text."""
        if max_tokens is None:
            max_tokens = self.model_config.get("max_output_tokens", 4000)

        if self.model_config.get("client_type") == "openai" or self.response_format == "openai":
            return self._call_openai_compatible(messages, temperature, max_tokens)
        if self.response_format == "claude":
            return self._call_claude_compatible(messages, temperature, max_tokens)

        raise ValueError(f"Unsupported response format: {self.response_format}")

    def _call_openai_compatible(
        self, messages: List[Dict[str, str]], temperature: float, max_tokens: int
    ) -> str:
        api_base = self.model_config["api_base"]
        if self.model_config.get("client_type") == "openai":
            client = OpenAI(
                api_key=self.model_config["api_key"],
                base_url=api_base,
                timeout=self.model_config.get("timeout", 30),
                max_retries=self.model_config.get("max_retries", 3),
            )
            response = client.chat.completions.create(
                model=self.model_config["name"],
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            return response.choices[0].message.content

        host = api_base.replace("https://", "").replace("http://", "")
        conn = http.client.HTTPSConnection(host)
        payload_data = {
            "model": self.model_config["name"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        payload_data.update(self.model_config.get("extra_params", {}) or {})
        payload = json.dumps(payload_data)
        headers = {
            "Authorization": f"Bearer {self.model_config['api_key']}",
            "Content-Type": "application/json",
        }
        conn.request("POST", self.model_config.get("api_endpoint", "/v1/chat/completions"), payload, headers)
        response_json = json.loads(conn.getresponse().read().decode("utf-8"))
        return response_json["choices"][0]["message"]["content"]

    def _call_claude_compatible(
        self, messages: List[Dict[str, str]], temperature: float, max_tokens: int
    ) -> str:
        host = self.model_config.get("api_base", "https://oa.api2d.net").replace("https://", "")
        conn = http.client.HTTPSConnection(host)
        payload = json.dumps(
            {
                "model": self.model_config["name"],
                "messages": messages,
                "stream": self.model_config.get("stream", False),
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )

        headers = {
            "Authorization": f"Bearer {self.model_config['api_key']}",
            "Content-Type": "application/json",
        }
        headers.update(self.model_config.get("extra_headers", {}) or {})

        conn.request("POST", self.model_config.get("api_endpoint", "/claude/v1/messages"), payload, headers)
        response_json = json.loads(conn.getresponse().read().decode("utf-8"))
        return response_json["content"][0]["text"]

    def call_with_system_prompt(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call the LLM with a system and user prompt pair."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.call_llm(messages, temperature, max_tokens)


def create_llm_client(config: Dict[str, Any], model_name: str = "deepseek") -> LLMClient:
    """Backward-compatible factory wrapper."""
    _ = model_name
    return LLMClient(config)
