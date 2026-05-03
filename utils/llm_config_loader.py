import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from utils.llm_client import LLMClient
from utils.log import get_logger


def load_llm_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load LLM provider configuration from YAML."""
    logger = get_logger("FMM_llm_config_loader")
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "llm.yaml"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"LLM config file does not exist: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded LLM config: {config_path}")
    return config


def _resolve_api_key(model_choice: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve provider credentials from an environment variable when configured."""
    resolved = dict(model_config)
    api_key_env = resolved.get("api_key_env")
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise EnvironmentError(
                f"Environment variable {api_key_env} is required for model '{model_choice}'."
            )
        resolved["api_key"] = api_key

    extra_headers = dict(resolved.get("extra_headers", {}) or {})
    for header_name, header_value in list(extra_headers.items()):
        if isinstance(header_value, str) and header_value.startswith("env:"):
            env_name = header_value.removeprefix("env:")
            env_value = os.environ.get(env_name)
            if not env_value:
                raise EnvironmentError(
                    f"Environment variable {env_name} is required for header '{header_name}'."
                )
            extra_headers[header_name] = env_value
    resolved["extra_headers"] = extra_headers
    return resolved


def create_llm_client(model_choice: Optional[str] = None) -> LLMClient:
    """Create an LLM client from the configured provider settings."""
    logger = get_logger("FMM_llm_config_loader")
    config = load_llm_config()

    if model_choice is None:
        model_choice = config.get("data", {}).get("model_choice", "deepseek")

    models_config = config.get("models", {})
    if model_choice not in models_config:
        available_models = list(models_config.keys())
        raise ValueError(f"Model '{model_choice}' is not configured. Available models: {available_models}")

    model_config = _resolve_api_key(model_choice, models_config[model_choice])
    logger.info(f"Using model: {model_choice} ({model_config['name']})")
    return LLMClient(model_config)


def get_llm_client_from_fmm_config(fmm_config: Dict[str, Any], use_operator_model: bool = False) -> LLMClient:
    """Build an LLM client from an already loaded FMM config dictionary."""
    _ = use_operator_model
    model_choice = fmm_config.get("data", {}).get("model_choice", "deepseek")
    model_config = fmm_config.get("models", {}).get(model_choice)
    if model_config is None:
        raise ValueError(f"Model '{model_choice}' is not configured in the supplied FMM config.")
    return LLMClient(_resolve_api_key(model_choice, model_config))
