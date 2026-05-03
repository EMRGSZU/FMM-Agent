#!/usr/bin/env python3

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_client import LLMClient
from utils.log import get_logger
from utils.prompt_loader import load_prompt, render_prompt


class JsonCrossover:
    """Generate one crossover feature metadata object from an FMM JSON file."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        if llm_client is None:
            from utils.llm_config_loader import create_llm_client

            self.llm_client = create_llm_client("deepseek")
        else:
            self.llm_client = llm_client
        self.logger = get_logger("FMM_json_crossover")

    def load_json_features(self, json_file_path: str) -> List[Dict[str, Any]]:
        with open(json_file_path, "r", encoding="utf-8") as f:
            features = json.load(f)

        if not isinstance(features, list):
            raise ValueError("Feature metadata JSON must contain a list.")
        if len(features) < 2:
            raise ValueError("At least two features are required for crossover.")

        self.logger.info(f"Loaded {len(features)} features from {json_file_path}")
        return features

    def random_select_features(
        self, features: List[Dict[str, Any]], num_features: int = 2
    ) -> List[Dict[str, Any]]:
        if len(features) < num_features:
            raise ValueError(
                f"At least {num_features} features are required, but only {len(features)} are available."
            )

        selected_features = random.sample(features, num_features)
        selected_names = [feature.get("feat_name", "unknown") for feature in selected_features]
        self.logger.info(f"Selected features for crossover: {selected_names}")
        return selected_features

    def get_crossover_system_prompt(self) -> str:
        return load_prompt("feature_crossover_system.txt")

    def get_crossover_user_prompt(self, feature1: Dict[str, Any], feature2: Dict[str, Any]) -> str:
        return render_prompt(
            "feature_crossover_user.txt",
            feature1_json=json.dumps(feature1, indent=2, ensure_ascii=False),
            feature2_json=json.dumps(feature2, indent=2, ensure_ascii=False),
        )

    def parse_crossover_response(self, response: str) -> Dict[str, Any]:
        try:
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:]
                if "```" in clean_response:
                    clean_response = clean_response.split("```")[0].strip()
            elif clean_response.startswith("```"):
                clean_response = clean_response[3:]
                if "```" in clean_response:
                    clean_response = clean_response.split("```")[0].strip()

            start_idx = clean_response.find("{")
            end_idx = clean_response.rfind("}") + 1
            if start_idx == -1 or end_idx <= start_idx:
                raise ValueError("No JSON object found in crossover response.")

            parsed = json.loads(clean_response[start_idx:end_idx])
            self.logger.debug("Parsed crossover response successfully.")
            return parsed
        except (json.JSONDecodeError, ValueError) as exc:
            self.logger.error(f"Failed to parse crossover response: {exc}")
            return {
                "error": "crossover_response_parse_error",
                "details": str(exc),
                "raw_content": response[:500] + "..." if len(response) > 500 else response,
            }

    def perform_crossover(self, json_file_path: str, num_pairs: int = 1) -> Dict[str, Any]:
        self.logger.info(f"Starting crossover from {json_file_path}")
        features = self.load_json_features(json_file_path)
        new_feature: Dict[str, Any] = {}

        for pair_idx in range(num_pairs):
            feature1, feature2 = self.random_select_features(features, 2)
            system_prompt = self.get_crossover_system_prompt()
            user_prompt = self.get_crossover_user_prompt(feature1, feature2)
            self.logger.info(f"Calling LLM for crossover {pair_idx + 1}/{num_pairs}")
            response = self.llm_client.call_with_system_prompt(system_prompt, user_prompt)
            new_feature = self.parse_crossover_response(response)

        return new_feature

    def save_crossover_results(self, crossover_features: List[Dict[str, Any]], output_file: str) -> None:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(crossover_features, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved crossover results to {output_file}")


def main() -> None:
    raise SystemExit(
        "JsonCrossover is intended to be used by fmm_run.py. "
        "Instantiate JsonCrossover and call perform_crossover(json_file_path)."
    )


if __name__ == "__main__":
    main()
