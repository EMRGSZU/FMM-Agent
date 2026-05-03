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


class JsonMutation:
    """Generate one mutated feature metadata object from an FMM JSON file."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        if llm_client is None:
            from utils.llm_config_loader import create_llm_client

            self.llm_client = create_llm_client("deepseek")
        else:
            self.llm_client = llm_client
        self.logger = get_logger("FMM_json_mutation")

    def load_json_features(self, json_file_path: str) -> List[Dict[str, Any]]:
        with open(json_file_path, "r", encoding="utf-8") as f:
            features = json.load(f)

        if not isinstance(features, list):
            raise ValueError("Feature metadata JSON must contain a list.")
        if len(features) < 1:
            raise ValueError("At least one feature is required for mutation.")

        self.logger.info(f"Loaded {len(features)} features from {json_file_path}")
        return features

    def random_select_feature(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not features:
            raise ValueError("At least one feature is required for mutation.")

        selected_feature = random.choice(features)
        self.logger.info(f"Selected feature for mutation: {selected_feature.get('feat_name', 'unknown')}")
        return selected_feature

    def get_mutation_system_prompt(self) -> str:
        return load_prompt("feature_mutation_system.txt")

    def get_mutation_user_prompt(self, feature: Dict[str, Any]) -> str:
        return render_prompt(
            "feature_mutation_user.txt",
            feature_json=json.dumps(feature, indent=2, ensure_ascii=False),
        )

    def parse_mutation_response(self, response: str) -> Dict[str, Any]:
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
                raise ValueError("No JSON object found in mutation response.")

            parsed = json.loads(clean_response[start_idx:end_idx])
            self.logger.debug("Parsed mutation response successfully.")
            return parsed
        except (json.JSONDecodeError, ValueError) as exc:
            self.logger.error(f"Failed to parse mutation response: {exc}")
            return {
                "error": "mutation_response_parse_error",
                "details": str(exc),
                "raw_content": response[:500] + "..." if len(response) > 500 else response,
            }

    def perform_mutation(self, json_file_path: str, num_mutations: int = 1) -> Dict[str, Any]:
        self.logger.info(f"Starting mutation from {json_file_path}")
        features = self.load_json_features(json_file_path)
        mutation_result: Dict[str, Any] = {}

        for mutation_idx in range(num_mutations):
            selected_feature = self.random_select_feature(features)
            system_prompt = self.get_mutation_system_prompt()
            user_prompt = self.get_mutation_user_prompt(selected_feature)
            self.logger.info(f"Calling LLM for mutation {mutation_idx + 1}/{num_mutations}")
            response = self.llm_client.call_with_system_prompt(system_prompt, user_prompt)
            mutation_result = self.parse_mutation_response(response)

        return mutation_result

    def save_mutation_results(self, mutated_features: List[Dict[str, Any]], output_file: str) -> None:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(mutated_features, f, indent=2, ensure_ascii=False)
        self.logger.info(f"Saved mutation results to {output_file}")


def main() -> None:
    raise SystemExit(
        "JsonMutation is intended to be used by fmm_run.py. "
        "Instantiate JsonMutation and call perform_mutation(json_file_path)."
    )


if __name__ == "__main__":
    main()
