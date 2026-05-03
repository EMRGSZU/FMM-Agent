#!/usr/bin/env python3

import json
from typing import Any, Dict

from utils.log import get_logger
from utils.prompt_loader import load_prompt, render_prompt


class FmiSummarizer:
    """Create and validate structured summaries for feature meta information."""

    def __init__(self):
        self.logger = get_logger("FMM_fmi_summarizer")

    def get_system_prompt(self) -> str:
        return load_prompt("fmi_summary_system.txt")

    def get_user_prompt_template(self, json_file_path: str) -> str:
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                feature_data = json.load(f)

            if not isinstance(feature_data, list):
                raise ValueError("Feature metadata JSON must contain a list.")
            if not feature_data:
                raise ValueError("Feature metadata list is empty.")

            selected_feature = feature_data[0]
            feature_meta_content = json.dumps(selected_feature, indent=2, ensure_ascii=False)
            feature_count = len(feature_data)
        except FileNotFoundError:
            self.logger.error(f"JSON file not found: {json_file_path}")
            feature_meta_content = "Error: JSON file not found."
            feature_count = 0
        except json.JSONDecodeError as exc:
            self.logger.error(f"Invalid JSON file: {exc}")
            feature_meta_content = "Error: invalid JSON file."
            feature_count = 0
        except Exception as exc:
            self.logger.error(f"Failed to read feature metadata: {exc}")
            feature_meta_content = "Error: failed to read feature metadata."
            feature_count = 0

        return render_prompt(
            "fmi_summary_user.txt",
            feature_count=feature_count,
            feature_meta_content=feature_meta_content,
        )

    def parse_response(self, response: str) -> Dict[str, Any]:
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
                raise ValueError("No JSON object found in FMI summary response.")

            parsed = json.loads(clean_response[start_idx:end_idx])
            self.logger.debug("Parsed FMI summary response successfully.")
            return parsed
        except (json.JSONDecodeError, ValueError) as exc:
            self.logger.error(f"Failed to parse FMI summary response: {exc}")
            return {
                "error": "fmi_summary_parse_error",
                "details": str(exc),
                "raw_content": response[:500] + "..." if len(response) > 500 else response,
            }

    def validate_response_format(self, summary: Dict[str, Any]) -> bool:
        if "error" in summary:
            self.logger.warning("FMI summary contains an error field.")
            return False

        required_fields = ["feature_overview", "key_insights", "feature_recommendations"]
        missing_fields = [field for field in required_fields if field not in summary]
        if missing_fields:
            self.logger.warning(f"FMI summary is missing fields: {missing_fields}")
            return False

        feature_overview = summary["feature_overview"]
        if not isinstance(feature_overview, dict):
            self.logger.warning("feature_overview must be an object.")
            return False

        overview_fields = [
            "total_features",
            "numeric_features",
            "categorical_features",
            "other_features",
            "missing_features",
        ]
        missing_overview = [field for field in overview_fields if field not in feature_overview]
        if missing_overview:
            self.logger.warning(f"feature_overview is missing fields: {missing_overview}")
            return False

        if not all(isinstance(feature_overview[field], int) for field in overview_fields):
            self.logger.warning("All feature_overview values must be integers.")
            return False

        if not isinstance(summary["key_insights"], list):
            self.logger.warning("key_insights must be a list.")
            return False
        if not isinstance(summary["feature_recommendations"], list):
            self.logger.warning("feature_recommendations must be a list.")
            return False

        return True

    def create_fallback_summary(self, json_file_path: str) -> Dict[str, Any]:
        self.logger.warning("Creating fallback FMI summary.")
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                features = json.load(f)

            if not isinstance(features, list):
                raise ValueError("Feature metadata JSON must contain a list.")

            total = len(features)
            numeric = len([feature for feature in features if feature.get("feat_type") == "numeric"])
            categorical = len([feature for feature in features if feature.get("feat_type") == "categorical"])
            other = total - numeric - categorical
            missing = len([feature for feature in features if feature.get("missing_rate", 0) > 0])
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
            self.logger.error(f"Failed to create fallback summary: {exc}")
            total = numeric = categorical = other = missing = 0

        return {
            "feature_overview": {
                "total_features": total,
                "numeric_features": numeric,
                "categorical_features": categorical,
                "other_features": other,
                "missing_features": missing,
            },
            "key_insights": [],
            "feature_recommendations": [],
            "meta": {
                "is_fallback": True,
                "total_features": total,
                "numeric_features": numeric,
                "categorical_features": categorical,
                "other_features": other,
                "missing_features": missing,
            },
        }
