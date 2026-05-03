#!/usr/bin/env python3

import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np


class FeatureScoring:
    """Compute FeatureScore values from feature metadata."""

    def __init__(self, lambda_weight: float = 0.6, c1: float = 10.0, c2: float = 50.0):
        self.lambda_weight = lambda_weight
        self.c1 = c1
        self.c2 = c2

    def normalize_mi_ig(self, mi_values: np.ndarray, ig_values: np.ndarray) -> tuple:
        mi_min, mi_max = np.min(mi_values), np.max(mi_values)
        ig_min, ig_max = np.min(ig_values), np.max(ig_values)
        mi_norm = np.full_like(mi_values, 0.5) if mi_max == mi_min else (mi_values - mi_min) / (mi_max - mi_min)
        ig_norm = np.full_like(ig_values, 0.5) if ig_max == ig_min else (ig_values - ig_min) / (ig_max - ig_min)
        return mi_norm, ig_norm

    def calculate_dispersion_score(self, q25: float, q75: float, min_val: float, max_val: float) -> float:
        epsilon = 1e-10
        range_val = max_val - min_val + epsilon
        if range_val <= epsilon:
            return 1.0
        return min((q75 - q25) / range_val, 1.0)

    def calculate_skew_score(self, skew_val: float) -> float:
        return np.exp(-abs(skew_val) / self.c1)

    def calculate_kurt_score(self, kurt_val: float) -> float:
        return np.exp(-abs(kurt_val) / self.c2)

    def calculate_stability_score(self, feature: Dict) -> float:
        dispersion_score = self.calculate_dispersion_score(
            feature["q25"], feature["q75"], feature["min"], feature["max"]
        )
        skew_score = self.calculate_skew_score(feature["skew"])
        kurt_score = self.calculate_kurt_score(feature["kurtosis"])
        return (dispersion_score + skew_score + kurt_score) / 3.0

    def calculate_feature_score(self, feature: Dict, mi_norm: float, ig_norm: float) -> float:
        rel_score = (mi_norm + ig_norm) / 2.0
        stability_score = self.calculate_stability_score(feature)
        return self.lambda_weight * rel_score + (1 - self.lambda_weight) * stability_score

    def calculate_feature_scores(self, all_features: List[Dict]) -> List[Dict]:
        if not all_features:
            return all_features

        mi_values = np.array([feature["mi_to_y"] for feature in all_features])
        ig_values = np.array([feature["info_gain"] for feature in all_features])
        mi_min, mi_max = np.min(mi_values), np.max(mi_values)
        ig_min, ig_max = np.min(ig_values), np.max(ig_values)

        for feature in all_features:
            current_mi = feature["mi_to_y"]
            current_ig = feature["info_gain"]
            mi_norm = 0.5 if mi_max == mi_min else (current_mi - mi_min) / (mi_max - mi_min)
            ig_norm = 0.5 if ig_max == ig_min else (current_ig - ig_min) / (ig_max - ig_min)
            feature["FeatureScore"] = float(self.calculate_feature_score(feature, mi_norm, ig_norm))

        return all_features

    def process_json_file(
        self,
        json_file_path: str,
        output_file_path: Optional[str] = None,
        number: int = 0,
    ) -> None:
        with open(json_file_path, "r", encoding="utf-8") as f:
            features = json.load(f)

        if not features:
            print(f"Warning: {json_file_path} is empty or invalid.")
            return

        features = self.calculate_feature_scores(features)
        features.sort(key=lambda item: item["FeatureScore"], reverse=True)

        if output_file_path is None:
            output_file_path = os.path.dirname(json_file_path)
        os.makedirs(output_file_path, exist_ok=True)

        file_name = os.path.join(output_file_path, f"iter_{number}.json")
        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(features, f, indent=2, ensure_ascii=False)

        print(f"Feature scoring complete: input={json_file_path}, output={file_name}, features={len(features)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Score feature metadata JSON files.")
    parser.add_argument("json_file", help="Input JSON metadata file.")
    parser.add_argument("-o", "--output", help="Output directory.")
    parser.add_argument("--_lambda", type=float, default=0.6, help="Relevance score weight.")
    parser.add_argument("--c1", type=float, default=10.0, help="Skewness score constant.")
    parser.add_argument("--c2", type=float, default=50.0, help="Kurtosis score constant.")
    args = parser.parse_args()

    if not os.path.exists(args.json_file):
        raise FileNotFoundError(f"Input file does not exist: {args.json_file}")

    scorer = FeatureScoring(lambda_weight=args._lambda, c1=args.c1, c2=args.c2)
    scorer.process_json_file(args.json_file, args.output)


if __name__ == "__main__":
    main()
