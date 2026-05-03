#!/usr/bin/env python3

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


class FeatureSelector:
    """Select feature metadata rows with the knee-point policy used by FMM-Agent."""

    def __init__(self, min_features: int = 5, max_features: int = 100):
        self.min_features = min_features
        self.max_features = max_features

    def select_by_knee_point(
        self,
        features: List[Dict],
        min_features: Optional[int] = None,
        max_features: Optional[int] = None,
    ) -> List[Dict]:
        min_count = self.min_features if min_features is None else min_features
        max_count = self.max_features if max_features is None else max_features

        n_features = len(features)
        if n_features == 0:
            return []
        if n_features <= min_count:
            return features[:min_count]

        scores = np.array([feature["FeatureScore"] for feature in features], dtype=float)
        x = np.arange(n_features, dtype=float)
        y = scores
        x1, y1 = 0.0, y[0]
        x2, y2 = float(n_features - 1), y[-1]

        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1
        denom = np.sqrt(a * a + b * b)
        if denom == 0:
            selected_count = max(min_count, min(max_count, n_features))
            return features[:selected_count]

        distances = np.abs(a * x + b * y + c) / denom
        knee_idx = int(np.argmax(distances))
        selected_count = knee_idx + 1
        selected_count = max(1, min(selected_count, max_count, n_features))
        selected_count = random.randint(1, selected_count)
        return features[:selected_count]

    def analyze_score_distribution(self, features: List[Dict]) -> Dict:
        scores = [feature["FeatureScore"] for feature in features]
        stats = {
            "total_features": len(features),
            "score_stats": {
                "min": min(scores),
                "max": max(scores),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "median": float(np.median(scores)),
            },
            "percentiles": {
                "25%": float(np.percentile(scores, 25)),
                "50%": float(np.percentile(scores, 50)),
                "75%": float(np.percentile(scores, 75)),
                "90%": float(np.percentile(scores, 90)),
                "95%": float(np.percentile(scores, 95)),
            },
        }

        if len(scores) > 1:
            differences = [scores[idx] - scores[idx + 1] for idx in range(len(scores) - 1)]
            stats["difference_stats"] = {
                "mean_gap": float(np.mean(differences)),
                "max_gap": float(max(differences)),
                "gap_p50": float(np.percentile(differences, 50)),
                "gap_p75": float(np.percentile(differences, 75)),
            }
        return stats

    def select_features(self, features: List[Dict]) -> Tuple[List[Dict], Dict]:
        if not features:
            return [], {
                "strategy": "knee",
                "total_features": 0,
                "selected_features": 0,
                "selection_ratio": 0.0,
                "score_range": {"selected_min": 0, "selected_max": 0, "overall_min": 0, "overall_max": 0},
                "score_stats": {},
            }

        features = sorted(features, key=lambda item: item["FeatureScore"], reverse=True)
        score_stats = self.analyze_score_distribution(features)
        selected = self.select_by_knee_point(features)

        selected_scores = [feature["FeatureScore"] for feature in selected] if selected else []
        selection_info = {
            "strategy": "knee",
            "total_features": len(features),
            "selected_features": len(selected),
            "selection_ratio": len(selected) / len(features) if features else 0,
            "score_range": {
                "selected_min": min(selected_scores) if selected_scores else 0,
                "selected_max": max(selected_scores) if selected_scores else 0,
                "overall_min": score_stats["score_stats"]["min"],
                "overall_max": score_stats["score_stats"]["max"],
            },
            "score_stats": score_stats,
        }
        return selected, selection_info

    def process_json_file(
        self, input_json_path: str, output_json_path: Optional[str] = None, save_analysis: bool = True
    ) -> Dict:
        with open(input_json_path, "r", encoding="utf-8") as f:
            features = json.load(f)

        if not features:
            print(f"Warning: {input_json_path} is empty or invalid.")
            return {}

        selected_features, selection_info = self.select_features(features)
        if output_json_path is None:
            input_dir = Path(input_json_path).parent
            output_json_path = input_dir / "selection_results" / f"{Path(input_json_path).stem}_selected_knee.json"

        output_json_path = Path(output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(selected_features, f, indent=2, ensure_ascii=False)

        if save_analysis:
            with open(output_json_path.with_suffix(".analysis.json"), "w", encoding="utf-8") as f:
                json.dump(selection_info, f, indent=2, ensure_ascii=False)

        print(
            "Feature selection complete: "
            f"input={input_json_path}, output={output_json_path}, "
            f"strategy=knee, selected={selection_info['selected_features']}"
        )
        return selection_info
