#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight checks for the knee-point feature selector."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from operators.feature_selector import FeatureSelector


def load_scored_features():
    scored_json_path = Path("test/data/scoring_results/FMM_train_0_scored.json")
    with open(scored_json_path, "r", encoding="utf-8") as f:
        return json.load(f), scored_json_path


def test_knee_selection():
    features, scored_json_path = load_scored_features()
    selector = FeatureSelector(min_features=5, max_features=100)
    selected, info = selector.select_features(features)

    print("=== Knee-point feature selection ===")
    print(f"Input file: {scored_json_path}")
    print(f"Total features: {info['total_features']}")
    print(f"Selected features: {info['selected_features']}")
    print(f"Selection ratio: {info['selection_ratio']:.2%}")

    assert selected, "Knee-point selection should return at least one feature."
    assert info["strategy"] == "knee"
    assert info["selected_features"] == len(selected)


def test_score_analysis():
    features, _ = load_scored_features()
    selector = FeatureSelector()
    stats = selector.analyze_score_distribution(features)

    print("\n=== Feature score distribution ===")
    score_stats = stats["score_stats"]
    print(f"Total features: {stats['total_features']}")
    print(f"Min score: {score_stats['min']:.4f}")
    print(f"Max score: {score_stats['max']:.4f}")
    print(f"Mean score: {score_stats['mean']:.4f}")
    print(f"Std score: {score_stats['std']:.4f}")
    print(f"Median score: {score_stats['median']:.4f}")

    assert stats["total_features"] == len(features)
    assert score_stats["max"] >= score_stats["min"]


def test_visualization_data():
    features, _ = load_scored_features()
    features.sort(key=lambda item: item["FeatureScore"], reverse=True)

    scores = [feature["FeatureScore"] for feature in features]
    feature_names = [feature["feat_name"] for feature in features]
    differences = [scores[i] - scores[i + 1] for i in range(len(scores) - 1)]

    analysis_data = {
        "feature_names": feature_names,
        "scores": scores,
        "differences": differences,
        "score_stats": {
            "min": min(scores),
            "max": max(scores),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        },
    }

    output_path = Path("test/data/scoring_results/score_analysis_data.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(analysis_data, f, indent=2, ensure_ascii=False)

    print("\n=== Score analysis data ===")
    print(f"Saved to: {output_path}")
    print(f"Features: {len(feature_names)}")
    print(f"Highest score: {scores[0]:.4f} ({feature_names[0]})")
    print(f"Lowest score: {scores[-1]:.4f} ({feature_names[-1]})")

    assert output_path.exists()


if __name__ == "__main__":
    print("Running feature selector checks...\n")
    test_knee_selection()
    test_score_analysis()
    test_visualization_data()
    print("\nAll feature selector checks completed.")
