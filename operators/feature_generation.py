import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mstats

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.llm_config_loader import create_llm_client
from utils.log import get_logger
from utils.prompt_loader import load_prompt, render_prompt


_LLM_CLIENT = None


def get_llm_client():
    """Return a lazily initialized LLM client for feature generation."""
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        _LLM_CLIENT = create_llm_client("deepseek")
    return _LLM_CLIENT


def get_trend_system_prompt() -> str:
    return load_prompt("trend_system.txt")


def get_trend_user_prompt(parent_meta_info: dict, child_meta_info: dict) -> str:
    ops_chain = child_meta_info.get("ops_chain", [])
    ops_description = "No transformation description available"
    if isinstance(ops_chain, list) and ops_chain:
        ops_description = str(ops_chain[0])

    return render_prompt(
        "trend_user.txt",
        parent_json=json.dumps(parent_meta_info, indent=2, ensure_ascii=False),
        child_json=json.dumps(child_meta_info, indent=2, ensure_ascii=False),
        ops_description=ops_description,
    )


def parse_llm_trend_response(response: str) -> dict:
    """Parse trend recommendations from an LLM response."""
    logger = get_logger("FMM_feature_generation")
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
            raise ValueError("No JSON object found in trend response.")

        parsed = json.loads(clean_response[start_idx:end_idx])
        required_fields = ["mean", "std", "skew", "kurt", "min", "max"]
        for field in required_fields:
            if field not in parsed:
                raise ValueError(f"Missing required field: {field}")
            if not isinstance(parsed[field], (int, float)):
                raise ValueError(f"Field {field} must be numeric")
        return parsed
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error(f"Failed to parse LLM trend response: {exc}")
        return {"mean": 0.0, "std": 0.0, "skew": 0.0, "kurt": 0.0, "min": 0.0, "max": 0.0}


def call_llm_for_trend(parent_meta_info: dict, child_meta_info: dict) -> dict:
    """Ask the LLM for a compact trend vector and fall back to heuristic trends."""
    logger = get_logger("FMM_feature_generation")
    try:
        llm_client = get_llm_client()
        system_prompt = get_trend_system_prompt()
        user_prompt = get_trend_user_prompt(parent_meta_info, child_meta_info)
        response = llm_client.call_with_system_prompt(system_prompt, user_prompt)
        trend = parse_llm_trend_response(response)
        logger.info(
            "Generated trend recommendations: "
            f"mean={trend['mean']:.3f}, std={trend['std']:.3f}, "
            f"skew={trend['skew']:.3f}, kurt={trend['kurt']:.3f}"
        )
        return trend
    except Exception as exc:
        logger.error(f"LLM trend analysis failed: {exc}")
        logger.info("Falling back to heuristic trend recommendations.")
        return get_simple_trend(parent_meta_info, child_meta_info)


def get_simple_trend(parent_meta_info: dict, child_meta_info: dict) -> dict:
    """Infer a conservative trend vector from the operation description."""
    trend = {"mean": 0.1, "std": 0.0, "skew": 0.0, "kurt": 0.0, "min": 0.0, "max": 0.0}

    ops_chain = child_meta_info.get("ops_chain", [])
    if not isinstance(ops_chain, list) or not ops_chain:
        return trend

    ops_description = str(ops_chain[-1]).lower()
    if "log" in ops_description:
        trend.update({"mean": -0.1, "std": -0.2, "skew": 0.3, "kurt": 0.1})
    elif "sqrt" in ops_description:
        trend.update({"mean": 0.0, "std": -0.1, "skew": 0.2})
    elif "scale" in ops_description or "normalize" in ops_description:
        trend.update({"std": 0.2, "min": -0.1, "max": 0.1})
    elif "clip" in ops_description:
        trend.update({"std": -0.1, "min": 0.1, "max": -0.1, "kurt": -0.2})
    elif "multiply" in ops_description or "product" in ops_description:
        trend.update({"mean": 0.2, "std": 0.3, "skew": 0.1, "kurt": 0.2})
    elif "divide" in ops_description or "ratio" in ops_description:
        trend.update({"std": 0.2, "min": -0.2, "max": 0.2, "kurt": 0.3})
    elif "add" in ops_description or "sum" in ops_description:
        trend.update({"mean": 0.1, "std": 0.0, "skew": 0.0})
    elif "subtract" in ops_description or "difference" in ops_description:
        trend.update({"mean": 0.0, "std": -0.1, "skew": 0.1})

    parent_score = float(parent_meta_info.get("FeatureScore", 0.0) or 0.0)
    child_score = float(child_meta_info.get("FeatureScore", 0.0) or 0.0)
    if child_score > parent_score * 1.2:
        trend["mean"] += 0.1
        trend["std"] += 0.05
    elif child_score < parent_score * 0.8:
        trend["mean"] -= 0.1
        trend["std"] -= 0.05

    for key in trend:
        trend[key] = max(-0.8, min(0.8, trend[key]))
    return trend


def _load_parent_metadata(output_file_path: str, parent_col: str) -> dict | None:
    if not os.path.exists(output_file_path):
        return None

    with open(output_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for feature in data:
        if feature.get("feat_name") == parent_col:
            return feature
    return None


def _safe_factor(value: float, min_factor: float = 0.1, max_factor: float = 3.0) -> float:
    factor = 1.0 + float(value)
    return max(min_factor, min(max_factor, factor))


def _stats_from_meta(meta_info: dict) -> dict:
    return {
        "mean": meta_info.get("mean", 0.0),
        "std": meta_info.get("std", 1.0),
        "skew": meta_info.get("skew", 0.0),
        "kurt": meta_info.get("kurtosis", 0.0),
        "min": meta_info.get("min", 0.0),
        "max": meta_info.get("max", 1.0),
        "q01": meta_info.get("q01", meta_info.get("min", 0.0)),
        "q99": meta_info.get("q99", meta_info.get("max", 1.0)),
    }


def generate_new_feature_value(
    train_raw,
    parent_feat_name,
    child_feat_name,
    child_meta_info,
    output_file_path,
):
    """
    Generate one derived feature column from a parent column and child FMM metadata.

    The transformation uses an LLM trend vector when available and falls back to a
    conservative heuristic. The generated values are clipped and winsorized to
    avoid numerical instability.
    """
    parent_col = parent_feat_name
    child_col = child_feat_name
    child_meta_info["lineage"] = "raw"

    if parent_col not in train_raw.columns:
        train_raw[child_col] = 0.0
        return train_raw

    parent_meta_info = _load_parent_metadata(output_file_path, parent_col)
    if parent_meta_info is None:
        parent_values = train_raw[parent_col].astype(float).values
        train_raw[child_col] = parent_values * parent_values
        return train_raw

    parent_stats = _stats_from_meta(parent_meta_info)
    child_stats_meta = _stats_from_meta(child_meta_info)
    trend = call_llm_for_trend(parent_meta_info, child_meta_info)

    parent_mean = float(parent_stats["mean"])
    parent_std = float(parent_stats["std"]) if float(parent_stats["std"]) > 1e-8 else 1.0
    parent_min = float(parent_stats["min"])
    parent_max = float(parent_stats["max"])

    target_mean = parent_mean * _safe_factor(trend.get("mean", 0.0))
    target_std = max(parent_std * _safe_factor(trend.get("std", 0.0)), 1e-8)
    target_min = parent_min if parent_min == 0.0 else parent_min * _safe_factor(trend.get("min", 0.0))
    target_max = parent_max if parent_max == 0.0 else parent_max * _safe_factor(trend.get("max", 0.0))

    if target_min >= target_max:
        target_min = min(parent_min, parent_max)
        target_max = max(parent_min, parent_max)

    parent_values = train_raw[parent_col].astype(float).values
    finite_mask = np.isfinite(parent_values)
    if not finite_mask.any():
        train_raw[child_col] = target_mean
        return train_raw

    values = parent_values.copy()
    values[~finite_mask] = np.nanmedian(parent_values[finite_mask])
    z = (values - parent_mean) / parent_std
    new_values = z * target_std + target_mean

    skew_trend = float(trend.get("skew", 0.0))
    kurt_trend = float(trend.get("kurt", 0.0))
    if abs(skew_trend) > 0.1 or abs(kurt_trend) > 0.1:
        skew_alpha = max(-0.2, min(0.2, skew_trend * 0.2))
        kurt_alpha = max(-0.1, min(0.1, kurt_trend * 0.1))
        shape_adjustment = skew_alpha * (z ** 3) + kurt_alpha * (z ** 2)
        new_values = new_values + shape_adjustment * target_std * 0.1

    q01 = min(child_stats_meta.get("q01", parent_stats["q01"]), parent_stats["q01"])
    q99 = max(child_stats_meta.get("q99", parent_stats["q99"]), parent_stats["q99"])
    clip_min = max(target_min, q01)
    clip_max = min(target_max, q99)
    new_values = np.clip(new_values, clip_min, clip_max)

    winsorized = mstats.winsorize(new_values, limits=[0.005, 0.005])
    new_values = np.array(winsorized.filled(np.nan) if np.ma.is_masked(winsorized) else winsorized)
    new_values = np.nan_to_num(new_values, nan=target_mean, posinf=clip_max, neginf=clip_min)

    train_raw[child_col] = new_values
    try:
        child_meta_info.update(
            {
                "mean": float(np.mean(new_values)),
                "std": float(np.std(new_values, ddof=1)) if len(new_values) > 1 else 0.0,
                "min": float(np.min(new_values)),
                "max": float(np.max(new_values)),
                "q01": float(np.quantile(new_values, 0.01)),
                "q99": float(np.quantile(new_values, 0.99)),
                "skew": float(mstats.skew(new_values)) if len(new_values) > 2 else 0.0,
                "kurtosis": float(mstats.kurtosis(new_values)) if len(new_values) > 3 else 0.0,
            }
        )
    except Exception as exc:
        child_meta_info["generation_error"] = str(exc)

    return train_raw
