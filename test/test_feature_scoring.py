#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试特征得分评价模块
"""

import sys
import json
import numpy as np
from pathlib import Path

# 添加主目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from operators.feature_scoring import FeatureScoring

def create_test_data():
    """创建测试数据"""
    test_features = [
        {
            "feat_name": "f1",
            "feat_type": "numeric",
            "mean": 0.041658592,
            "std": 0.1003642337,
            "skew": 6.339244274,
            "kurtosis": 42.7971926913,
            "min": 0.0015455032,
            "max": 0.8295439556,
            "q25": 0.017096165,
            "q75": 0.0263541145,
            "mi_to_y": 0.647365226,
            "info_gain": 0.794559935,
            "lineage": "raw",
            "ops_chain": [],
            "gen_round": 0
        },
        {
            "feat_name": "f2",
            "feat_type": "numeric",
            "mean": 0.447582844,
            "std": 0.1076514557,
            "skew": 0.8616755721,
            "kurtosis": 2.2901060118,
            "min": 0.1760231804,
            "max": 0.8459558479,
            "q25": 0.3886790011,
            "q75": 0.5075831026,
            "mi_to_y": 0.153293418,
            "info_gain": 0.1881485186,
            "lineage": "raw",
            "ops_chain": [],
            "gen_round": 0
        },
        {
            "feat_name": "f3",
            "feat_type": "numeric",
            "mean": 0.2920173837,
            "std": 0.1442911102,
            "skew": 1.7326802632,
            "kurtosis": 2.7027678678,
            "min": 0.0838439262,
            "max": 0.7512403136,
            "q25": 0.2064781108,
            "q75": 0.32257215,
            "mi_to_y": 0.3166818122,
            "info_gain": 0.3886873593,
            "lineage": "raw",
            "ops_chain": [],
            "gen_round": 0
        }
    ]
    return test_features


def test_individual_calculations():
    """测试各个计算函数"""
    print("=== 测试各个计算函数 ===")
    scorer = FeatureScoring(lambda_weight=0.6, c1=10.0, c2=50.0)
    
    # 测试数据
    test_features = create_test_data()
    
    for i, feature in enumerate(test_features):
        print(f"\n特征 {feature['feat_name']}:")
        
        # 测试离散度评分
        dispersion_score = scorer.calculate_dispersion_score(
            feature['q25'], feature['q75'], feature['min'], feature['max']
        )
        print(f"  离散度评分: {dispersion_score:.4f}")
        
        # 测试偏度评分
        skew_score = scorer.calculate_skew_score(feature['skew'])
        print(f"  偏度评分: {skew_score:.4f} (skew={feature['skew']:.4f})")
        
        # 测试峰度评分
        kurt_score = scorer.calculate_kurt_score(feature['kurtosis'])
        print(f"  峰度评分: {kurt_score:.4f} (kurtosis={feature['kurtosis']:.4f})")
        
        # 测试稳定性评分
        stability_score = scorer.calculate_stability_score(feature)
        print(f"  稳定性评分: {stability_score:.4f}")


def test_normalization():
    """测试归一化函数"""
    print("\n=== 测试归一化函数 ===")
    scorer = FeatureScoring()
    
    # 测试数据
    mi_values = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    ig_values = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    
    mi_norm, ig_norm = scorer.normalize_mi_ig(mi_values, ig_values)
    
    print(f"原始MI值: {mi_values}")
    print(f"归一化MI值: {mi_norm}")
    print(f"原始IG值: {ig_values}")
    print(f"归一化IG值: {ig_norm}")


def test_full_scoring():
    """测试完整的评分流程"""
    print("\n=== 测试完整评分流程 ===")
    
    # 创建测试JSON文件
    # test_features = create_test_data()
    test_json_path = "test/data/FMM_train_0.json"
    
    # with open(test_json_path, 'w', encoding='utf-8') as f:
    #     json.dump(test_features, f, indent=2, ensure_ascii=False)
    
    # 使用评分器处理
    scorer = FeatureScoring(lambda_weight=0.6, c1=10.0, c2=50.0)
    scorer.process_json_file(test_json_path)
    
    # 读取结果
    with open(test_json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # print(f"\n评分结果:")
    # for i, feature in enumerate(results):
    #     print(f"{i+1}. {feature['feat_name']}: {feature['FeatureScore']:.4f}")
        # if 'score_details' in feature:
        #     details = feature['score_details']
        #     print(f"   - 相关性得分: {details['rel_score']:.4f}")
        #     print(f"   - 稳定性得分: {details['stability_score']:.4f}")
        #     print(f"   - 离散度评分: {details['dispersion_score']:.4f}")
        #     print(f"   - 偏度评分: {details['skew_score']:.4f}")
        #     print(f"   - 峰度评分: {details['kurt_score']:.4f}")


def test_edge_cases():
    """测试边界情况"""
    print("\n=== 测试边界情况 ===")
    scorer = FeatureScoring()
    
    # 测试所有值相同的情况
    print("\n测试所有值相同的情况:")
    mi_same = np.array([0.5, 0.5, 0.5])
    ig_same = np.array([0.3, 0.3, 0.3])
    mi_norm, ig_norm = scorer.normalize_mi_ig(mi_same, ig_same)
    print(f"MI归一化结果: {mi_norm}")
    print(f"IG归一化结果: {ig_norm}")
    
    # 测试极端偏度和峰度
    print("\n测试极端偏度和峰度:")
    extreme_skew = 100.0
    extreme_kurt = 200.0
    skew_score = scorer.calculate_skew_score(extreme_skew)
    kurt_score = scorer.calculate_kurt_score(extreme_kurt)
    print(f"极端偏度({extreme_skew})评分: {skew_score:.6f}")
    print(f"极端峰度({extreme_kurt})评分: {kurt_score:.6f}")
    
    # 测试离散度边界
    print("\n测试离散度边界:")
    # 情况1: IQR = Range
    disp1 = scorer.calculate_dispersion_score(0.25, 0.75, 0.0, 1.0)
    print(f"IQR=Range时离散度评分: {disp1:.4f}")
    
    # 情况2: IQR > Range (理论上不可能，但测试min函数)
    disp2 = scorer.calculate_dispersion_score(0.0, 1.0, 0.0, 0.5)
    print(f"IQR>Range时离散度评分: {disp2:.4f}")


if __name__ == '__main__':
    print("开始测试特征得分评价模块...")
    
    # test_individual_calculations()
    # test_normalization()
    # test_edge_cases()
    test_full_scoring()
    
    print("\n=== 所有测试完成 ===")