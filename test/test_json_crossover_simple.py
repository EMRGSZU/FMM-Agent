#!/usr/bin/env python3
"""
简洁的JSON特征交叉操作测试
从FMM_train_0.json中随机挑选两个JSON数据进行交叉处理
"""

import json
import sys
from pathlib import Path

# 添加主目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from operators.meta_crossover import JsonCrossover
from utils.fmm_logger import setup_fmm_logging, get_fmm_logger
from utils.llm_config_loader import create_llm_client


def test_json_crossover():
    """测试JSON交叉操作 - 简洁版本"""
    
    # 设置日志
    log_file_path = setup_fmm_logging("./logs")
    logger = get_fmm_logger("test_crossover", emoji="🧬")
    
    logger.info("开始JSON特征交叉操作测试")
    
    # 创建LLM客户端
    logger.info("创建LLM客户端...")
    llm_client = create_llm_client()
    logger.info("✓ LLM客户端创建成功")
    
    # 创建交叉操作器
    crossover = JsonCrossover(llm_client=llm_client)
    logger.info("✓ JsonCrossover实例创建成功")
    
    # 使用最新的JSON文件
    json_file = "/Users/feiyulv/Master/LLM/llm/algorithm/FMM-Agent/meta-info/meta-info_20251122_233944/FMM_train_0.json"
    
    try:
        # 加载特征数据
        features = crossover.load_json_features(json_file)
        logger.info(f"✓ 成功加载 {len(features)} 个特征")
        
        # 随机选择两个特征
        selected_features = crossover.random_select_features(features, 2)
        logger.info(f"✓ 随机选择了特征: {[f['feat_name'] for f in selected_features]}")
        
        # 生成提示词
        system_prompt = crossover.get_crossover_system_prompt()
        user_prompt = crossover.get_crossover_user_prompt(selected_features[0], selected_features[1])
        
        logger.info("✓ 提示词生成完成")
        logger.debug(f"系统提示词长度: {len(system_prompt)}")
        logger.debug(f"用户提示词长度: {len(user_prompt)}")
        
        # 执行交叉操作
        logger.info("执行交叉操作...")
        crossover_features = crossover.perform_crossover(json_file, num_pairs=1)
        
        # 显示结果
        logger.info(f"🎉 成功生成 {len(crossover_features)} 个交叉特征:")
        for i, feat in enumerate(crossover_features, 1):
            logger.info(f"  {i}. {feat.get('feat_name', 'unknown')}")
            logger.info(f"     理由: {feat.get('rationale', '无说明')[:100]}...")
        
        # 保存结果
        if crossover_features:
            output_file = "./logs/crossover_results.json"
            crossover.save_crossover_results(crossover_features, output_file)
            logger.info(f"✓ 结果已保存到: {output_file}")
        
        return crossover_features
        
    except Exception as e:
        logger.error(f"交叉操作失败: {e}")
        raise


def test_simple_crossover():
    """超简洁版本 - 只挑两个JSON做交叉"""
    
    # 设置日志
    logger = get_fmm_logger("simple_crossover", emoji="⚡")
    
    logger.info("=== 简单交叉操作测试 ===")
    
    # 创建LLM客户端
    llm_client = create_llm_client()
    
    # 创建交叉操作器
    crossover = JsonCrossover(llm_client=llm_client)
    
    # JSON文件路径
    json_file = "/Users/feiyulv/Master/LLM/llm/algorithm/FMM-Agent/meta-info/meta-info_20251122_233944/FMM_train_0.json"
    
    # 加载并随机选择两个特征
    features = crossover.load_json_features(json_file)
    selected = crossover.random_select_features(features, 2)
    
    logger.info(f"选择特征1: {selected[0]['feat_name']}")
    logger.info(f"选择特征2: {selected[1]['feat_name']}")
    
    # 显示特征详情
    logger.info(f"特征1统计信息: mean={selected[0].get('mean')}, std={selected[0].get('std')}")
    logger.info(f"特征2统计信息: mean={selected[1].get('mean')}, std={selected[1].get('std')}")
    
    # 执行交叉
    new_features = crossover.perform_crossover(json_file, num_pairs=1)
    
    logger.info(f"生成新特征数量: {len(new_features)}")
    # for feat in new_features:
    #     logger.info(f"新特征: {feat['feat_name']}")
    
    return new_features


if __name__ == "__main__":
    # 运行简洁测试
    print("运行JSON特征交叉操作测试...")
    
    try:
        # 选择运行哪个测试
        # results = test_json_crossover()  # 完整版本
        results = test_simple_crossover()  # 简洁版本
        print(results)
        
        print(f"\n✅ 测试完成！生成了 {len(results)} 个交叉特征")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        sys.exit(1)