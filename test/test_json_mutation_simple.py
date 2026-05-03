#!/usr/bin/env python3
"""
测试JSON特征变异操作
"""

import sys
from pathlib import Path

# 添加主目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from operators.meta_mutation import JsonMutation
from utils.fmm_logger import get_fmm_logger
from utils.llm_config_loader import create_llm_client


def test_json_mutation():
    """测试JSON特征变异操作"""
    logger = get_fmm_logger("test_mutation", emoji="🧪")
    logger.info("开始测试JSON特征变异操作...")
    
    # 创建LLM客户端
    llm_client = create_llm_client()
    
    # 创建变异操作器
    mutation = JsonMutation(llm_client=llm_client)
    
    # JSON文件路径
    json_file_path = "/Users/feiyulv/Master/LLM/llm/algorithm/FMM-Agent/meta-info/meta-info_20251122_233944/FMM_train_0.json"
    
    try:
        # 执行变异操作
        logger.info(f"从文件加载特征: {json_file_path}")
        mutated_features = mutation.perform_mutation(
            json_file_path=json_file_path,
            num_mutations=1  # 执行1次变异
        )
        
        # 保存结果
        if mutated_features:
            output_file = "/Users/feiyulv/Master/LLM/llm/algorithm/FMM-Agent/meta-info/test_mutated_features.json"
            mutation.save_mutation_results(mutated_features, output_file)
            
            # 打印结果摘要
            print(f"\n=== 变异操作结果摘要 ===")
            print(f"生成了 {len(mutated_features)} 个新特征:")
            for i, feat in enumerate(mutated_features, 1):
                print(f"  {i}. {feat.get('feat_name', 'unknown')}")
                print(f"     方法: {feat.get('mutation_method', '无说明')}")
                print(f"     理由: {feat.get('rationale', '无说明')[:100]}...")
                print()
            
            logger.info(f"测试完成！生成了 {len(mutated_features)} 个变异特征")
            return True
        else:
            logger.warning("没有生成任何变异特征")
            return False
            
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return False


def test_simple_mutation():
    """测试简单的变异操作"""
    logger = get_fmm_logger("test_simple", emoji="🔬")
    logger.info("开始测试简单变异操作...")
    
    # 创建LLM客户端
    llm_client = create_llm_client()
    
    # 创建变异操作器
    mutation = JsonMutation(llm_client=llm_client)
    
    try:
        # JSON文件路径
        json_file_path = "/Users/feiyulv/Master/LLM/llm/algorithm/FMM-Agent/meta-info/meta-info_20251122_233944/FMM_train_0.json"
        
        # 加载特征
        features = mutation.load_json_features(json_file_path)
        
        # 随机选择一个特征
        selected_feature = mutation.random_select_feature(features)
        
        # 获取提示词
        system_prompt = mutation.get_mutation_system_prompt()
        user_prompt = mutation.get_mutation_user_prompt(selected_feature)
        
        logger.info(f"选择的特征: {selected_feature.get('feat_name', 'unknown')}")
        logger.info(f"系统提示词长度: {len(system_prompt)} 字符")
        logger.info(f"用户提示词长度: {len(user_prompt)} 字符")
        
        # 调用LLM
        logger.info("调用LLM进行变异操作...")
        response = llm_client.call_with_system_prompt(system_prompt, user_prompt)
        
        # 解析响应
        result = mutation.parse_mutation_response(response)

        logger.info(f"变异操作结果: {result}")
        return True
            
    except Exception as e:
        logger.error(f"简单测试失败: {e}")
        return False


if __name__ == "__main__":
    print("=== 测试JSON特征变异操作 ===\n")
    
    # 运行完整测试
    # success1 = test_json_mutation()
    
    # print("\n" + "="*50 + "\n")
    
    # 运行简单测试
    success2 = test_simple_mutation()
    
    print(f"\n=== 测试结果 ===")
    # print(f"完整测试: {'✅ 通过' if success1 else '❌ 失败'}")
    print(f"简单测试: {'✅ 通过' if success2 else '❌ 失败'}")