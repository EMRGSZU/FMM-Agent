#!/usr/bin/env python3
"""
增强版LLM客户端测试脚本
演示如何使用不同的model_choice调用不同的LLM模型
"""

import os
import sys
import yaml
from utils.enhanced_llm_client import EnhancedLLMClient, create_llm_client
from utils.log import get_logger

def load_config():
    """加载配置文件"""
    config_path = os.path.join("configs", "llm.yaml")
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def test_different_models():
    """测试不同的模型选择"""
    logger = get_logger("test_enhanced_llm", emoji="🧪")
    
    # 加载配置
    config = load_config()
    
    # 测试用的提示词
    system_prompt = "你是一个幽默的助手，请用轻松有趣的方式回答问题。"
    user_prompt = "请讲一个关于程序员的笑话"
    
    # 获取可用模型列表
    available_models = list(config.get("models", {}).keys())
    logger.info(f"可用模型: {available_models}")
    
    # 测试每个模型
    for model_choice in available_models:
        logger.info(f"\n{'='*50}")
        logger.info(f"测试模型: {model_choice}")
        logger.info(f"{'='*50}")
        
        try:
            # 创建客户端
            client = EnhancedLLMClient(config, model_choice)
            
            # 调用LLM
            response = client.call_with_system_prompt(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.7,
                max_tokens=200
            )
            
            logger.info(f"✅ {model_choice.upper()} 响应成功:")
            logger.info(f"响应内容: {response}")
            
        except Exception as e:
            logger.error(f"❌ {model_choice.upper()} 调用失败: {e}")
            continue

def test_model_switching():
    """测试模型切换功能"""
    logger = get_logger("test_model_switching", emoji="🔄")
    
    config = load_config()
    client = EnhancedLLMClient(config, "deepseek")  # 初始使用deepseek
    
    logger.info("测试模型切换功能")
    logger.info(f"当前模型: {client.model_choice}")
    
    # 切换到GPT
    client.switch_model("gpt")
    logger.info(f"切换到模型: {client.model_choice}")
    
    # 切换到Claude
    client.switch_model("claude")
    logger.info(f"切换到模型: {client.model_choice}")
    
    # 切换到Gemini
    client.switch_model("gemini")
    logger.info(f"切换到模型: {client.model_choice}")

def test_direct_llm_call():
    """测试直接调用LLM方法"""
    logger = get_logger("test_direct_call", emoji="📞")
    
    config = load_config()
    client = EnhancedLLMClient(config, "deepseek")
    
    # 构建消息列表
    messages = [
        {"role": "system", "content": "你是一个知识渊博的助手。"},
        {"role": "user", "content": "请简单介绍一下人工智能"}
    ]
    
    logger.info("测试直接调用LLM方法")
    
    try:
        response = client.call_llm(messages, temperature=0.3, max_tokens=150)
        logger.info(f"响应内容: {response}")
    except Exception as e:
        logger.error(f"调用失败: {e}")

if __name__ == "__main__":
    logger = get_logger("main_test", emoji="🚀")
    
    logger.info("开始增强版LLM客户端测试")
    
    # 测试1: 测试不同模型
    logger.info("\n📋 测试1: 测试不同模型")
    test_different_models()
    
    # 测试2: 测试模型切换
    logger.info("\n📋 测试2: 测试模型切换功能")
    test_model_switching()
    
    # 测试3: 测试直接调用
    logger.info("\n📋 测试3: 测试直接调用LLM方法")
    test_direct_llm_call()
    
    logger.info("\n✅ 所有测试完成！")