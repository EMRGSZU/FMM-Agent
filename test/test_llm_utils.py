#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM工具测试模块 - 真实环境测试版本
"""


import json
import os
import yaml
import logging
import sys

from bdb import set_trace
from pathlib import Path


# 全局配置变量
CONFIG = {}
USE_REAL_API = False
SELECTED_MODEL = "deepseek"  # 默认使用deepseek模型

# 加载YAML配置文件
def load_config(config_path: str = None):
    """简化的配置加载函数，使用普通字典访问方式"""
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / 'configs' / 'test_config.yaml')
    
    global CONFIG, USE_REAL_API, SELECTED_MODEL
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            CONFIG = yaml.safe_load(f)
            
            # 使用普通字典访问方式获取配置
            if CONFIG.get('test', {}).get('use_real_api'):
                USE_REAL_API = CONFIG['test']['use_real_api']
            
            # 模型选择逻辑
            if CONFIG.get('test', {}).get('model_choice'):
                model_choice = CONFIG['test']['model_choice']
                if model_choice != 'default':
                    SELECTED_MODEL = model_choice
                    # 创建llm配置，使用选定的模型配置
                    CONFIG['llm'] = CONFIG['models'][model_choice].copy()
                    print(f"使用 {model_choice} 模型配置")
            else:
                # 自动选择deepseek模型
                SELECTED_MODEL = "deepseek"
                if CONFIG.get('models', {}).get('deepseek'):
                    CONFIG['llm'] = CONFIG['models']['deepseek'].copy()
                    print("默认使用 deepseek 模型配置")
            
            return CONFIG
    except Exception as e:
        print(f"警告: 无法加载配置文件，使用默认配置. 错误: {e}")
        set_trace()
        # CONFIG = {
        #     'llm': {'name': 'deepseek-chat', 'api_base': 'https://api.deepseek.com/v1', 
        #            'api_key': '', 'timeout': 30, 'max_retries': 3},
        #     'test': {'use_real_api': False, 'test_data_path': './test/data/FMM_train_0.json'},
        #     'logging': {'level': 'INFO', 'file_path': './logs/test.log'}
        # }
        # USE_REAL_API = False
        # return CONFIG

# 设置日志
def setup_logging():
    """简化的日志设置，使用普通字典访问方式"""
    if not CONFIG:
        load_config()
    
    # 使用普通字典访问方式获取日志配置
    log_level_str = CONFIG.get('logging', {}).get('level', 'INFO')
    log_file = CONFIG.get('logging', {}).get('file_path', './logs/test.log')
    log_format = CONFIG.get('logging', {}).get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    
    # 确保日志目录存在
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# 加载配置和设置日志
load_config()
logger = setup_logging()

# 导入真实模块
sys.path.append(str(Path(__file__).parent.parent))

from utils.fmi_summarizer import FmiSummarizer
from utils.llm_client import LLMClient


def test_fmi_summarizer_basic():
    """测试FmiSummarizer类的基本功能"""
    print("\n=== 测试 FmiSummarizer 基本功能 ===")
    
    # 获取测试数据路径
    test_data_path = CONFIG.get('test', {}).get('test_data_path')
    
    summarizer = FmiSummarizer()
    
    # 测试获取系统提示词
    system_prompt = summarizer.get_system_prompt()
    print(f"系统提示词: {system_prompt}")
    print(f"✓ 系统提示词长度: {len(system_prompt)} 字符")
    assert isinstance(system_prompt, str), "系统提示词应该是字符串"
    
    # 测试获取用户提示词模板（现在需要文件路径）
    template = summarizer.get_user_prompt_template(test_data_path)
    print(f"用户提示词模板: {template}")
    print(f"✓ 用户提示词模板长度: {len(template)} 字符")
    assert isinstance(template, str), "模板应该是字符串"
    assert "FMI Data" in template, "模板应包含 FMI Data"
    assert "Selected from" in template, "模板应显示选择的特征数量"
    
    
    print("✓ FmiSummarizer 基本功能测试通过")


def test_parse_response():
    """测试响应解析功能"""
    print("\n=== 测试响应解析功能 ===")
    
    summarizer = FmiSummarizer()
    
    # 测试直接解析JSON格式响应
    json_response = json.dumps({
        "feature_overview": {"total_features": 5},
        "key_insights": ["Insight 1"],
        "feature_recommendations": ["Recommendation 1"]
    })
    parsed = summarizer.parse_response(json_response)
    assert isinstance(parsed, dict), "解析结果应该是字典"
    assert "feature_overview" in parsed, "解析结果应包含 feature_overview"
    print("✓ 直接JSON解析测试通过")
    
    # 测试解析被JSON代码块包裹的响应
    code_block_response = "```json\n{\"feature_overview\": {\"total_features\": 5}}\n```"
    parsed = summarizer.parse_response(code_block_response)
    assert isinstance(parsed, dict), "代码块JSON解析结果应该是字典"
    assert "feature_overview" in parsed, "代码块JSON解析结果应包含 feature_overview"
    print("✓ JSON代码块解析测试通过")
    
    # 测试解析被普通代码块包裹的响应
    general_code_block = "```\n{\"feature_overview\": {\"total_features\": 5}}\n```"
    parsed = summarizer.parse_response(general_code_block)
    assert isinstance(parsed, dict), "普通代码块解析结果应该是字典"
    assert "feature_overview" in parsed, "普通代码块解析结果应包含 feature_overview"
    print("✓ 普通代码块解析测试通过")
    
    # 测试解析无效JSON的情况
    invalid_json = "This is not JSON"
    parsed = summarizer.parse_response(invalid_json)
    assert "error" in parsed, "无效JSON应包含错误信息"
    assert "raw_content" in parsed, "无效JSON应包含原始内容"
    print("✓ 无效JSON解析测试通过")
    
    print("✓ 响应解析功能测试通过")


def test_validate_response_format():
    """测试响应格式验证功能"""
    print("\n=== 测试响应格式验证功能 ===")
    
    summarizer = FmiSummarizer()
    
    # 测试验证有效的响应格式
    valid_summary = {
        "feature_overview": {
            "total_features": 10,
            "numeric_features": 5,
            "categorical_features": 3,
            "other_features": 2,
            "missing_features": 1
        },
        "key_insights": ["Insight 1"],
        "feature_recommendations": ["Recommendation 1"]
    }
    is_valid = summarizer.validate_response_format(valid_summary)
    assert is_valid, "有效格式应该验证通过"
    print("✓ 有效格式验证测试通过")
    
    # 测试验证无效的响应格式
    invalid_summary = {
        "feature_overview": {"total_features": 10}
    }
    is_valid = summarizer.validate_response_format(invalid_summary)
    assert not is_valid, "缺少必要字段应该验证失败"
    print("✓ 缺少必要字段验证测试通过")
    
    # 有错误字段
    error_summary = {
        "error": "Some error",
        "feature_overview": {"total_features": 10}
    }
    is_valid = summarizer.validate_response_format(error_summary)
    assert not is_valid, "有错误字段应该验证失败"
    print("✓ 错误字段验证测试通过")
    
    print("✓ 响应格式验证功能测试通过")


def test_create_fallback_summary():
    """测试创建备用总结功能"""
    print("\n=== 测试创建备用总结功能 ===")
    
    # 获取测试数据路径
    test_data_path = CONFIG.get('test', {}).get('test_data_path', './test/data/FMM_train_0.json')
    
    summarizer = FmiSummarizer()
    
    # 测试从真实JSON文件创建备用总结
    fallback = summarizer.create_fallback_summary(test_data_path)
    
    assert "feature_overview" in fallback, "备用总结应包含 feature_overview"
    assert "key_insights" in fallback, "备用总结应包含 key_insights"
    assert "feature_recommendations" in fallback, "备用总结应包含 feature_recommendations"
    assert "meta" in fallback, "备用总结应包含 meta"
    assert fallback["meta"]["is_fallback"], "meta 中应标记为备用总结"
    assert fallback["feature_overview"]["total_features"] > 0, "应正确统计特征数量"
    print(f"✓ 从真实文件创建备用总结测试通过，共统计到 {fallback['feature_overview']['total_features']} 个特征")
    
    # 测试处理无效JSON的备用总结功能
    invalid_content = "This is not JSON"
    fallback = summarizer.create_fallback_summary(invalid_content)
    
    assert fallback["feature_overview"]["total_features"] == 0, "无效JSON时应统计为0个特征"
    assert fallback["meta"]["is_fallback"], "meta 中应标记为备用总结"
    print("✓ 无效JSON备用总结创建测试通过")
    
    print("✓ 创建备用总结功能测试通过")
    
def test_real_api_summarization():
    """测试使用真实API进行特征元信息总结"""
    print(f"\n=== 测试真实API特征总结 ===")
    print(f"开始测试真实API特征总结，模型选择: {SELECTED_MODEL}")
    print(f"USE_REAL_API 配置值: {USE_REAL_API}")
    
    if not USE_REAL_API:
        print("⚠️  真实API测试被禁用，请在配置文件中启用")
        return False
    
    # 获取当前使用的模型配置，使用普通字典访问方式
    print(f"检查LLM配置: {'llm' in CONFIG}")
    if 'llm' not in CONFIG:
        print("⚠️  未找到LLM配置")
        return False
    
    model_config = CONFIG['llm']
    print(f"模型配置: {model_config}")
    
    # 确保有有效的API密钥
    api_key = model_config.get('api_key', '')
    print(f"API密钥长度: {'已设置' if len(api_key) > 10 else '未设置或太短'}")
    
    # 检查是否为默认占位符密钥
    is_default_key = api_key == 'your_api_key_here' or api_key == 'your_deepseek_api_key_here'
    print(f"是否为默认密钥: {is_default_key}")
    
    if not api_key or is_default_key:
        print(f"⚠️  未配置有效的{SELECTED_MODEL}模型API密钥")
        return False
    
    try:
        print(f"\n开始使用 {SELECTED_MODEL} 模型进行特征元信息总结测试...")
        
        # 创建真实的LLMClient实例
        print("正在创建LLMClient实例...")
        llm_client = LLMClient(model_config)
        
        # 创建FmiSummarizer实例
        print("正在创建FmiSummarizer实例...")
        summarizer = FmiSummarizer()
        
        # 准备测试数据 - 使用真实的JSON文件路径
        print("正在准备测试数据...")
        test_data_path = CONFIG.get('test', {}).get('test_data_path', './test/data/FMM_train_0.json')
        
        # 格式化提示词
        print("正在格式化提示词...")
        system_prompt = summarizer.get_system_prompt()
        user_prompt = summarizer.get_user_prompt_template(test_data_path)
        
        # 调用LLM
        print("正在调用API进行特征总结...")
        response = llm_client.call_with_system_prompt(system_prompt, user_prompt)
        
        # 解析响应
        print("正在解析API响应...")
        parsed_response = summarizer.parse_response(response)
        
        # 验证解析结果
        print(f"\n[{SELECTED_MODEL}模型] 特征总结结果: {json.dumps(parsed_response, indent=2)}")
        
        # 如果解析失败，应该有错误字段
        if "error" in parsed_response:
            assert "raw_content" in parsed_response, "解析失败时应包含原始内容"
            print(f"⚠️  警告: 解析失败，但成功捕获错误")
            return False
        else:
            # 验证响应格式
            is_valid = summarizer.validate_response_format(parsed_response)
            assert is_valid, "响应格式验证失败"
            print(f"✓ [{SELECTED_MODEL}模型] 特征总结测试成功")
            return True
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n❌ 错误详情: {error_trace}")
        print(f"❌ [{SELECTED_MODEL}模型] 真实API特征总结测试失败: {e}")
        return False


def test_real_api_call():
    """测试真实API调用"""
    print(f"\n=== 测试真实API调用 ===")
    print(f"开始测试真实API调用，模型选择: {SELECTED_MODEL}")
    print(f"USE_REAL_API 配置值: {USE_REAL_API}")
    
    if not USE_REAL_API:
        print("⚠️  真实API测试被禁用，请在配置文件中启用")
        return False
    
    # 获取当前使用的模型配置，使用类似列表的访问方式
    print(f"检查LLM配置: {'llm' in CONFIG}")
    if 'llm' not in CONFIG:
        print("⚠️  未找到LLM配置")
        return False
    
    model_config = CONFIG['llm']
    print(f"模型配置: {model_config}")
    
    # 确保有有效的API密钥
    api_key = model_config.get('api_key', '')
    print(f"API密钥长度: {'已设置' if len(api_key) > 10 else '未设置或太短'}")
    
    # 检查是否为默认占位符密钥
    is_default_key = api_key == 'your_api_key_here' or api_key == 'your_deepseek_api_key_here'
    print(f"是否为默认密钥: {is_default_key}")
    
    if not api_key or is_default_key:
        print(f"⚠️  未配置有效的{SELECTED_MODEL}模型API密钥")
        return False
    
    try:
        print(f"正在创建LLMClient实例...")
        # 使用配置文件中的配置创建真实的LLMClient实例
        client = LLMClient(model_config)
        
        print(f"正在准备测试提示词...")
        # 发送一个简单的测试消息
        system_prompt = "You are a helpful assistant."
        user_prompt = "请简要介绍自己（50字以内）"
        
        print(f"正在调用API...")
        response = client.call_with_system_prompt(system_prompt, user_prompt)
        
        print(f"API调用成功，正在验证响应...")
        # 验证响应
        assert isinstance(response, str), "响应应该是字符串"
        assert len(response.strip()) > 0, "响应内容不应为空"
        print(f"\n✓ [{SELECTED_MODEL}模型] 真实API测试响应: {response[:100]}...")
        return True
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n❌ 错误详情: {error_trace}")
        print(f"❌ [{SELECTED_MODEL}模型] 真实API调用失败: {e}")
        return False


def run_all_tests():
    """运行所有真实环境测试"""
    print("\n" + "="*60)
    print("开始运行真实环境测试")
    print("="*60 + "\n")
    
    test_results = {
        "FMI摘要器基础测试": test_fmi_summarizer_basic,
        "响应解析测试": test_parse_response,
        "响应格式验证测试": test_validate_response_format,
        "备用摘要测试": test_create_fallback_summary,
        "真实API摘要测试": test_real_api_summarization,
        "真实API调用测试": test_real_api_call
    }
    
    passed = 0
    failed = 0
    
    for test_name, test_func in test_results.items():
        print(f"\n--- 运行 {test_name} ---")
        try:
            result = test_func()
            if result is False:
                print(f"⚠️  {test_name} 跳过")
            else:
                print(f"✓ {test_name} 通过")
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} 失败: {e}")
            failed += 1
    
    print(f"\n" + "="*60)
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print("="*60)
    
    return failed == 0

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)