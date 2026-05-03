import http.client
import json
import os
import yaml
from openai import OpenAI

def gpt5_1():
    conn = http.client.HTTPSConnection("oa.api2d.net")
    payload = json.dumps({
       "model": "gpt-5.1",
       "messages": [
          {
             "role": "user",
             "content": "讲个笑话"
          }
       ],
       "safe_mode": False
    })
    headers = {
       'Authorization': 'Bearer fk231384-FmyurWltY7evn8SA0A3SlC584k1UquWi',
       'Content-Type': 'application/json'
    }
    conn.request("POST", "/v1/chat/completions", payload, headers)
    res = conn.getresponse()
    data = res.read()
    response_json = json.loads(data.decode("utf-8"))
    print(response_json['choices'][0]['message']['content'])
"""
{"id":"chatcmpl-CpnK5kEVsCgrycAGhhUYKgiPaQSeq","object":"chat.completion","created":1766460361,"model":"gpt-5.1-2025-11-13","choices":[{"index":0,"message":{"role":"assistant","content":"有个人去看牙医，问医生：\n\n“拔一颗牙多少钱？”\n\n医生说：“300。”\n\n他想了想说：“这么贵啊，能不能拔慢一点？”","refusal":null,"annotations":[]},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":47,"total_tokens":57,"prompt_tokens_details":{"cached_tokens":0,"audio_tokens":0},"completion_tokens_details":{"reasoning_tokens":0,"audio_tokens":0,"accepted_prediction_tokens":0,"rejected_prediction_tokens":0},"pre_token_count":16384,"pre_total":956,"adjust_total":953,"final_total":3},"service_tier":"default","system_fingerprint":null}
"""

def claude4():
    conn = http.client.HTTPSConnection("oa.api2d.net")
    payload = json.dumps({
       "model": "claude-sonnet-4-5",
       "messages": [
          {
             "role": "user",
             "content": "讲个笑话"
          }
       ],
       "stream": False,
       "max_tokens": 100
    })
    headers = {
      #  'x-api-key': 'fk231384-FmyurWltY7evn8SA0A3SlC584k1UquWi',
       'Authorization': 'Bearer fk231384-FmyurWltY7evn8SA0A3SlC584k1UquWi', 
       'Content-Type': 'application/json'
    }
    conn.request("POST", "/claude/v1/messages", payload, headers)
    res = conn.getresponse()
    data = res.read()
    response_json = json.loads(data.decode("utf-8"))
    print(response_json['content'][0]['text'])

def gemini3():
   conn = http.client.HTTPSConnection("oa.api2d.net")
   payload = json.dumps({
      "model": "gemini-2.0-flash",
      "messages": [
         {
            "role": "user",
            "content": "讲个笑话"
         }
      ],
      "safe_mode": "false",
      "moderation": "false",
      "moderation_stop": "false",
      "stream": "false",
      "max_tokens": 100
   })
   headers = {
      'Authorization': 'Bearer fk231384-FmyurWltY7evn8SA0A3SlC584k1UquWi',
      'Content-Type': 'application/json'
   }
   conn.request("POST", "/v1/chat/completions", payload, headers)
   res = conn.getresponse()
   data = res.read()
   response_json = json.loads(data.decode("utf-8"))
   print(response_json['choices'][0]['message']['content'])
"""
{"choices":[{"finish_reason":"stop","index":0,"message":{"content":"为什么程序员喜欢用深色主题？\n\n因为亮瞎了他们的双眼，更容易发现Bug！\n","role":"assistant"}}],"created":1766470234,"id":"WTJKadyrE5y7juMPkaCRKQ","model":"gemini-2.0-flash","object":"chat.completion","usage":{"completion_tokens":22,"prompt_tokens":46,"total_tokens":68,"final_total":1}}
"""




def load_config():
    """加载YAML配置文件"""
    with open('algorithm/FMM-Agent/configs/llm.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def get_deepseek_client():
    """创建DeepSeek客户端"""
    config = load_config()
    deepseek_config = config.get('models', {}).get('deepseek', {})
    
    # 获取API密钥（优先使用环境变量）
    api_key = deepseek_config.get('api_key', '')
    
    return OpenAI(
        api_key=api_key,
        base_url=deepseek_config.get('api_base', 'https://api.deepseek.com'),
        timeout=deepseek_config.get('timeout', 30),
        max_retries=deepseek_config.get('max_retries', 3)
    )

def deepseek_joke():
    """讲个笑话"""
    try:
        client = get_deepseek_client()
        
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
               {"role": "system", "content": "你是一个歌手"},
               {"role": "user", "content": "讲个笑话"}
            ],
            max_tokens=100,
            stream=False
        )
        print(response.choices[0].message.content)
      #   return response.choices[0].message.content
        
    except Exception as e:
        print(f"请求失败: {e}")
        return None
   


if __name__ == "__main__":
   #  gpt5_1()
   #  claude4()
   #  gemini3()
    deepseek_joke()
