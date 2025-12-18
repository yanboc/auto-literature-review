import os

API_KEY = os.getenv("DASHSCOPE_API_KEY")

CONFIG = {
    "api_key": API_KEY,
    "api_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "model": "qwen-3.5-32b",
    "temperature": 0.5,
    "max_tokens": 4096,
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}
