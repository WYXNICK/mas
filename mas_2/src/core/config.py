"""
核心配置模块
负责加载环境变量和定义全局常量
"""
import os
from dotenv import load_dotenv

# 加载 .env 文件中的变量
load_dotenv()


class Config:
    """全局配置类"""
    
    # API 配置
    API_KEY = os.getenv("OPENAI_API_KEY")
    BASE_URL = os.getenv("BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    
    # 模型配置
    MODEL_NAME = os.getenv("MODEL_NAME", "qwen-turbo")
    DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.5"))
    
    # LangChain 配置
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false").lower()
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "mas_2")


# 全局配置实例
config = Config()

