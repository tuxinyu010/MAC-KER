from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState, END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from typing import Literal
from langgraph.graph import StateGraph, START
import matplotlib.pyplot as plt
import io
from PIL import Image
import matplotlib.pyplot as plt
import re
import os

# 官方
api_key = "your api key"
base_url="https://api.deepseek.com"

llm = ChatDeepSeek(
    model="deepseek-reasoner",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key= api_key,
    base_url=base_url,
)

# 文件处理工具模块（仅读取指导理论）
with open(r"theory_guided\ACT_causes.md", "r", encoding='utf-8') as f:
    ACT_causes = f.read()

# 文件处理工具模块（仅读取指导理论）
with open(r"theory_guided\ACT_links.md", "r", encoding='utf-8') as f:
    ACT_links = f.read()

# 文件处理工具模块（仅读取分类标准）
categories_result=[]
with open(r"theory_guided\category_result.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        category2 = line.strip()
        if category2:  # 确保非空行
            categories_result.append(category2)
categories_result='\n'.join(categories_result)

# 事故案例读取
# 定义文件夹路径
source_dir = r"data"
output_dir = r"result"

# 确保输出文件夹存在
os.makedirs(output_dir, exist_ok=True)

# 获取源文件夹中所有txt文件（按文件名排序）
txt_files = sorted(
    [f for f in os.listdir(source_dir) if f.endswith(".txt")],
    key=lambda x: os.path.splitext(x)[0]  # 按文件名（不含扩展名）排序
)

