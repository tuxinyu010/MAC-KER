import sys
from config import *
from system_prompt import *
from agent.aci_agent import *
from agent.acrt_agent import *
from agent.clr_agent import *
from agent.lvt_agent import *
from node.accident_cause_identify import *
from node.accident_counterfactual_reasoning import *
from node.causal_link_reasoning import *
from node.logic_verification import *
from graph import *
from graph_show import *

events = graph.stream(
            {
                "messages": [
                    HumanMessage( 
                        content=
                        "只有完成了所有同伴的任务传递，才可以结束。结束时输出：“实验结束”",
                    )
                ],
            },
            # Maximum number of steps to take in the graph
            {"recursion_limit": 1500},
        )

for s in events:
    # 如果事件是列表，将其转换为字符串
    if isinstance(s, list):
        # 将列表中的元素连接成字符串
        s = " | ".join(str(item) for item in s)
    elif not isinstance(s, str):
        # 如果不是字符串也不是列表，转换为字符串
        s = str(s)
    
    # 现在s保证是字符串
    output_content += s + "\n"
    output_content += "----\n"
    
    
# 写入输出文件
output_filename = os.path.splitext(filename)[0] + "_print.txt"
output_path = os.path.join(output_dir, output_filename)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(output_content)

