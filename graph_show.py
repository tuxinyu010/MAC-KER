from config import *
from system_prompt import *
from agent.aci_agent import *
from agent.acrt_agent import *
from node.accident_cause_identify import *
from node.accident_counterfactual_reasoning import *
from node.causal_link_reasoning import *
from node.logic_verification import *
from graph import *

#绘制整体流程图
# 获取图形描述数据（假设返回二进制PNG数据）
# 获取图形数据（假设返回二进制 PNG 数据）
graph_data = graph.get_graph().draw_mermaid_png()

# 使用 Matplotlib 显示图像
image = Image.open(io.BytesIO(graph_data))
plt.figure(figsize=(10, 6))
plt.axis('off')
plt.imshow(image)
plt.show()
