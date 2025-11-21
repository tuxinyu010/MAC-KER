from config import *
from system_prompt import *
from agent.aci_agent import *
from agent.acrt_agent import *
from node.accident_cause_identify import *
from node.accident_counterfactual_reasoning import *
from node.causal_link_reasoning import *
from node.logic_verification import *

# 配置完整工作流
 # 配置完整工作流
workflow = StateGraph(MessagesState)
workflow.add_node("accident_cause_identifier", cause_identify_node)
workflow.add_node("accident_counterfactual_reasoner", accident_counterfactual_reasoning_node)
workflow.add_node("causal_link_reasoner", causal_link_reasoning_node)
workflow.add_node("logic_verificater", logic_verification_node)

# 配置流转路径
workflow.add_edge(START, "accident_cause_identifier")
graph = workflow.compile()

