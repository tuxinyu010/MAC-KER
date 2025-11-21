from config import *
from system_prompt import *
from agent.clr_agent import *

# 因果链路生成节点
def causal_link_reasoning_node(state: MessagesState) -> Command[Literal["logic_verificater"]]:

     # 构造新的用户提示（将前序结果作为输入）
    new_prompt = f"""根据上一步骤识别出的事故原因识别直接因果关系和并列关系，请严格按照输出格式要求生成三元组："""
    
    # 创建新的用户消息
    new_human_message = HumanMessage(content=new_prompt)
    
    # 更新消息序列（保留历史消息并追加新提示）
    updated_state = state.copy()
    updated_state["messages"].append(new_human_message)

    # 执行因果推理
    result = clr_agent.invoke(updated_state)

    # 流程控制（保持原有逻辑）
    goto = get_next_node(result["messages"][-1], "logic_verificater")
    
    # 更新消息
    result["messages"][-1] = AIMessage(
        content=result["messages"][-1].content,
        name="causal_link_reasoner"
    )

    return Command(
        update={
            "messages": result["messages"]},
        goto=goto,
    )
