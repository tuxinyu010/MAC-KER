from config import *
from system_prompt import *
from agent.acrt_agent import *

# 反事实推理校验节点
def accident_counterfactual_reasoning_node(state: MessagesState) -> Command[Literal["accident_cause_identifier","causal_link_reasoner"]]:

    new_prompt = f"""根据上一步骤识别出的事故原因，分析检验直接原因与事故类型之间的因果稳健性"""
    
    # 创建新的用户消息
    new_human_message = HumanMessage(content=new_prompt)
    
    # 更新消息序列（保留历史消息并追加新提示）
    updated_state = state.copy()
    updated_state["messages"].append(new_human_message)

    # 执行反事实推理
    result = acrt_agent.invoke(updated_state)
    
    # 获取最后一个消息内容 (关键修复)
    last_message = result["messages"][-1]
    content = str(last_message.content)  # 确保转换为字符串
    
    # 流程控制（保持原有逻辑）
    if "验证通过" in content:
        goto = get_next_node(last_message, "causal_link_reasoner")
    else:
        goto = get_next_node(last_message, "accident_cause_identifier")
    
    # 重试计数
    state["retry_count"] = state.get("retry_count", 0) + 1
    if state["retry_count"] > 3: 
        goto = "causal_link_reasoner"  # 强制终止
    
    # 提取修正建议（如果需要）
    if "验证不通过" in content:
        feedback_match = re.search(r"修正建议：(.*?)(?:\n|$)", content)
        feedback = feedback_match.group(1) if feedback_match else "请检查直接原因必要性，即排除单个原因，事故不发生或可以显著消除事故后果的严重程度"
        
        # 创建结构化反馈消息
        feedback_msg = HumanMessage(
            content=f"## 修正指令\n{feedback}\n\n## 原始事故案例\n{accident_data}",
            name="counterfactual_feedback"
        )
        
        # 清理历史+保留反馈
        cleaned_messages = clean_messages(state, "")
        cleaned_messages.append(feedback_msg)
        
        return Command(
            update={"messages": cleaned_messages},
            goto=goto
        )
    
    # 验证通过的情况
    # 更新消息
    result["messages"][-1] = AIMessage(
        content=content,
        name="accident_counterfactual_reasoner"
    )
    
    return Command(
        update={"messages": result["messages"]},
        goto=goto,
    )






