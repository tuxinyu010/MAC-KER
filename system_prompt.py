from config import *
#from system_prompt import *

#构建多AI助手协作的系统提示模板
#suffix -- 可追加的自定义提示后缀，用于添加场景特定的指示
def make_system_prompt(suffix: str) -> str:
    return (
       # 协作基础提示
       "你是一个电力安全领域专家，与其他助手合作。"
       # 渐进式处理说明
       "如果你不能完全回答，没关系，另一个助手会基于你的输出结果进一步回答。"
       # 自定义提示追加
        f"\n{suffix}"
    )

#工作流控制函数 - 决定下一步执行节点
#参数：last_message -- 上一个节点的输出消息;goto -- 默认的下一个节点标识
def get_next_node(last_message: BaseMessage, goto: str):

            # 终止条件判断：任何代理发出完成信号
            # 增强终止条件检测
            termination_signals = [
                "最终三元组",
                "链路校验通过",
                "校验后完整三元组为"
            ]
            
            # 检查所有终止信号
            if any(signal in last_message.content for signal in termination_signals):
                return END

            # 继续默认流程
            return goto

# 在所有节点函数中添加消息清理逻辑
def clean_messages(state: MessagesState, current_content: str) -> list:
    """保留关键消息：初始输入 + 当前节点输出 + 必要反馈"""
    return [
        state["messages"][0],  # 保留初始输入
        AIMessage(content=current_content, name="current_node_output")
    ]

# 特殊清理函数 - 根据节点类型调整保留消息
def clean_messages1(state: MessagesState, current_content: str, is_final=False) -> list:
            """
            精简消息历史，只保留关键信息：
            - 非最终节点：保留初始输入 + 当前节点输出
            - 最终节点：仅保留最终三元组和终止确认
            """
            # 保留初始系统提示
            keep_messages = [state["messages"][0]]
            
            if is_final:
                # 最终节点：只保留三元组和终止确认
                final_triples = re.search(r"校验后完整三元组为:(.*?)\n", current_content)
                if final_triples:
                    # 创建精简的最终结果消息
                    keep_messages.append(AIMessage(
                        content=f"最终三元组:\n{final_triples.group(1).strip()}",
                        name="final_causal_links"
                    ))
                keep_messages.append(AIMessage(
                    content="## 实验终止确认 ##\n所有校验已完成",
                    name="termination_confirmation"
                ))
            else:
                # 非最终节点：保留当前节点输出
                keep_messages.append(AIMessage(
                    content=current_content,
                    name="current_output"
                ))
            
            return keep_messages