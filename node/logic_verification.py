from config import *
from system_prompt import *
from agent.lvt_agent import *

# 逻辑一致性校验节点
def logic_verification_node(state: MessagesState) -> Command[Literal["causal_link_reasoner",END]]:
            # 构造精简提示
            new_prompt = "进行因果链路逻辑一致性校验（只输出结论）"
            new_human_message = HumanMessage(content=new_prompt)
            
            # 更新消息序列（保留历史消息并追加新提示）
            updated_state = state.copy()
            updated_state["messages"].append(new_human_message)

            # 执行因果推理
            result = lvt_agent.invoke(updated_state)
            last_message = result["messages"][-1]
            content = last_message.content
            
            # 流程控制
            if "链路校验通过" in content:
                # 提取三元组内容
                triples_match = re.search(r"校验后完整三元组为:(.*)", content, re.DOTALL)
                triples_content = triples_match.group(1).strip() if triples_match else "未提取到三元组"
                
                # 创建最终精简输出
                final_output = (
                    "## 最终验证结果 ##\n"
                    "链路校验通过\n"
                    f"最终三元组:\n{triples_content}\n"
                    "## 实验终止确认 ##"
                )
                
                # 只保留最终结果
                cleaned_messages1 = [
                    AIMessage(content=final_output, name="final_result")
                ]

                return Command(
                    update={"messages": cleaned_messages1},
                    goto=END
                )
            
            # 验证失败处理
            cleaned_messages1 = clean_messages1(state, content)
            
            # 重试计数
            state["retry_count"] = state.get("retry_count", 0) + 1
            goto = "causal_link_reasoner"
            
            if state["retry_count"] > 5:
                # 超过重试次数，强制终止并输出当前结果
                final_output = (
                    "## 最终验证结果 ##\n"
                    "警告：超过最大重试次数\n"
                    f"当前三元组:\n{content}\n"
                    "## 实验终止确认 ##"
                )
                cleaned_messages1 = [
                    AIMessage(content=final_output, name="final_result")
                ]
                return Command(
                    update={"messages": cleaned_messages1},
                    goto=END
                )
            
            return Command(
                update={"messages": cleaned_messages1},
                goto=goto,
            )




