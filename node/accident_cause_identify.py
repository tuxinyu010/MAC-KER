from config import *
from system_prompt import *
from agent.aci_agent import *


 # 定义事故原因识别节点和数据流向
def cause_identify_node(state: MessagesState) -> Command[Literal["accident_counterfactual_reasoner"]]:
    
    # 执行aci_agent分析
    result = aci_agent.invoke(state) # 输出包括：HumanMessage提示、 AIMessage的最终处理结果、additional_kwargs的reasoning_content推理过程773token、各类型输入和输出token值

    # 流程控制：判断是否是最终答案，如果不是，将数据传递给goto指向的agent
    goto = get_next_node(result["messages"][-1], "accident_counterfactual_reasoner")

    result["messages"][-1] = AIMessage(
        content=result["messages"][-1].content,
        name="accident_cause_identifier"
    ) 

    # 更新"messages"
    return Command(
        update={"messages": result["messages"]},
        goto=goto,
    )
