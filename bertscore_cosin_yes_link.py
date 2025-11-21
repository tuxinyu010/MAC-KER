import numpy as np
from bert_score import score
import matplotlib.pyplot as plt
import pandas as pd

def split_causes(line):
    """智能分割原因文本"""
    # 尝试按真正的换行符分割
    causes = [c.strip() for c in line.splitlines() if c.strip()]
    
    # 如果没有找到真正的换行符，尝试按字面\n分割
    if len(causes) <= 1 and r'\n' in line:
        causes = [c.strip() for c in line.split(r'\n') if c.strip()]
    
    # 如果还是没有分割成功，尝试按分号分割
    if len(causes) <= 1 and ';' in line:
        causes = [c.strip() for c in line.split(';') if c.strip()]
    
    return causes

def evaluate_causes(ref_line, cand_line, threshold=0.6):
    """
    评估单行事故原因识别的准确性（基于阈值二元分类）
    
    参数:
    ref_line: 标准数据集中的一行 (包含多个原因)
    cand_line: 待评估数据集中的一行 (包含多个识别原因)
    threshold: 相似度阈值 (默认0.6)
    
    返回: 包含precision, recall, f1和匹配对信息的字典
    """
    ref_causes = split_causes(ref_line)
    cand_causes = split_causes(cand_line)
    
    # 特殊情况处理
    if not ref_causes and not cand_causes:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}
    if not ref_causes:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "tp": 0, "fp": len(cand_causes), "fn": 0}
    if not cand_causes:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": len(ref_causes)}

    # 2. 构建相似度矩阵
    sim_matrix = []
    for c_cause in cand_causes:
        row_scores = []
        for r_cause in ref_causes:
            # 计算单个原因对的BERTScore F1值
            _, _, F1 = score([c_cause], [r_cause], 
                            lang='zh',                # 语言：'en'英文, 'zh'中文
                            model_type='casual_link_mining/model/bert-base-chinese',  # 模型类型
                            num_layers=9,             # 使用哪一层输出（默认使用最佳层）
                            batch_size=32,            # 批处理大小（调整以控制内存）
                            nthreads=4,               # 线程数
                            rescale_with_baseline=True,  # 是否标准化chinese分数
                            verbose=False)            # 是否显示详细输出
            row_scores.append(F1.item())  # 转换为Python浮点数
        sim_matrix.append(row_scores)
    
    # 3. 基于阈值的二元分类
    tp = 0  # 真正例
    fp = 0  # 假正例
    fn = 0  # 假反例
    
     #3. 基于最大相似度的匹配统计
    matched_refs = set()  # 已匹配的标准原因索引
    matched_cands = set() # 已匹配的候选原因索引
    
    # 第一步：为每个标准原因找到最佳匹配
    for j in range(len(ref_causes)):
        best_sim = 0
        best_cand_idx = -1
        
        # 在当前标准原因列中找最高相似度
        for i in range(len(cand_causes)):
            sim = sim_matrix[i][j]
            if sim > best_sim:
                best_sim = sim
                best_cand_idx = i
                
        # 检查是否满足匹配条件
        if best_sim >= threshold and best_cand_idx not in matched_cands:
            tp += 1
            matched_refs.add(j)
            matched_cands.add(best_cand_idx)
    
    # 第二步：统计未匹配项
    fp = len(cand_causes) - len(matched_cands)  # 冗余候选原因
    fn = len(ref_causes) - len(matched_refs)    # 未被覆盖的标准原因
    
    # 4. 计算指标
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ref_count": len(ref_causes),
        "cand_count": len(cand_causes)
    }

# 主评估函数
def main_evaluation(ref_file, cand_file, threshold=0.6):
    """
    整体评估函数
    
    参数:
    ref_file: 标准数据集文件路径
    cand_file: 待评估数据集文件路径
    threshold: 相似度阈值
    """
    # 存储所有样本的结果
    all_precision = []
    all_recall = []
    all_f1 = []
    
    # 微平均统计量
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_ref_count = 0
    total_cand_count = 0
    
    # 打开日志文件
    log_filename = f"evaluation_threshold_{threshold}.txt"
    with open(log_filename, "w", encoding="utf-8") as log_file:
        # 写入文件头信息
        header = (
            f"评估报告 (阈值={threshold})\n"
            f"标准文件: {ref_file}\n"
            f"待评估文件: {cand_file}\n"
            f"{'='*50}\n"
        )
        print(header)
        log_file.write(header)
        
        with open(ref_file, 'r', encoding='utf-8') as f_ref, \
             open(cand_file, 'r', encoding='utf-8') as f_cand:
            
            # 逐行处理两个文件
            for line_idx, (ref_line, cand_line) in enumerate(zip(f_ref, f_cand)):
                # 评估当前行
                result = evaluate_causes(ref_line, cand_line, threshold)
                
                # 收集结果
                all_precision.append(result["precision"])
                all_recall.append(result["recall"])
                all_f1.append(result["f1"])
                
                # 更新微平均统计量
                total_tp += result["tp"]
                total_fp += result["fp"]
                total_fn += result["fn"]
                total_ref_count += result["ref_count"]
                total_cand_count += result["cand_count"]
                
                # 构建当前行结果信息
                line_result = (
                    f"行 {line_idx+1}: "
                    f"精确率={result['precision']:.3f}, "
                    f"召回率={result['recall']:.3f}, "
                    f"F1={result['f1']:.3f}, "
                    f"TP={result['tp']}, FP={result['fp']}, FN={result['fn']}\n"
                )
                
                # 打印到控制台并保存到文件
                print(line_result, end='')
                log_file.write(line_result)
        
        # ================= 计算宏平均 =================
        macro_precision = np.mean(all_precision)
        macro_recall = np.mean(all_recall)
        macro_f1 = np.mean(all_f1)
        
        # ================= 计算微平均 =================
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
        
        # ================= 构建最终结果信息 =================
        summary = (
            "\n===== 整体评估结果 =====\n"
            f"评估样本数: {len(all_f1)}\n"
            f"标准原因总数: {total_ref_count}\n"
            f"识别原因总数: {total_cand_count}\n\n"
            
            "===== 宏平均 =====\n"
            f"宏平均精确率: {macro_precision:.3f}\n"
            f"宏平均召回率: {macro_recall:.3f}\n"
            f"宏平均F1值: {macro_f1:.3f}\n\n"
            
            "===== 微平均 =====\n"
            f"微平均精确率: {micro_precision:.3f}\n"
            f"微平均召回率: {micro_recall:.3f}\n"
            f"微平均F1值: {micro_f1:.3f}\n\n"
            
            "===== 混淆矩阵汇总 =====\n"
            f"真正例(TP): {total_tp:.0f}\n"
            f"假正例(FP): {total_fp:.0f}\n"
            f"假反例(FN): {total_fn:.0f}\n"
            f"精确率 = TP/(TP+FP) = {total_tp:.0f}/({total_tp:.0f}+{total_fp:.0f}) = {micro_precision:.3f}\n"
            f"召回率 = TP/(TP+FN) = {total_tp:.0f}/({total_tp:.0f}+{total_fn:.0f}) = {micro_recall:.3f}\n"
        )
        
        # 打印最终结果并保存到文件
        print(summary)
        log_file.write(summary)
    
   
    
    # ================= 生成详细结果表格 =================
    results_df = pd.DataFrame({
        'Precision': all_precision,
        'Recall': all_recall,
        'F1': all_f1,
        'Ref_Count': [len(line.split('\n')) for line in open(ref_file, 'r', encoding='utf-8')],
        'Cand_Count': [len(line.split('\n')) for line in open(cand_file, 'r', encoding='utf-8')]
    })
    
    # 保存为CSV文件
    csv_filename = f'casual_link_mining/evaluate/results/detailed_results_qwen_prompt_threshold_{threshold}.csv'
    results_df.to_csv(csv_filename, index_label='Line_Index')
    print(f"\n详细结果已保存到 '{csv_filename}'")
    
    return {
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn
    }

# 使用示例
if __name__ == "__main__":
    # 文件路径设置
    reference_file = "casual_link_mining\evaluate\data_score\links\因果链路-标准数据集.txt"  # 标准数据集
    candidate_file = "casual_link_mining\evaluate\data_score\links\因果链路-qwen.txt"  # 待评估数据集
    threshold = 0.6  # 二元分类阈值
    
    # 执行评估
    results = main_evaluation(reference_file, candidate_file, threshold)
    
    # 打印宏平均和微平均结果
    print("\n最终评估摘要:")
    print(f"阈值: {threshold}")
    print(f"宏平均精确率: {results['macro_precision']:.3f}")
    print(f"宏平均召回率: {results['macro_recall']:.3f}")
    print(f"宏平均F1: {results['macro_f1']:.3f}")
    print(f"微平均精确率: {results['micro_precision']:.3f}")
    print(f"微平均召回率: {results['micro_recall']:.3f}")
    print(f"微平均F1: {results['micro_f1']:.3f}")
