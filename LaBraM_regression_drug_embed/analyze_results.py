import argparse
import json
import os
import pandas as pd

def aggregate_regression_cv_results(args):

    dataset = args.dataset
    
    print(f"\n--- 开始处理回归实验: {dataset} ---")
    

    BASE_OUTPUT_DIR = os.path.join(args.base_dir, dataset)
    print(f"正在从以下目录读取数据: {BASE_OUTPUT_DIR}")
    
    all_fold_metrics = []
    # 遍历10个fold
    for fold in range(10):
        json_path = os.path.join(BASE_OUTPUT_DIR, f"fold{fold}", "best_metrics.json")
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                processed_data = {key.replace('/', '_'): value for key, value in data.items()}
                processed_data['fold'] = fold
                all_fold_metrics.append(processed_data)
                print(f"成功加载并处理: {json_path}")

            except Exception as e:
                print(f"错误: 无法读取或解析文件 {json_path}: {e}")
        else:
            print(f"警告: 找不到文件 {json_path}，已跳过。")

    if not all_fold_metrics:
        print(f"错误：在路径 {BASE_OUTPUT_DIR} 下未能加载任何结果文件。请检查路径或实验名称。")
        return

    # 转换为DataFrame进行分析
    results_df = pd.DataFrame(all_fold_metrics)
    
    # 调整列顺序
    for col_name in ['best_epoch', 'fold']:
        if col_name in results_df.columns:
            cols = results_df.columns.tolist()
            cols.insert(0, cols.pop(cols.index(col_name)))
            results_df = results_df.reindex(columns=cols)

    print("\n\n--- 10折交叉验证全部指标汇总 (每一折) ---")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
        print(results_df)

    # 1. 计算所有指标的平均值和标准差
    cols_to_summarize = [col for col in results_df.columns if col not in ['fold', 'best_epoch']]
    summary_df = results_df[cols_to_summarize].agg(['mean', 'std']).transpose()
    
    print("\n\n--- 最终所有指标的平均性能 (Mean & Std) ---")
    print(summary_df.to_string())

    # 2. 筛选并保存高效折
    PEARSON_THRESHOLD = 0.1 
    print(f"\n\n--- 正在筛选并保存高效折 (总 val_Pearson > {PEARSON_THRESHOLD}) 的数据 ---")
    
    high_perf_folds_df = results_df[results_df['val_Pearson'] > PEARSON_THRESHOLD].copy()

    output_dir_for_saving = os.path.join("./final_analysis_results/", dataset)
    os.makedirs(output_dir_for_saving, exist_ok=True)

    if high_perf_folds_df.empty:
        print(f"当前实验中没有找到总 val_Pearson > {PEARSON_THRESHOLD} 的折。")
    else:
        # 保存高效折的详细数据
        high_perf_path = os.path.join(output_dir_for_saving, "high_performance_folds_details.csv")
        high_perf_folds_df.to_csv(high_perf_path, index=False, float_format='%.4f')
        print(f"成功！{len(high_perf_folds_df)}个高效折的数据已保存至: {high_perf_path}")

        # 3. 对高效折数据求平均，并追加到总览文件
        print("\n--- 正在更新高效折平均性能汇总文件 ---")
        cols_to_average = [col for col in high_perf_folds_df.columns if col not in ['fold', 'best_epoch']]
        average_high_perf = high_perf_folds_df[cols_to_average].mean()
        summary_row = average_high_perf.to_frame().T
        summary_row.insert(0, 'dataset', dataset)
        summary_csv_path = "./final_analysis_results/regression_high_perf_AVERAGE_summary.csv"
        
        header = not os.path.exists(summary_csv_path)
        summary_row.to_csv(summary_csv_path, mode='a', header=header, index=False, float_format='%.4f')
        print(f"成功！高效折平均性能已更新至: {summary_csv_path}")

    # 4. 保存本次实验的完整结果
    details_path = os.path.join(output_dir_for_saving, "final_cv_details_all_metrics.csv")
    summary_path = os.path.join(output_dir_for_saving, "final_cv_summary_all_metrics.csv")
    results_df.to_csv(details_path, index=False, float_format='%.4f')
    summary_df.to_csv(summary_path, float_format='%.4f')
    
    print(f"\n本次实验的10折详细结果已保存至: {details_path}")
    print(f"本次实验的10折汇总结果已保存至: {summary_path}")
    print(f"--- 完成处理实验: {dataset} ---\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Regression Results Analyzer", add_help=False)

    parser.add_argument(
        '--base_dir', 
        type=str, 
        default='/data1/wangkuiyu/LEM_CRR/LaBrain_drug_embed_regression/checkpoints/',
        help='存放所有实验结果的根目录'
    )
  
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True, 
        help='"dataset" in run_labram'
    )
    args = parser.parse_args()
    aggregate_regression_cv_results(args)