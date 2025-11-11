# analyze_consistency.py

import pandas as pd
import os
import glob
from collections import Counter


def analyze_prediction_consistency(
        directory='.',
        file_pattern='result_fold*.tsv',
        template_file='participants.tsv',
        output_file='participants_with_consistency.tsv'
):
    """
    聚合10折交叉验证的结果，统计每个被试被预测为'MDD'和'HC'的次数，
    并将这些计数值添加到模板文件中。
    """
    # 1. 找到所有匹配的结果文件
    file_paths = glob.glob(os.path.join(directory, file_pattern))
    if not file_paths:
        print(f"错误: 在目录 '{directory}' 中找不到任何匹配 '{file_pattern}' 的文件。")
        return

    print(f"找到了 {len(file_paths)} 个结果文件进行分析:\n" + "\n".join(file_paths))

    # 2. 读取所有文件并将预测结果存储在字典中
    # 结构: {'sub-1': ['MDD', 'MDD', 'HC', ...], 'sub-10': ['HC', 'HC', 'HC', ...]}
    all_predictions = {}

    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, sep='\t')
            for index, row in df.iterrows():
                participant_id = row['participant_id']
                diagnosis = row['diagnosis']

                if participant_id not in all_predictions:
                    all_predictions[participant_id] = []

                if pd.notna(diagnosis) and diagnosis in ['MDD', 'HC']:
                    all_predictions[participant_id].append(diagnosis)
        except Exception as e:
            print(f"读取或处理文件 {file_path} 时出错: {e}")

    # 3. 为每个被试统计 'MDD' 和 'HC' 的票数
    print("\n正在为每个被试统计预测票数...")
    consistency_counts = {}
    for participant_id, preds in all_predictions.items():
        counts = Counter(preds)
        consistency_counts[participant_id] = {
            'MDD_votes': counts.get('MDD', 0),  # 使用 .get(key, 0) 来处理某个类别票数为0的情况
            'HC_votes': counts.get('HC', 0)
        }

    print("统计完成。")

    # 4. 读取模板文件，并添加新的统计列
    try:
        # 使用模板文件作为基础
        final_df = pd.read_csv(template_file, sep='\t')
    except FileNotFoundError:
        print(f"错误: 模板文件 '{template_file}' 未找到。")
        return

    # 5. 将统计结果映射到 DataFrame 的新列
    # 我们使用 .map() 函数两次，一次用于MDD票数，一次用于HC票数

    # 提取每个被试的MDD票数
    mdd_votes_map = {pid: counts['MDD_votes'] for pid, counts in consistency_counts.items()}
    # 提取每个被试的HC票数
    hc_votes_map = {pid: counts['HC_votes'] for pid, counts in consistency_counts.items()}

    # 将票数映射到新列
    final_df['MDD_votes'] = final_df['participant_id'].map(mdd_votes_map)
    final_df['HC_votes'] = final_df['participant_id'].map(hc_votes_map)

    # 对于模板中存在但结果文件中没有的被试，将票数填充为 0
    final_df['MDD_votes'].fillna(0, inplace=True)
    final_df['HC_votes'].fillna(0, inplace=True)

    # 将浮点数转换为整数
    final_df['MDD_votes'] = final_df['MDD_votes'].astype(int)
    final_df['HC_votes'] = final_df['HC_votes'].astype(int)

    # 6. 保存最终的TSV文件
    final_df.to_csv(output_file, sep='\t', index=False)
    print(f"\n分析完成！带有预测一致性统计的结果已保存至: {output_file}")
    print("\n文件预览:")
    print(final_df.head())




# apply_threshold.py

import pandas as pd
import os
import glob
from collections import Counter

def apply_hc_threshold(
    directory='.', 
    file_pattern='result_fold*.tsv', 
    template_file='participants.tsv',
    output_file='participants_with_threshold_diagnosis.tsv',
    hc_threshold=3
):
    """
    聚合10折结果，应用一个阈值来确定最终诊断，并更新模板文件。
    
    规则: 如果 HC 票数 >= hc_threshold，则诊断为 HC，否则为 MDD。
    """
    # 1. 找到所有匹配的结果文件
    file_paths = glob.glob(os.path.join(directory, file_pattern))
    if not file_paths:
        print(f"错误: 在目录 '{directory}' 中找不到任何匹配 '{file_pattern}' 的文件。")
        return

    print(f"找到了 {len(file_paths)} 个结果文件进行分析:\n" + "\n".join(file_paths))
    
    # 2. 读取所有预测结果
    all_predictions = {}
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, sep='\t')
            for index, row in df.iterrows():
                participant_id = row['participant_id']
                diagnosis = row['diagnosis']
                
                if participant_id not in all_predictions:
                    all_predictions[participant_id] = []
                
                if pd.notna(diagnosis) and diagnosis in ['MDD', 'HC']:
                    all_predictions[participant_id].append(diagnosis)
        except Exception as e:
            print(f"读取或处理文件 {file_path} 时出错: {e}")

    # 3. 为每个被试统计 HC 票数并应用阈值规则
    print("\n" + "="*40)
    print(f"应用诊断规则: 如果 HC 票数 >= {hc_threshold}，则为 HC，否则为 MDD")
    print("="*40)

    final_diagnosis = {}
    for participant_id, preds in all_predictions.items():
        if preds:
            # 只统计 'HC' 的票数
            hc_votes = Counter(preds).get('HC', 0)
            
            # 应用阈值规则
            if hc_votes >= hc_threshold:
                final_diagnosis[participant_id] = 'HC'
            else:
                final_diagnosis[participant_id] = 'MDD'
        else:
            final_diagnosis[participant_id] = 'n/a' # 如果某个被试没有任何预测

    # 统计最终诊断结果
    final_mdd_count = list(final_diagnosis.values()).count('MDD')
    final_hc_count = list(final_diagnosis.values()).count('HC')
    total_subjects = len(final_diagnosis)

    print(f"总被试数: {total_subjects}")
    print(f"最终诊断为 'MDD' 的被试数: {final_mdd_count}")
    print(f"最终诊断为 'HC' 的被试数:  {final_hc_count}")

    # 4. 读取模板文件并添加新的 'diagnosis' 列
    try:
        final_df = pd.read_csv(template_file, sep='\t')
    except FileNotFoundError:
        print(f"错误: 模板文件 '{template_file}' 未找到。")
        return

    # 5. 将最终诊断结果映射到 DataFrame
    final_df['diagnosis'] = final_df['participant_id'].map(final_diagnosis)
    
    # 对于模板中存在但结果文件中没有的被试，填充 'n/a'
    final_df['diagnosis'].fillna('n/a', inplace=True)

    # 6. 保存最终的TSV文件
    final_df.to_csv(output_file, sep='\t', index=False)
    print(f"\n分析完成！应用阈值后的最终结果已保存至: {output_file}")
    print("\n文件预览:")
    print(final_df.head())


if __name__ == "__main__":
    # 您可以在这里修改文件名，如果需要的话
    analyze_prediction_consistency(
        directory='.',
        file_pattern='result_fold*.tsv',
        template_file='get_res/participants.tsv',
        output_file='get_res/participants_with_consistency_votes.tsv'
    )
    apply_hc_threshold(hc_threshold=2)