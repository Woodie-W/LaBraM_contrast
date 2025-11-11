import re


def find_max_accuracy(file_path):
    # 正则表达式模式
    pattern = r'"val_balanced_accuracy":\s*([\d.]+)'
    max_accuracy = 0

    try:
        with open(file_path, 'r') as file:
            content = file.read()
            # 找出所有匹配项
            matches = re.findall(pattern, content)

            # 将字符串转换为浮点数并找出最大值
            if matches:
                accuracies = [float(x) for x in matches]
                max_accuracy = max(accuracies)
                print(f"找到 {len(matches)} 个值")
                print(f"最大的 val_balanced_accuracy 是: {max_accuracy}")
            else:
                print("没有找到匹配的值")


    except FileNotFoundError:
        print("找不到文件")
    except Exception as e:
        print(f"发生错误: {e}")

    print("所有的值：")
    for i, acc in enumerate(accuracies, 1):
        print(f"{i}. {acc}")
    return max_accuracy


# 使用示例
# file_path = "checkpoints/finetune_tdbrain1215_base/fold9/log.txt"  # 替换为你的文件路径
all_accuracy = []
for i in range(10):
    max_accuracy = find_max_accuracy(f'checkpoints/finetune_tdbrain1217_td_loadmodel_base/fold{i}/log.txt')
    all_accuracy.append(max_accuracy)
    print(max_accuracy)
print("all_accuracy", all_accuracy)
mean_accuracy = sum(all_accuracy) / len(all_accuracy)
print("mean_accuracy", mean_accuracy)
# max_val = find_max_accuracy(file_path)