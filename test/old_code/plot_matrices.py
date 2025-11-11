import argparse
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path
import re  # 导入正则表达式模块


def get_plot_args():
    parser = argparse.ArgumentParser("Confusion Matrix Plotting Script")
    parser.add_argument("--results_dir", required=True, type=str,
                        help=".npy文件所在的目录")
    parser.add_argument("--nb_classes", required=True, type=int, help="分类类别数 (1 代表二分类)")
    parser.add_argument("--epoch", default='best', type=str,
                        help="要绘制的特定epoch编号或'best'。")
    parser.add_argument("--class_names", nargs='+', default=None,
                        help="按顺序排列的类别名称，例如: HC MDD")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="二分类时的分类阈值")
    parser.add_argument("--output_filename", type=str, default="all_confusion_matrices.png",
                        help="输出的合并图像的文件名")
    return parser.parse_args()



def main(args):
    # --- 1. Determine class names ---
    if args.class_names:
        if args.nb_classes == 1 and len(args.class_names) == 2:
            class_names = args.class_names
        else:
            assert len(args.class_names) == args.nb_classes, "Number of class names does not match nb_classes"
            class_names = args.class_names
    else:
        if args.nb_classes == 1:
            class_names = ['Negative', 'Positive']
        else:
            class_names = [str(i) for i in range(args.nb_classes)]

    # --- 2. Find all valid folds for the specified epoch ---
    results_dir = Path(args.results_dir)
    epoch_suffix = f"_epoch{args.epoch}" if args.epoch != 'best' else ""
    glob_pattern = f"predictions_fold*{epoch_suffix}.npy"
    pred_files = list(results_dir.glob(glob_pattern))
    
    valid_folds = []
    fold_regex = re.compile(r'fold(\d+)')
    for f in pred_files:
        match = fold_regex.search(f.name)
        if match:
            fold_idx = int(match.group(1))
            true_path = results_dir / f"true_labels_fold{fold_idx}{epoch_suffix}.npy"
            if true_path.exists():
                valid_folds.append(fold_idx)

    valid_folds.sort()

    if not valid_folds:
        print(f"Error: No valid 'predictions' and 'true_labels' file pairs found for epoch '{args.epoch}'.")
        print(f"Searched with pattern: {glob_pattern} in {results_dir}")
        return

    num_folds = len(valid_folds)
    print(f"Found {num_folds} valid folds for epoch '{args.epoch}': {valid_folds}")

    # --- 3. Create a large subplot grid (2 folds per row, 4 plots per row) ---
    num_rows = math.ceil(num_folds / 2)
    num_cols = 4
    
    # Increase figure width to accommodate 4 plots per row
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(22, 6 * num_rows), squeeze=False)
    fig.suptitle(f'Confusion Matrices for All Folds (Epoch: {args.epoch})', fontsize=24, y=0.98)

    # --- 4. Loop through each valid fold and plot on the correct subplot ---
    for i, fold_idx in enumerate(valid_folds):
        print(f"--- Processing Fold {fold_idx} ---")
        
        # Correctly load files using the epoch suffix
        pred_path = results_dir / f"predictions_fold{fold_idx}{epoch_suffix}.npy"
        true_path = results_dir / f"true_labels_fold{fold_idx}{epoch_suffix}.npy"

        probabilities = np.load(pred_path)
        true_labels = np.load(true_path).flatten()

        if args.nb_classes == 1:
            predicted_labels = (probabilities.flatten() >= args.threshold).astype(int)
        else:
            predicted_labels = np.argmax(probabilities, axis=1)

        # Handle cases where the test set is missing some classes
        unique_labels = sorted(list(set(true_labels) | set(predicted_labels)))
        if len(unique_labels) != len(class_names):
            print(f"Warning: Fold {fold_idx} has a mismatch between expected classes ({len(class_names)}) and actual labels found ({len(unique_labels)}). Plotting with actual labels.")
            actual_class_names = [class_names[int(label)] for label in unique_labels]
            cm_to_plot = confusion_matrix(true_labels, predicted_labels, labels=unique_labels)
        else:
            cm_to_plot = confusion_matrix(true_labels, predicted_labels)
            actual_class_names = class_names

        # --- Plotting Logic ---
        # Calculate the row and column index for the current fold's plots
        row = i // 2
        col_start = (i % 2) * 2  # Will be 0 for the first fold in a row, 2 for the second

        ax1 = axes[row, col_start]
        ax2 = axes[row, col_start + 1]

        # a) Plot the raw confusion matrix on the left subplot (ax1)
        sns.heatmap(cm_to_plot, annot=True, fmt='d', cmap='Blues',
                    xticklabels=actual_class_names, yticklabels=actual_class_names, ax=ax1, cbar=False)
        ax1.set_title(f'Fold {fold_idx} - Confusion Matrix', fontsize=14)
        ax1.set_ylabel('True Label', fontsize=12)
        ax1.set_xlabel('Predicted Label', fontsize=12)

        # b) Plot the normalized confusion matrix on the right subplot (ax2)
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_sum = cm_to_plot.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.where(cm_sum > 0, cm_to_plot.astype('float') / cm_sum, 0)

        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Greens',
                    xticklabels=actual_class_names, yticklabels=actual_class_names, ax=ax2, vmin=0, vmax=1)
        ax2.set_title(f'Fold {fold_idx} - Normalized Matrix', fontsize=14)
        ax2.set_ylabel('True Label', fontsize=12)
        ax2.set_xlabel('Predicted Label', fontsize=12)

    # --- 5. Clean up unused subplots if the number of folds is odd ---
    if num_folds % 2 != 0:
        # Turn off the last two axes in the last row
        axes[-1, -1].axis('off')
        axes[-1, -2].axis('off')

    # --- 6. Adjust layout and save the entire figure ---
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust rect to make space for the suptitle

    if args.output_filename:
        # Use the provided filename, ensuring it has a .png extension
        filename, ext = os.path.splitext(args.output_filename)
        save_path = results_dir / f"{filename}_epoch{args.epoch}.png"
    else:
        # Automatically generate a filename
        save_path = results_dir / f"all_confusion_matrices_epoch{args.epoch}.png"
        
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"\nAll confusion matrices have been combined and saved to: {save_path}")


if __name__ == "__main__":
    args = get_plot_args()
    main(args)

if __name__ == "__main__":
    args = get_plot_args()
    main(args)