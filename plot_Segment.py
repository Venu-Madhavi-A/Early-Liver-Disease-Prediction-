import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import cv2 as cv
from Segmentation_Evaluation import Segmentation_Evaluation

def Statistical(val):
    out = np.zeros((5))
    out[0] = max(val)
    out[1] = min(val)
    out[2] = np.mean(val)
    out[3] = np.median(val)
    out[4] = np.std(val)
    return out

no_of_dataset = 2


def Plot_seg_Results():
    for a in range(no_of_dataset):
        # Ensure TkAgg backend is used for interactive plots
        matplotlib.use('TkAgg')

        # Load evaluation data
        eval_data = np.load('Eval_seg.npy', allow_pickle=True)[a]

        # Define the terms for statistical analysis
        Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'Dice', 'Jaccard']

        # Extract values from eval_data for statistical analysis
        value = eval_data[:, :, :]

        # Initialize an array to store statistical results (Best, Worst, Mean, Median, Std)
        stat = np.zeros((value.shape[1], value.shape[2], 5))

        # Loop through algorithms (rows) and metrics (columns)
        for j in range(value.shape[1]):  # For each algorithm
            for k in range(value.shape[2]):  # For each metric (Accuracy, Sensitivity, etc.)
                # Calculate the statistics for each algorithm and metric
                stat[j, k, :] = Statistical(value[:, j, k])  # Assuming `Statistical` is defined

        # Plot each of the metrics (Terms)
        for k in range(len(Terms)):
            fig, ax = plt.subplots(figsize=(10, 6))  # Larger figure for better readability
            X = np.arange(5)  # X-axis positions (Best, Worst, Mean, Median, Std)

            # Plot bar charts for each model with a slight offset to separate bars
            ax.bar(X + 0.00, stat[0, k, :], color='#f97306', width=0.10, label="VGG16")
            ax.bar(X + 0.10, stat[1, k, :], color='#cc3f81', width=0.10, label="GRADIENT_BOOSTING")
            ax.bar(X + 0.20, stat[2, k, :], color='#ccbc3f', width=0.10, label="LDA")
            ax.bar(X + 0.30, stat[3, k, :], color='c', width=0.10, label="LSA")
            ax.bar(X + 0.40, stat[4, k, :], color='k', width=0.10, label="IBEOSA-A-LSA")

            # Set X-axis labels, Y-axis label, and title
            ax.set_xticks(X + 0.20)  # Center the ticks between bars
            ax.set_xticklabels(['Best', 'Worst', 'Mean', 'Median', 'Std'], fontsize=12)
            ax.set_ylabel(Terms[k], fontsize=14, fontweight='bold')
            ax.set_xlabel('Statistical Analysis', fontsize=14, fontweight='bold')
            # ax.set_title(f"{Terms[k]} Comparison for Dataset {a + 1}", fontsize=16, fontweight='bold')

            # Add the legend
            ax.legend(loc='upper left', fontsize=12)

            # Tight layout for better spacing
            plt.tight_layout()

            # Save the figure with a high resolution (300 dpi)
            plt.savefig(f"./Results/Segmented_Image_bar_Dataset_{a + 1}_{Terms[k]}.png", dpi=300)
            plt.show()

if __name__ == '__main__':
    Plot_seg_Results()