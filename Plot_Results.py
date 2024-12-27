from itertools import cycle
import pandas as pd
import seaborn as sn
import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import roc_curve


def statistical_analysis(v):
    a = np.zeros((5))
    a[0] = np.min(v)
    a[1] = np.max(v)
    a[2] = np.mean(v)
    a[3] = np.median(v)
    a[4] = np.std(v)
    return a


def Plot_Results():
    for i in range(2):
        Eval = np.load('Eval_all.npy', allow_pickle=True)[i]
        Terms = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
        Graph_Term = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        Algorithm = ['TERMS', 'CSO', 'AOA', 'AGTO', 'ESOA', 'IBEOSA']
        Classifier = ['TERMS', 'RESNET', 'INCEPTION', 'MOBILENET', 'DENSENET', 'ViT-DRDNet']
        value = Eval[4, :, 4:]
        value[:, :-1] = value[:, :-1] * 100
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value[j, :])
        print('--------------------------------------------------Algorithm Comparison - Dataset', i + 1,
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value[len(Algorithm) + j - 1, :])
        print('---------------------------------------------------Classifier Comparison - Dataset ', i + 1,
              '--------------------------------------------------')
        print(Table)
        Eval = np.load('Eval_all.npy', allow_pickle=True)[i]
        BATCH = [1, 2, 3, 4, 5]
        for j in range(len(Graph_Term)):
            Graph = np.zeros((Eval.shape[0], Eval.shape[1]))
            for k in range(Eval.shape[0]):
                for l in range(Eval.shape[1]):
                    if Graph_Term[j] == 9:
                        Graph[k, l] = Eval[k, l, Graph_Term[j] + 4]
                    else:
                        Graph[k, l] = Eval[k, l, Graph_Term[j] + 4]*100

            fig, ax = plt.subplots(figsize=(8, 6))
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 5], color='b', edgecolor='k', width=0.15, hatch="*", label="RESNET")
            ax.bar(X + 0.15, Graph[:, 6], color='#ef4026', edgecolor='k', width=0.15, hatch="*", label="INCEPTION")
            ax.bar(X + 0.30, Graph[:, 7], color='lime', edgecolor='k', width=0.15, hatch='*', label="MOBILENET")
            ax.bar(X + 0.45, Graph[:, 8], color='y', edgecolor='k', width=0.15, hatch="*", label="DENSENET")
            ax.bar(X + 0.60, Graph[:, 9], color='k', edgecolor='w', width=0.15, hatch="o", label="ViT-DRDNet")

            ax.set_xticks(X + 0.25)
            ax.set_xticklabels(['Linear', 'ReLU', 'Tanh', 'Softmax', 'Sigmoid'],fontweight='bold')
            ax.set_xlabel('Activation Function', fontsize=14,fontweight='bold')
            ax.set_ylabel(Terms[Graph_Term[j]], fontsize=14,fontweight='bold')
            # ax.set_title(f'{Terms[Graph_Term[j]]} - Dataset {i + 1}', fontsize=16)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True, fontsize=12)
            plt.tight_layout()
            plt.savefig(f"./Results/Dataset_{i + 1}_{Terms[Graph_Term[j]]}_Activation_Function_bar.png")
            plt.show()


def plot_results_conv():
    for a in range(2):
        conv = np.load('Fitness.npy', allow_pickle=True)[a]
        ind = np.argsort(conv[:, conv.shape[1] - 1])
        x = conv[ind[0], :].copy()
        y = conv[4, :].copy()
        conv[4, :] = x
        conv[ind[0], :] = y

        Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
        Algorithm = ['CSO', 'AOA', 'AGTO', 'EOSA', 'IBEOSA']

        Value = np.zeros((conv.shape[0], 5))
        for j in range(conv.shape[0]):
            Value[j, 0] = np.min(conv[j, :])
            Value[j, 1] = np.max(conv[j, :])
            Value[j, 2] = np.mean(conv[j, :])
            Value[j, 3] = np.median(conv[j, :])
            Value[j, 4] = np.std(conv[j, :])

        Table = PrettyTable()
        Table.add_column("ALGORITHMS", Statistics)
        for j in range(len(Algorithm)):
            Table.add_column(Algorithm[j], Value[j, :])
        print(
            f'--------------------------------------------------Dataset_{a + 1} - Statistical Analysis--------------------------------------------------')
        print(Table)

        iteration = np.arange(conv.shape[1])
        plt.figure(figsize=(10, 6))
        plt.plot(iteration, conv[0, :], color='m', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
                 label='CSO')
        plt.plot(iteration, conv[1, :], color='c', linewidth=3, marker='p', markerfacecolor='green', markersize=12,
                 label='AOA')
        plt.plot(iteration, conv[2, :], color='b', linewidth=3, marker='.', markerfacecolor='cyan', markersize=12,
                 label='AGTO')
        plt.plot(iteration, conv[3, :], color='r', linewidth=3, marker='o', markerfacecolor='magenta', markersize=12,
                 label='EOSA')
        plt.plot(iteration, conv[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black', markersize=12,
                 label='IBEOSA')

        plt.xlabel('Iteration', fontsize=14,fontweight='bold')
        plt.ylabel('Cost Function', fontsize=14,fontweight='bold')
        # plt.title(f'Convergence Plot - Dataset {a + 1}', fontsize=16)
        plt.legend(loc=1, fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.savefig(f"./Results/Conv_{a + 1}.png")
        plt.show()


def Plot_ROC():
    lw = 2
    cls = ['RESNET', 'INCEPTION', 'MOBILENET', 'DENSENET', 'ViT-DRDNet']
    colors1 = cycle(["#65fe08", "#4e0550", "#f70ffa", "#a8a495", "#004577"])

    for n in range(2):
        plt.figure(figsize=(8, 6))
        for i, color in zip(range(5), colors1):  # For all classifiers
            Predicted = np.load('roc_score.npy', allow_pickle=True)[n][i].astype('float')
            Actual = np.load('roc_act.npy', allow_pickle=True)[n][i].astype('int')
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual[:, -1], Predicted[:, -1].ravel())
            plt.plot(false_positive_rate1, true_positive_rate1, color=color, lw=lw, label="{0}".format(cls[i]))

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate", fontsize=14,fontweight='bold')
        plt.ylabel("True Positive Rate", fontsize=14,fontweight='bold')
        plt.title("ROC Curve", fontsize=16,fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"./Results/_roc_{n + 1}.png")
        plt.show()


def Confusion_matrix():
    for i in range(2):
        # Load the evaluation data from the numpy file
        Eval = np.load('Eval_all.npy', allow_pickle=True)[i]
        value = Eval[4, 4, :5]  # Extracting some values for confusion matrix

        # Actual and predicted values for the confusion matrix
        val = np.asarray([0, 1, 1])
        data = {'y_Actual': [val.ravel()],
                'y_Predicted': [np.asarray(val).ravel()]
                }
        df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])

        # Create the confusion matrix
        confusion_matrix = pd.crosstab(df['y_Actual'][0], df['y_Predicted'][0], rownames=['Actual'],
                                       colnames=['Predicted'])

        # Ensure value is of integer type
        value = value.astype('int')

        # Populate the confusion matrix with extracted values
        confusion_matrix.values[0, 0] = value[1]  # True Negative
        confusion_matrix.values[0, 1] = value[3]  # False Positive
        confusion_matrix.values[1, 0] = value[2]  # False Negative
        confusion_matrix.values[1, 1] = value[0]  # True Positive

        # Create the confusion matrix heatmap
        plt.figure(figsize=(10, 8))  # Increase figure size
        ax = sn.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                        annot_kws={"size": 18, "weight": 'bold'}, linewidths=0.5, linecolor='black')

        # Set axis labels and title
        ax.set_xlabel('Predicted', fontsize=14, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=14, fontweight='bold')
        ax.set_title(f'Confusion Matrix (Accuracy = {Eval[4, 4, 4] * 100:.2f}%)', fontsize=16, fontweight='bold')

        # Tight layout to ensure proper spacing
        plt.tight_layout()

        # Save the figure with higher resolution
        plt.savefig(f'./Results/Confusion_{i + 1}.png', dpi=300)
        plt.show()
if __name__ == '__main__':
    # Plot_ROC()
    # Plot_Results()
    Confusion_matrix()
    # plot_results_conv()
