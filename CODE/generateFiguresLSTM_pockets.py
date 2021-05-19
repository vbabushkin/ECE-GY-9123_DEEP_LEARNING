# Generating the confusion matrices and barplots for report for LSTM Pockets detection
########################################################################################################################
from __future__ import print_function
import pickle
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pylab import plot, xlim, figure, ylim, legend, boxplot, setp, axes

########################################################################################################################
# AUXILIARY VARIABLES
########################################################################################################################
classNames = [ 'no pocket','pocket' ]
########################################################################################################################
# SET PARAMETERS
########################################################################################################################
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
n_classes = 2
n_folds = 12
numFeatures = 24
########################################################################################################################
# GET THE PERFORMANCE REPORTS BUCCAL
########################################################################################################################
reports_12_fn = ('RESULTS/reportsLSTM_Buccal_w_pockets.p')
with open(reports_12_fn, 'rb') as fp:
    reportsBuccal = pickle.load(fp)

classes = list(reportsBuccal[0].keys())[0:n_classes]
avgPrecisionBuccal = np.zeros(n_folds)
avgRecallBuccal = np.zeros(n_folds)
avgF1Buccal = np.zeros(n_folds)

for f in range(n_folds):
    tmpPrecisionBuccal = []
    tmpRecallBuccal = []
    tmpF1Buccal = []
    for clIdx in range(len(classes)):
        tmpPrecisionBuccal.append(reportsBuccal[f][classes[clIdx]]['precision'])
        tmpRecallBuccal.append(reportsBuccal[f][classes[clIdx]]['recall'])
        tmpF1Buccal.append(reportsBuccal[f][classes[clIdx]]['f1-score'])

    avgPrecisionBuccal[f] = np.mean(tmpPrecisionBuccal)
    avgRecallBuccal[f] = np.mean(tmpF1Buccal)
    avgF1Buccal[f] = np.mean(tmpF1Buccal)

accuracyBuccal = np.zeros(n_folds)

for fold in range(n_folds):
    accuracyBuccal[fold] = reportsBuccal[fold]['accuracy']
########################################################################################################################
# GET THE PERFORMANCE REPORTS LINGUAL
########################################################################################################################
reports_12_fn = ('RESULTS/reportsLSTM_Lingual_w_pockets.p')
with open(reports_12_fn, 'rb') as fp:
    reportsLingual = pickle.load(fp)

classes = list(reportsLingual[0].keys())[0:n_classes]
avgPrecisionLingual = np.zeros(n_folds)
avgRecallLingual = np.zeros(n_folds)
avgF1Lingual = np.zeros(n_folds)

for f in range(n_folds):
    tmpPrecisionLingual = []
    tmpRecallLingual = []
    tmpF1Lingual = []
    for clIdx in range(len(classes)):
        tmpPrecisionLingual.append(reportsLingual[f][classes[clIdx]]['precision'])
        tmpRecallLingual.append(reportsLingual[f][classes[clIdx]]['recall'])
        tmpF1Lingual.append(reportsLingual[f][classes[clIdx]]['f1-score'])

    avgPrecisionLingual[f] = np.mean(tmpPrecisionLingual)
    avgRecallLingual[f] = np.mean(tmpF1Lingual)
    avgF1Lingual[f] = np.mean(tmpF1Lingual)

accuracyLingual = np.zeros(n_folds)

for fold in range(n_folds):
    accuracyLingual[fold] = reportsLingual[fold]['accuracy']

########################################################################################################################
# PLOT AVERAGED PRECISION/RECALL/ACCURACY/F1 for Lingual/Buccal
#########################################################################################################################
plt.style.use('default')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# function for setting the colors of the box plots pairs
colorBucc = "#024178"  # 'darkblue'#"#023d70"
colorLing = "#3c94c3"  # 'cornflowerblue'#"#3c94c3"


def box_plot(data, edge_color, fill_color):
    bp = ax.boxplot(data, patch_artist=True)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)


def setBoxColors(bp):
    setp(bp['boxes'][0], color=colorBucc)
    setp(bp['boxes'][0], facecolor=colorBucc, alpha=0.6)
    setp(bp['caps'][0], color=colorBucc)
    setp(bp['caps'][1], color=colorBucc)
    setp(bp['whiskers'][0], color=colorBucc)
    setp(bp['whiskers'][1], color=colorBucc)
    setp(bp['fliers'][0], color=colorBucc)
    setp(bp['fliers'][1], color=colorBucc)
    setp(bp['medians'][0], color=colorBucc)

    setp(bp['boxes'][1], color='darkblue')
    setp(bp['boxes'][1], facecolor=colorLing, alpha=0.6)
    setp(bp['caps'][2], color='darkblue')
    setp(bp['caps'][3], color='darkblue')
    setp(bp['whiskers'][2], color='darkblue')
    setp(bp['whiskers'][3], color='darkblue')
    setp(bp['fliers'][0], color='darkblue')
    setp(bp['fliers'][1], color='darkblue')
    setp(bp['medians'][1], color='darkblue')


# Some fake data to plot
A = [accuracyBuccal, accuracyLingual]
B = [avgPrecisionBuccal, avgPrecisionLingual]
C = [avgRecallBuccal, avgRecallLingual]
D = [avgF1Buccal, avgF1Lingual]

fig = figure(figsize=(8, 4.9))
ax = axes()

# first boxplot pair
bp = boxplot(A, positions=[1.0, 2.0], widths=0.8, patch_artist=True, sym='+')
setBoxColors(bp)

# second boxplot pair
bp = boxplot(B, positions=[4, 5], widths=0.8, patch_artist=True, sym='+')
setBoxColors(bp)

# thrid boxplot pair
bp = boxplot(C, positions=[7, 8], widths=0.8, patch_artist=True, sym='+')
setBoxColors(bp)

# fourth boxplot pair
bp = boxplot(D, positions=[10, 11], widths=0.8, patch_artist=True, sym='+')
setBoxColors(bp)

# set axes limits and labels
xlim(0, 12)
ylim(0.2, 1)
plt.yticks(fontsize=16)
ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-score'], fontsize="16")
ax.set_xticks([1.5, 4.5, 7.5, 10.5])

# draw temporary red and blue lines and use them to create a legend
bucc, = plot([1, 1], colorBucc, linewidth=3)
ling, = plot([1, 1], color=colorLing, linewidth=3)

legend((bucc, ling), ('LR Buccal', 'LL Lingual'), fontsize="16")
bucc.set_visible(False)
ling.set_visible(False)
fig.tight_layout()
plt.savefig('FIGURES/LSTM_metrics_12fold_barplot_paper_'+str(numFeatures)+'_features_w_pockets.pdf')

########################################################################################################################
# PLOT THE NORMALIZED CONFUSION MATRIX BUCCAL
########################################################################################################################
import seaborn.apionly as sns

confusionMatrices_12_fn = ('RESULTS/confusionMatricesLSTM_Buccal_w_pockets.p')
with open(confusionMatrices_12_fn, 'rb') as fp:
    confusionMatricesBuccal = pickle.load(fp)

classNames = [ 'no pocket','pocket' ]
n_classes = 2
n_folds = 12
# normalize confusion matrices
normalizedAvgCM = np.zeros((n_classes, n_classes))
for i in range(len(confusionMatricesBuccal)):
    cm = confusionMatricesBuccal[i]
    normalizedAvgCM += cm / cm.astype(np.float).sum(axis=1)

normalizedAvgCM = normalizedAvgCM / n_folds

# plot one time prediction confusion matrix
df_cm = pd.DataFrame(normalizedAvgCM, index=classNames, columns=classNames)
plt.figure(figsize=(5.6, 3.7))
sns.set(font_scale=1.2)  # for label size
ax = sns.heatmap(df_cm, cbar_kws={'ticks': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}, vmin=0, vmax=1.0,
                 annot=True, annot_kws={"size": 20}, fmt='2.2f', cmap="Blues")  # font size
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_yticklabels(classNames, rotation=90, fontsize="16", va="center")
ax.tick_params(axis='x', which='major', pad=10)
ax.set_xticklabels(classNames, rotation=0, fontsize="16", va="center")
ax.set_ylabel('True Label', fontsize="20")
ax.set_xlabel('Predicted Label', fontsize="20")
plt.tight_layout()
plt.savefig('FIGURES/normCM_LSTM_BUCCAL_six_regions_manual_'+str(numFeatures)+'_features_w_pockets.pdf')

########################################################################################################################
# PLOT THE NORMALIZED CONFUSION MATRIX LINGUAL
########################################################################################################################
confusionMatrices_12_fn = ('RESULTS/confusionMatricesLSTM_Lingual_w_pockets.p')
with open(confusionMatrices_12_fn, 'rb') as fp:
    confusionMatricesLingual = pickle.load(fp)

# normalize confusion matrices
normalizedAvgCM = np.zeros((n_classes, n_classes))
for i in range(len(confusionMatricesBuccal)):
    cm = confusionMatricesLingual[i]
    normalizedAvgCM += cm / cm.astype(np.float).sum(axis=1)

normalizedAvgCM = normalizedAvgCM / n_folds

# plot one time prediction confusion matrix
df_cm = pd.DataFrame(normalizedAvgCM, index=classNames, columns=classNames)
plt.figure(figsize=(5.6, 3.7))
sns.set(font_scale=1.2)  # for label size
ax = sns.heatmap(df_cm, cbar_kws={'ticks': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}, vmin=0, vmax=1.0,
                 annot=True, annot_kws={"size": 20}, fmt='2.2f', cmap="Blues")  # font size
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_yticklabels(classNames, rotation=90, fontsize="16", va="center")
ax.tick_params(axis='x', which='major', pad=10)
ax.set_xticklabels(classNames, rotation=0, fontsize="16", va="center")
ax.set_ylabel('True Label', fontsize="20")
ax.set_xlabel('Predicted Label', fontsize="20")
plt.tight_layout()
plt.savefig('FIGURES/normCM_LSTM_LINGUAL_six_regions_manual_'+str(numFeatures)+'_features_w_pockets.pdf')

########################################################################################################################
# OUTPUT THE REPORTS
########################################################################################################################
acc_mean = []
# Compute mean accuracy
for i in range(n_folds):
    acc_mean.append(reportsBuccal[i]['accuracy'])

# print performance metrics after 10 fold CV
avgPrecision = np.zeros(n_classes)
avgRecall = np.zeros(n_classes)
avgF1 = np.zeros(n_classes)
classes = list(reportsBuccal[0].keys())[0:n_classes]
for clIdx in range(len(classes)):
    tmpPrecision = []
    tmpRecall = []
    tmpF1 = []

    for f in range(n_folds):
        tmpPrecision.append(reportsBuccal[f][classes[clIdx]]['precision'])
        tmpRecall.append(reportsBuccal[f][classes[clIdx]]['recall'])
        tmpF1.append(reportsBuccal[f][classes[clIdx]]['f1-score'])

    avgPrecision[clIdx] = np.mean(tmpPrecision)
    avgRecall[clIdx] = np.mean(tmpRecall)
    avgF1[clIdx] = np.mean(tmpF1)
print("Accuracy Buccal :" + str(np.mean(acc_mean)))
print("Precision Buccal :" + str(np.mean(avgPrecision)))
print("Recall Buccal :" + str(np.mean(avgRecall)))
print("F1 Buccal :" + str(np.mean(avgF1)))

acc_mean = []
# Compute mean accuracy
for i in range(n_folds):
    acc_mean.append(reportsLingual[i]['accuracy'])

# print performance metrics after 12 fold CV
avgPrecision = np.zeros(n_classes)
avgRecall = np.zeros(n_classes)
avgF1 = np.zeros(n_classes)
classes = list(reportsLingual[0].keys())[0:n_classes]
for clIdx in range(len(classes)):
    tmpPrecision = []
    tmpRecall = []
    tmpF1 = []

    for f in range(n_folds):
        tmpPrecision.append(reportsLingual[f][classes[clIdx]]['precision'])
        tmpRecall.append(reportsLingual[f][classes[clIdx]]['recall'])
        tmpF1.append(reportsLingual[f][classes[clIdx]]['f1-score'])

    avgPrecision[clIdx] = np.mean(tmpPrecision)
    avgRecall[clIdx] = np.mean(tmpRecall)
    avgF1[clIdx] = np.mean(tmpF1)
print("Accuracy Lingual :" + str(np.mean(acc_mean)))
print("Precision Lingual :" + str(np.mean(avgPrecision)))
print("Recall Lingual :" + str(np.mean(avgRecall)))
print("F1 Lingual :" + str(np.mean(avgF1)))