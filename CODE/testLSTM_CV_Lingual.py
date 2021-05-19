# Probed region detection for Lingual side with LSTN
# https://www.kaggle.com/purplejester/a-simple-lstm-based-time-series-classifier
########################################################################################################################
# importing the libraries
########################################################################################################################

import pickle

import tensorflow.compat.v1 as tf
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from torch.nn import functional as F
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import TensorDataset, DataLoader

tf.disable_v2_behavior()
import torch.nn as nn
from loadData import loadData

import pandas as pd
import numpy as np

# for reading and displaying images
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split

# PyTorch libraries and modules
import torch
import time

########################################################################################################################
# AUXILIARY VARIABLES
########################################################################################################################

mainFolder = '/Users/vahan/Desktop/SENSORIMOTOR PROJECT/sensorimotorProject/'

featuresNames = ["deviceLocalPos_x", "deviceLocalPos_y", "deviceLocalPos_z",
                 "gripperAngleDeg", "gripperAngVel",
                 "deviceLocalLinVel_x", "deviceLocalLinVel_y", "deviceLocalLinVel_z",
                 "deviceLocalAngVel_x", "deviceLocalAngVel_y", "deviceLocalAngVel_z",
                 "gripperForce",
                 "deviceLocalForce_x", "deviceLocalForce_y", "deviceLocalForce_z",
                 "R00_loc", "R10_loc", "R20_loc",
                 "R01_loc", "R11_loc", "R21_loc",
                 "R02_loc", "R12_loc", "R22_loc"]

classNamesLingual = ['17_DL', '17_L', '17_ML',
                     '18_DL', '18_L', '18_ML',
                     '19_DL', '19_L', '19_ML',
                     '20_DL', '20_L', '20_ML',
                     '21_DL', '21_L', '21_ML',
                     ]


########################################################################################################################
# AUXILIARY FUNCTIONS
########################################################################################################################
def create_grouped_array(data, group_col='series_id', drop_cols=['series_id']):
    X_grouped = np.row_stack([
        group.drop(columns=drop_cols).values[None]
        for _, group in data.groupby(group_col)])
    return X_grouped


def create_loaders(train_ds, valid_ds, bs=512, jobs=0):
    train_dl = DataLoader(train_ds, bs, shuffle=True, num_workers=jobs)
    valid_dl = DataLoader(valid_ds, bs, shuffle=False, num_workers=jobs)
    return train_dl, valid_dl


def create_datasets(X, y, test_size=0.2, dropcols=['series_id']):
    enc = LabelEncoder()
    y_enc = enc.fit_transform(y['label'])
    X_grouped = create_grouped_array(X, drop_cols=['series_id'])
    X_train, X_valid, y_train, y_valid = train_test_split(X_grouped, y_enc, test_size=0.1)
    X_train, X_valid = [torch.tensor(arr, dtype=torch.float32) for arr in (X_train, X_valid)]
    y_train, y_valid = [torch.tensor(arr, dtype=torch.long) for arr in (y_train, y_valid)]
    train_ds = TensorDataset(X_train, y_train)
    valid_ds = TensorDataset(X_valid, y_valid)
    return train_ds, valid_ds, enc


def accuracy(output, target):
    return (output.argmax(dim=1) == target).float().mean().item()


class CyclicLR(_LRScheduler):
    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]


def cosine(t_max, eta_min=0):
    def scheduler(epoch, base_lr):
        t = epoch % t_max
        return eta_min + (base_lr - eta_min) * (1 + np.cos(np.pi * t / t_max)) / 2

    return scheduler


########################################################################################################################
# import the data:
########################################################################################################################
(xbuccal, xlingual, ybuccal, ylingual) = loadData(mainFolder)

(xLingual1, xLingual2, xLingual3, xLingual4, xLingual5, xLingual6, xLingual7, xLingual8, xLingual9, xLingual10,
 xLingual11, xLingual12) = xlingual

(yLingual1, yLingual2, yLingual3, yLingual4, yLingual5, yLingual6, yLingual7, yLingual8, yLingual9, yLingual10,
 yLingual11, yLingual12) = ylingual

########################################################################################################################
# specify important features
########################################################################################################################
selectedFeatures = list(range(24))
########################################################################################################################
# SET PARAMETERS
########################################################################################################################
bs = 128  # batch size
n_epochs = 20  # defining the number of epochs
n_folds = 12  # number of CV folds
input_dim = 24
hidden_dim = 256
layer_dim = 3
output_dim = 15  # number of classes
n_classes = output_dim
lr = 0.001  # learning rate of ADAM optimizer
ID_COLS = ['series_id']
n = 100  # for scheduler
cols = [str(i) for i in range(24)]  # columns for dataframes
numFeatures = len(selectedFeatures)
########################################################################################################################
# create folds for CV
########################################################################################################################
foldsLingual = [xLingual1, xLingual2, xLingual3, xLingual4, xLingual5, xLingual6, xLingual7, xLingual8, xLingual9,
                xLingual10, xLingual11, xLingual12]
labelsLingual = [yLingual1.astype(int) - 16, yLingual2.astype(int) - 16, yLingual3.astype(int) - 16,
                 yLingual4.astype(int) - 16,
                 yLingual5.astype(int) - 16, yLingual6.astype(int) - 16, yLingual7.astype(int) - 16,
                 yLingual8.astype(int) - 16,
                 yLingual9.astype(int) - 16, yLingual10.astype(int) - 16, yLingual11.astype(int) - 16,
                 yLingual12.astype(int) - 16]


########################################################################################################################
# LSTM CLASS
########################################################################################################################
class LSTMClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        return [t for t in (h0, c0)]


########################################################################################################################
# prepare datasets
########################################################################################################################
yhatPerFold = []
ytsPerFold = []
accuracyTrain = []
accuracyVal = []
confusionMatricesLingual = []
normConfusionMatrices = []
reportsLingual = []
rsq = []
acc = []

testAccuracyPerFold = []
# empty list to store training losses
trainLossPerFold = []
# empty list to store validation losses
valLossPerFold = []
# empty list to store training accuracies
trainAccuracyPerFold = []
# empty list to store validation accuracies
valAccuracyPerFold = []

for fold in range(n_folds):
    t0 = time.time()
    print("runnig fold " + str(fold) + " Lingual...")
    tmpFoldsLingual = foldsLingual.copy()
    tmpLabelsLingual = labelsLingual.copy()
    Xts = tmpFoldsLingual.pop(fold)[:, selectedFeatures]
    yts = tmpLabelsLingual.pop(fold)

    Xtr = np.vstack(tmpFoldsLingual)[:, selectedFeatures]
    ytr = np.hstack(tmpLabelsLingual)

    scaler = preprocessing.StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xts = scaler.transform(Xts)

    Xtr_df = pd.DataFrame(Xtr, columns=cols)
    Xtr_df.insert(0, "series_id", Xtr_df.index, True)

    ytr_df = pd.DataFrame(ytr, columns=['label'])
    ytr_df.insert(0, "series_id", ytr_df.index, True)

    Xts_df = pd.DataFrame(Xts, columns=cols)
    Xts_df.insert(0, "series_id", Xts_df.index, True)

    yts_df = pd.DataFrame(yts, columns=['label'])
    yts_df.insert(0, "series_id", yts_df.index, True)

    train_ds, valid_ds, enc = create_datasets(Xtr_df, ytr_df)

    print(f'Creating data loaders with batch size: {bs} fold: {fold}')
    trn_dl, val_dl = create_loaders(train_ds, valid_ds, bs)

    iterations_per_epoch = len(trn_dl)
    ########################################################################################################################
    # LSTM
    ########################################################################################################################
    model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.RMSprop(model.parameters(), lr=lr)
    # we use the cos scheduler to alter the learning rate cyclically
    sched = CyclicLR(opt, cosine(t_max=iterations_per_epoch * 2, eta_min=lr / 100))
    ########################################################################################################################
    # LSTM training
    ########################################################################################################################
    print('Start model training')
    tmpTrainAccuracyPerFold = []
    tmpValAccuracyPerFold = []
    tmpTrainLossPerFold = []
    tmpValLossPerFold = []
    for epoch in range(1, n_epochs + 1):
        t0_epoch = time.time()
        correctTr, totalTr = 0, 0
        for i, (x_batch, y_batch) in enumerate(trn_dl):
            model.train()
            sched.step()
            opt.zero_grad()
            out = model(x_batch)
            predsTr = F.log_softmax(out, dim=1).argmax(dim=1)
            loss = criterion(out, y_batch)
            loss.backward()
            opt.step()
            totalTr += y_batch.size(0)
            correctTr += (predsTr == y_batch).sum().item()
        # calculate training accuracy
        trAcc = correctTr / totalTr

        correct, total = 0, 0
        for x_val, y_val in val_dl:
            model.eval()
            out = model(x_val)
            preds = F.log_softmax(out, dim=1).argmax(dim=1)
            total += y_val.size(0)
            valLoss = criterion(out, y_val)
            correct += (preds == y_val).sum().item()
        valAcc = correct / total
        tmpTrainAccuracyPerFold.append(trAcc)
        tmpValAccuracyPerFold.append(valAcc)
        tmpTrainLossPerFold.append(loss.item())
        tmpValLossPerFold.append(valLoss.item())

        t_epoch = time.time()
        print(
            f'Epoch: {epoch:3d}. Train loss: {loss.item():.4f}. Train accuracy: {trAcc:2.2%}.  Validation accuracy: {valAcc:2.2%}. Elapsed time: {t_epoch - t0_epoch:2}')

    # create test dataset
    X_grouped = np.row_stack([
        group.drop(columns='series_id').values[None]
        for _, group in Xts_df.groupby('series_id')])
    X_grouped = torch.tensor(X_grouped.transpose(0, 2, 1)).float()
    y_true = torch.tensor(yts).long()

    test_dl = DataLoader(TensorDataset(X_grouped, y_true), batch_size=64, shuffle=False)
    yhat = []
    print('Predicting on test dataset')

    for batch, y_val in test_dl:
        model.eval()
        batch = batch.permute(0, 2, 1)
        out = model(batch)
        y_hat = F.log_softmax(out, dim=1).argmax(dim=1)
        yhat += y_hat.tolist()
        testLoss = criterion(out, y_hat)
        correct += (y_hat == y_val).sum().item()
        total += y_val.size(0)

    testAccuracy = correct / total
    print(f' test accuracy.: {testAccuracy:2.2%}')
    print("Confusion matrix on the test data")
    cm = confusion_matrix(yts, yhat, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    print(cm)
    normCm = cm / cm.astype(np.float).sum(axis=1)
    normalizedAvgCM = normCm
    t1 = time.time()
    yhatPerFold.append(yhat)
    ytsPerFold.append(yts)
    a = classification_report(yts, yhat, target_names=classNamesLingual, output_dict=True)
    reportsLingual.append(a)
    confusionMatricesLingual.append(cm)
    normConfusionMatrices.append(normCm)
    testAccuracyPerFold.append(testAccuracy)
    trainLossPerFold.append(np.mean(tmpTrainLossPerFold))
    valLossPerFold.append(np.mean(tmpValLossPerFold))
    valAccuracyPerFold.append(np.mean(tmpValAccuracyPerFold))
    trainAccuracyPerFold.append(np.mean(tmpTrainAccuracyPerFold))
    print('Test loss :', testLoss, '\ttest accuracy  :', str(testAccuracy), '\t time elapsed per fold: ', str(t1 - t0))

########################################################################################################################
# SAVE
########################################################################################################################
# to avoid losing data due to  libc++abi.dylib: terminating with uncaught exception of type std::__1::system_error: condition_variable wait failed: Invalid argument
reports_12_fn = ('RESULTS/reportsLSTM_Lingual_w.p')
with open(reports_12_fn, 'wb') as fp:
    pickle.dump(reportsLingual, fp)

confusionMatrices_12_fn = ('RESULTS/confusionMatricesLSTM_Lingual_w.p')
with open(confusionMatrices_12_fn, 'wb') as fp:
    pickle.dump(confusionMatricesLingual, fp)

normConfusionMatrices_12_fn = ('RESULTS/normConfusionMatricesLSTM_Lingual_w.p')
with open(normConfusionMatrices_12_fn, 'wb') as fp:
    pickle.dump(normConfusionMatrices, fp)

predictedLabels_12_fn = ('RESULTS/predLabelsLSTM_Lingual_w.p')
with open(predictedLabels_12_fn, 'wb') as fp:
    pickle.dump(yhatPerFold, fp)

actualLabels_12_fn = ('RESULTS/actualLabelsLSTM_Lingual_w.p')
with open(actualLabels_12_fn, 'wb') as fp:
    pickle.dump(ytsPerFold, fp)

valLoss_12_fn = ('RESULTS/valLossLSTM_Lingual_w.p')
with open(valLoss_12_fn, 'wb') as fp:
    pickle.dump(valLossPerFold, fp)

valAccuracy_12_fn = ('RESULTS/valAccuracyLSTM_Lingual_w.p')
with open(valAccuracy_12_fn, 'wb') as fp:
    pickle.dump(valAccuracyPerFold, fp)

trainLoss_12_fn = ('RESULTS/trainLossLSTM_Lingual_w.p')
with open(trainLoss_12_fn, 'wb') as fp:
    pickle.dump(trainLossPerFold, fp)

valAccuracy_12_fn = ('RESULTS/valAccuracyLSTM_Lingual_w.p')
with open(valAccuracy_12_fn, 'wb') as fp:
    pickle.dump(valAccuracyPerFold, fp)

trainAccuracy_12_fn = ('RESULTS/trainAccuracyLSTM_Lingual_w.p')
with open(trainAccuracy_12_fn, 'wb') as fp:
    pickle.dump(trainAccuracyPerFold, fp)

testAccuracy_12_fn = ('RESULTS/testAccuracyLSTM_Lingual_w.p')
with open(testAccuracy_12_fn, 'wb') as fp:
    pickle.dump(testAccuracyPerFold, fp)
########################################################################################################################
########################################################################################################################
# GET THE PERFORMANCE REPORTS
########################################################################################################################
classes = list(reportsLingual[0].keys())[0:output_dim]
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
# OUTPUT THE REPORTS
########################################################################################################################
# print performance metrics after 12 fold CV
acc_mean = []
# Compute mean accuracy
for i in range(n_folds):
    acc_mean.append(reportsLingual[i]['accuracy'])

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
########################################################################################################################
# PLOT THE NORMALIZED CONFUSION MATRIX LINGUAL
########################################################################################################################
import seaborn.apionly as sns

# normalize confusion matrices
normalizedAvgCM = np.zeros((output_dim, output_dim))
for i in range(len(confusionMatricesLingual)):
    cm = confusionMatricesLingual[i]
    normalizedAvgCM += cm / cm.astype(np.float).sum(axis=1)

normalizedAvgCM = normalizedAvgCM / n_folds

# plot one time prediction confusion matrix
df_cm = pd.DataFrame(normalizedAvgCM, index=classNamesLingual, columns=classNamesLingual)
plt.figure(figsize=(15.6, 8.0))
sns.set(font_scale=1.5)  # for label size
ax = sns.heatmap(df_cm, cbar_kws={'ticks': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}, vmin=0, vmax=1.0,
                 annot=True, annot_kws={"size": 18}, fmt='2.2f', cmap="Blues")  # font size
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_yticklabels(classNamesLingual, rotation=0, fontsize="18", va="center")
ax.tick_params(axis='x', which='major', pad=20)
ax.set_xticklabels(classNamesLingual, rotation=45, fontsize="18", va="center")
ax.set_ylabel('True Label', fontsize="22")
ax.set_xlabel('Predicted Label', fontsize="22")
plt.tight_layout()
plt.savefig('FIGURES/normCM_LSTM_LINGUAL_six_regions_manual_' + str(numFeatures) + '_features_w.pdf')
