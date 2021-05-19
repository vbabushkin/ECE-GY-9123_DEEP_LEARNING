# Pocket detection for Buccal side with CNN using a sliding window of size w
#
########################################################################################################################
# importing the libraries
########################################################################################################################
import pickle

import tensorflow.compat.v1 as tf
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn.apionly as sns
tf.disable_v2_behavior()
import torch.nn as nn
from loadPocketData import loadPocketData
import pandas as pd
import numpy as np
# for reading and displaying images
import matplotlib.pyplot as plt
# for creating validation set
from sklearn.model_selection import train_test_split
# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, BCELoss, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d
from torch.optim import Adam

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

classNamesBuccal = ['no pocket', 'pocket']
########################################################################################################################
# import the data:
########################################################################################################################
(xbuccal, xlingual, ybuccal, ylingual) = loadPocketData(mainFolder)

(xBuccal1, xBuccal2, xBuccal3, xBuccal4, xBuccal5, xBuccal6, xBuccal7, xBuccal8, xBuccal9, xBuccal10, xBuccal11,
 xBuccal12) = xbuccal

(yBuccal1, yBuccal2, yBuccal3, yBuccal4, yBuccal5, yBuccal6, yBuccal7, yBuccal8, yBuccal9, yBuccal10, yBuccal11,
 yBuccal12) = ybuccal

########################################################################################################################
# specify important features
########################################################################################################################
# selectedFeatures = list(range(24))#[0, 1, 19, 20, 21]#[0, 1, 2, 15, 16, 17, 18, 19, 20, 21, 22, 23] list(range(24))
selectedFeatures = [0, 1, 2, 17, 19, 23]
########################################################################################################################
# SET PARAMETERS
########################################################################################################################
w = 12  # sliding window
numClasses = 2  # number of classes
learning_rate = 0.07  # learning rate of ADAM optimizer
n_epochs = 100  # defining the number of epochs
n_folds = 12
# kernel dimensions
k1 = 2
k2 = 3
########################################################################################################################
# create folds for CV
########################################################################################################################
foldsBuccal = [xBuccal1, xBuccal2, xBuccal3, xBuccal4, xBuccal5, xBuccal6, xBuccal7, xBuccal8, xBuccal9, xBuccal10,
               xBuccal11, xBuccal12]
labelsBuccal = [yBuccal1, yBuccal2, yBuccal3, yBuccal4,
                yBuccal5, yBuccal6, yBuccal7, yBuccal8,
                yBuccal9, yBuccal10, yBuccal11, yBuccal12]

yhatPerFold = []
ytsPerFold = []
accuracyTrain = []
accuracyVal = []
confusionMatricesBuccal = []
normConfusionMatrices = []
reportsBuccal = []
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
    print("runnig fold " + str(fold) + " buccal...")
    tmpFoldsBuccal = foldsBuccal.copy()
    tmpLabelsBuccal = labelsBuccal.copy()
    Xts = tmpFoldsBuccal.pop(fold)[:, selectedFeatures]
    yts = tmpLabelsBuccal.pop(fold)

    Xtr = np.vstack(tmpFoldsBuccal)[:, selectedFeatures]
    ytr = np.hstack(tmpLabelsBuccal)

    scaler = preprocessing.StandardScaler().fit(Xtr)
    Xtr = scaler.transform(Xtr)
    Xts = scaler.transform(Xts)

    tmpData = Xtr[0, :]
    newXtr = tmpData
    newYtr = ytr[0]
    train_img = []
    for i in range(1, int(w * np.floor(Xtr.shape[0] / w))):
        if i % w != 0:
            tmpData = np.vstack((tmpData, Xtr[i, :]))
        else:
            train_img.append(tmpData)
            newYtr = np.hstack((newYtr, ytr[i]))
            tmpData = Xtr[i, :]
    train_img = np.array(train_img)

    newYtr = newYtr[0:newYtr.shape[0] - 1]
    train_img.shape

    tmpData = Xts[0, :]
    newYts = yts[0]
    test_img = []
    for i in range(1, int(w * np.floor(Xts.shape[0] / w))):
        if i % w != 0:
            tmpData = np.vstack((tmpData, Xts[i, :]))
        else:
            test_img.append(tmpData)
            newYts = np.hstack((newYts, yts[i]))
            tmpData = Xts[i, :]
    test_img = np.array(test_img)

    newYts = newYts[0:newYts.shape[0] - 1]
    test_img.shape

    train_x = train_img
    train_y = newYtr

    test_x = test_img
    test_y = newYts

    ########################################################################################################################
    # CNN
    ########################################################################################################################
    # create validation set
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.1)
    (train_x.shape, train_y.shape), (val_x.shape, val_y.shape)

    # converting training images into torch format
    train_x = train_x.reshape(train_x.shape[0], 1, train_x.shape[1], train_x.shape[2])
    train_x = torch.from_numpy(train_x).float()

    # converting the target into torch format
    train_y = train_y.astype(int);
    train_y = torch.from_numpy(train_y).float()

    # shape of training data
    train_x.shape, train_y.shape  # [83569, 1, 8, 4] 83569 timepoints, 1 dummy dimension(channels), 8 -- window size, 4 -- number of features

    # converting validation images into torch format
    val_x = val_x.reshape(val_x.shape[0], 1, val_x.shape[1], val_x.shape[2])
    val_x = torch.from_numpy(val_x).float()

    # converting the target into torch format
    val_y = val_y.astype(int);
    val_y = torch.from_numpy(val_y).float()

    # shape of validation data
    val_x.shape, val_y.shape


    ########################################################################################################################
    # Create network architecture
    ########################################################################################################################
    # PrintLayer just prints out the sizes of input tensors to each layer
    class PrintLayer(nn.Module):
        def __init__(self):
            super(PrintLayer, self).__init__()

        def forward(self, x):
            # Do your print / debug stuff here
            print(x.size())
            return x


    class Net(Module):
        def __init__(self):
            super(Net, self).__init__()

            self.cnn_layers = Sequential(
                # Defining a 2D convolution layer
                # PrintLayer(),#[33172, 1, 12, 6]
                Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
                # PrintLayer(),#[33172, 4, 12, 6]
                BatchNorm2d(4),
                # PrintLayer(),#[33172, 4, 12, 6]
                ReLU(inplace=True),
                MaxPool2d(kernel_size=(k1, k2)),
                # PrintLayer(),#[33172, 4, 6, 3]
                # Defining another 2D convolution layer
                Conv2d(4, 1, kernel_size=3, stride=1, padding=1),
                # PrintLayer(),#[33172, 1, 6, 3]
                BatchNorm2d(1),
                ReLU(inplace=True),
                # PrintLayer()#[33172, 1, 6, 3]
            )

            self.linear_layers = Sequential(
                # PrintLayer(),  # Add Print layer for debug #[33172, 18]
                Linear(w, 1),  # 4*7*7 #4 *number = windowSize#4*1*6 for w=24, 4 * 1 * 12 for w=48
                # PrintLayer()  # Add Print layer for debug
            )

        # Defining the forward pass
        def forward(self, x):
            x = self.cnn_layers(x)
            x = x.view(x.size(0), -1)
            x = self.linear_layers(x)
            return torch.sigmoid(x)


    ########################################################################################################################
    # CNN training
    ########################################################################################################################
    # defining the model
    model = Net()
    # defining the optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # defining the loss function
    criterion = BCELoss()
    # checking if GPU is available
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()

    print(model)

    # training the model
    def train(epoch):
        model.train()
        tr_loss = 0
        # getting the training set
        x_train, y_train = Variable(train_x), Variable(train_y)
        # getting the validation set
        x_val, y_val = Variable(val_x), Variable(val_y)
        # converting the data into GPU format
        if torch.cuda.is_available():
            x_train = x_train.cuda()
            y_train = y_train.cuda()
            x_val = x_val.cuda()
            y_val = y_val.cuda()

        # clearing the Gradients of the model parameters
        optimizer.zero_grad()

        # prediction for training and validation set
        output_train = model(x_train)
        output_val = model(x_val)

        # computing the training and validation loss
        loss_train = criterion(output_train.view(-1), y_train)
        loss_val = criterion(output_val.view(-1), y_val)

        # computing the updated weights of all the model parameters
        loss_train.backward()
        optimizer.step()
        tr_loss = loss_train.item()

        valAccuracy = torch.sum((torch.argmax(output_val, dim=1).type(torch.FloatTensor) == y_val).detach()).float() / \
                      y_val.shape[0]
        trainAccuracy = torch.sum(
            (torch.argmax(output_train, dim=1).type(torch.FloatTensor) == y_train).detach()).float() / y_train.shape[0]
        if epoch % 2 == 0:  # printing the validation loss and train/val accuracies
            print('Epoch : ', epoch + 1, '\t', 'loss :', loss_val, '\ttrain accuracy  :', str(trainAccuracy),
                  '\tvalidation accuracy  :', str(valAccuracy))
        return (loss_train, loss_val, trainAccuracy, valAccuracy)


    # empty list to store training losses
    train_losses = []
    # empty list to store validation losses
    val_losses = []
    # empty list to store training accuracies
    train_accuracies = []
    # empty list to store validation accuracies
    val_accuracies = []
    # training the model
    for epoch in range(n_epochs):
        (loss_train, loss_val, trainAccuracy, valAccuracy) = train(epoch)
        train_losses.append(loss_train)
        val_losses.append(val_losses)
        train_accuracies.append(trainAccuracy)
        val_accuracies.append(valAccuracy)

    ########################################################################################################################
    # CNN testing
    ########################################################################################################################
    # set model to evaluation mode to avoid gradients training suring validation
    model.eval()
    print("================================")
    print("Evaluating...")

    loss = 0

    # converting training images into torch format
    test_x = test_img.reshape(test_img.shape[0], 1, test_img.shape[1], test_img.shape[2])
    test_x = torch.from_numpy(test_x).float()

    # converting the target into torch format
    test_y = newYts.astype(int);
    test_y = torch.from_numpy(test_y).float()

    confusion = np.zeros((numClasses, numClasses))
    x_test, y_test = Variable(test_x), Variable(test_y)
    output = model(x_test)

    rows = y_test.cpu().numpy()
    cols = output.max(1)[1].cpu().numpy()

    tmpConfusion = np.zeros((numClasses, numClasses))

    loss_test = criterion(output.view(-1), y_test)

    yhat = torch.argmax(output, dim=1).type(torch.FloatTensor)

    confusion = confusion_matrix(y_test.detach().numpy(), yhat.detach().numpy(), labels=range(numClasses))
    acc_test = np.trace(confusion) / np.sum(confusion)

    testAccuracy = torch.sum((torch.argmax(output, dim=1).type(torch.FloatTensor) == y_test).detach()).float() / \
                   y_test.shape[0]
    print('Test loss :', loss_test, '\ttest accuracy  :', str(testAccuracy))

    yhatPerFold.append(yhat)
    ytsPerFold.append(yts)
    a = classification_report(y_test, yhat, target_names=classNamesBuccal, output_dict=True)
    reportsBuccal.append(a)
    print("Confusion matrix on the test data")
    cm = confusion_matrix(y_test, yhat, labels=[0, 1])
    print(cm)
    confusionMatricesBuccal.append(cm)
    normCm = cm / cm.astype(np.float).sum(axis=1)
    normConfusionMatrices.append(normCm)
    testAccuracyPerFold.append(testAccuracy)
    trainLossPerFold.append(train_losses[-1])
    valLossPerFold.append(val_losses[-1])
    trainAccuracyPerFold.append(train_accuracies[-1])
    valAccuracyPerFold.append(val_accuracies[-1])

########################################################################################################################
# SAVE
########################################################################################################################
reports_12_fn = ('RESULTS/reportsCNN_Buccal_w_' + str(w) + '_pockets.p')
with open(reports_12_fn, 'wb') as fp:
    pickle.dump(reportsBuccal, fp)

confusionMatrices_12_fn = ('RESULTS/confusionMatricesCNN_Buccal_w_' + str(w) + '_pockets.p')
with open(confusionMatrices_12_fn, 'wb') as fp:
    pickle.dump(confusionMatricesBuccal, fp)

normConfusionMatrices_12_fn = ('RESULTS/normConfusionMatricesCNN_Buccal_w_' + str(w) + '_pockets.p')
with open(normConfusionMatrices_12_fn, 'wb') as fp:
    pickle.dump(normConfusionMatrices, fp)

predictedLabels_12_fn = ('RESULTS/predLabelsCNN_Buccal_w_' + str(w) + '_pockets.p')
with open(predictedLabels_12_fn, 'wb') as fp:
    pickle.dump(yhatPerFold, fp)

actualLabels_12_fn = ('RESULTS/actualLabelsCNN_Buccal_w_' + str(w) + '_pockets.p')
with open(actualLabels_12_fn, 'wb') as fp:
    pickle.dump(ytsPerFold, fp)

valLoss_12_fn = ('RESULTS/valLossCNN_Buccal_w_' + str(w) + '_pockets.p')
with open(valLoss_12_fn, 'wb') as fp:
    pickle.dump(valLossPerFold, fp)

valAccuracy_12_fn = ('RESULTS/valAccuracyCNN_Buccal_w_' + str(w) + '_pockets.p')
with open(valAccuracy_12_fn, 'wb') as fp:
    pickle.dump(valAccuracyPerFold, fp)

trainLoss_12_fn = ('RESULTS/trainLossCNN_Buccal_w_' + str(w) + '_pockets.p')
with open(trainLoss_12_fn, 'wb') as fp:
    pickle.dump(trainLossPerFold, fp)

trainAccuracy_12_fn = ('RESULTS/trainAccuracyCNN_Buccal_w_' + str(w) + '_pockets.p')
with open(trainAccuracy_12_fn, 'wb') as fp:
    pickle.dump(trainAccuracyPerFold, fp)

testAccuracy_12_fn = ('RESULTS/testAccuracyCNN_Buccal_w_' + str(w) + '_pockets.p')
with open(testAccuracy_12_fn, 'wb') as fp:
    pickle.dump(testAccuracyPerFold, fp)
########################################################################################################################
# PLOT THE NORMALIZED CONFUSION MATRIX BUCCAL
########################################################################################################################
numFeatures = len(selectedFeatures)

# normalize confusion matrices
normalizedAvgCM = np.zeros((numClasses, numClasses))
for i in range(len(confusionMatricesBuccal)):
    cm = confusionMatricesBuccal[i]
    normalizedAvgCM += cm / cm.astype(np.float).sum(axis=1)
normalizedAvgCM = normalizedAvgCM / n_folds
# plot one time prediction confusion matrix
df_cm = pd.DataFrame(normalizedAvgCM, index=classNamesBuccal, columns=classNamesBuccal)
plt.figure(figsize=(15.6, 8.0))
sns.set(font_scale=1.5)  # for label size
ax = sns.heatmap(df_cm, cbar_kws={'ticks': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}, vmin=0, vmax=1.0,
                 annot=True, annot_kws={"size": 18}, fmt='2.2f', cmap="Blues")  # font size
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
# ax.set_ylim(sorted(ax.get_xlim(), reverse=True))
ax.set_yticklabels(classNamesBuccal, rotation=0, fontsize="18", va="center")
ax.tick_params(axis='x', which='major', pad=20)
ax.set_xticklabels(classNamesBuccal, rotation=45, fontsize="18", va="center")
ax.set_ylabel('True Label', fontsize="22")
ax.set_xlabel('Predicted Label', fontsize="22")
plt.tight_layout()
plt.savefig(
    'FIGURES/normCM_CNN_BUCCAL_six_regions_manual_' + str(numFeatures) + '_features_w_' + str(w) + '_pockets.pdf')

########################################################################################################################
# GET THE PERFORMANCE REPORTS
########################################################################################################################
acc_mean = []
# Compute mean accuracy
for i in range(n_folds):
    acc_mean.append(reportsBuccal[i]['accuracy'])
# print performance metrics after 10 fold CV
avgPrecision = np.zeros(numClasses)
avgRecall = np.zeros(numClasses)
avgF1 = np.zeros(numClasses)
classes = list(reportsBuccal[0].keys())[0:numClasses]
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
