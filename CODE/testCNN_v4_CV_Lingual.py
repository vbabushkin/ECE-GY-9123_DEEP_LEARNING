# THIS IS WORKING
# using CNN with sliding window 2D CNN
# Imports
# https://www.analyticsvidhya.com/blog/2019/10/building-image-classification-models-cnn-pytorch/
########################################################################################################################
# importing the libraries
########################################################################################################################

import pickle

import tensorflow.compat.v1 as tf
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

tf.disable_v2_behavior()
import torch.nn as nn
from loadData import loadData
import seaborn.apionly as sns
import pandas as pd
import numpy as np

# for reading and displaying images
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d
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

classNamesLingual = ['17_DL', '17_L', '17_ML',
                     '18_DL', '18_L', '18_ML',
                     '19_DL', '19_L', '19_ML',
                     '20_DL', '20_L', '20_ML',
                     '21_DL', '21_L', '21_ML',
                     ]
########################################################################################################################
# import the data:
########################################################################################################################
(xbuccal, xlingual, ybuccal, ylingual) = loadData(mainFolder)

(xBuccal1, xBuccal2, xBuccal3, xBuccal4, xBuccal5, xBuccal6, xBuccal7, xBuccal8, xBuccal9, xBuccal10, xBuccal11,
 xBuccal12) = xbuccal
(xLingual1, xLingual2, xLingual3, xLingual4, xLingual5, xLingual6, xLingual7, xLingual8, xLingual9, xLingual10,
 xLingual11, xLingual12) = xlingual
(yBuccal1, yBuccal2, yBuccal3, yBuccal4, yBuccal5, yBuccal6, yBuccal7, yBuccal8, yBuccal9, yBuccal10, yBuccal11,
 yBuccal12) = ybuccal
(yLingual1, yLingual2, yLingual3, yLingual4, yLingual5, yLingual6, yLingual7, yLingual8, yLingual9, yLingual10,
 yLingual11, yLingual12) = ylingual

########################################################################################################################
# specify important features
########################################################################################################################
# selectedFeatures = list(range(24))#[0, 1, 19, 20, 21]#[0, 1, 2, 15, 16, 17, 18, 19, 20, 21, 22, 23] list(range(24))
selectedFeatures = [0, 1, 17, 19]  # this set of features with w = 8 gives best results for buccal 0.92%
########################################################################################################################
# SET PARAMETERS
########################################################################################################################
w = 12  # sliding window
num_classes = 15  # number of classes
learning_rate = 0.07  # learning rate of ADAM optimizer
n_epochs = 100  # defining the number of epochs
n_folds = 12
# kernel dimensions
k1 = 2
k2 = 2
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
    print("runnig fold " + str(fold) + " lingual...")
    tmpFoldsLingual = foldsLingual.copy()
    tmpLabelsLingual = labelsLingual.copy()
    Xts = tmpFoldsLingual.pop(fold)[:, selectedFeatures]
    yts = tmpLabelsLingual.pop(fold)

    Xtr = np.vstack(tmpFoldsLingual)[:, selectedFeatures]
    ytr = np.hstack(tmpLabelsLingual)

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
    train_y = torch.from_numpy(train_y)

    # shape of training data
    train_x.shape, train_y.shape  # [83569, 1, 8, 4] 83569 timepoints, 1 dummy dimension(channels), 8 -- window size, 4 -- number of features

    # converting validation images into torch format
    val_x = val_x.reshape(val_x.shape[0], 1, val_x.shape[1], val_x.shape[2])
    val_x = torch.from_numpy(val_x).float()

    # converting the target into torch format
    val_y = val_y.astype(int);
    val_y = torch.from_numpy(val_y)

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
                # PrintLayer(),#[74283, 1, 9, 4]
                Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
                # PrintLayer(),#[74283, 4, 9, 4]
                BatchNorm2d(4),
                # PrintLayer(),#[74283, 4, 9, 4]
                ReLU(inplace=True),
                MaxPool2d(kernel_size=(k1, k2)),
                # PrintLayer(),#[74283, 4, 4, 2]
                # Defining another 2D convolution layer
                Conv2d(4, 1, kernel_size=3, stride=1, padding=1),
                # PrintLayer(),#[74283, 4, 4, 2]
                BatchNorm2d(1),
                ReLU(inplace=True),
                # MaxPool2d(kernel_size=(k2_1,k2_2)),
                # PrintLayer()#[74283, 4, 2, 1]
            )

            self.linear_layers = Sequential(
                # PrintLayer(),  # Add Print layer for debug #[74283, 8]
                Linear(w, num_classes),  # 4*7*7 #4 *number = windowSize#4*1*6 for w=24, 4 * 1 * 12 for w=48
                # PrintLayer()  # Add Print layer for debug
            )

        # Defining the forward pass
        def forward(self, x):
            x = self.cnn_layers(x)
            x = x.view(x.size(0), -1)
            x = self.linear_layers(x)
            return x


    ########################################################################################################################
    # CNN training
    ########################################################################################################################
    # defining the model
    model = Net()
    # defining the optimizer
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # defining the loss function
    criterion = CrossEntropyLoss()
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
        loss_train = criterion(output_train, y_train)
        loss_val = criterion(output_val, y_val)

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

    model.eval()
    print("================================")
    print("Evaluating...")

    loss = 0

    # converting training images into torch format
    test_x = test_img.reshape(test_img.shape[0], 1, test_img.shape[1], test_img.shape[2])
    test_x = torch.from_numpy(test_x).float()

    # converting the target into torch format
    test_y = newYts.astype(int);
    test_y = torch.from_numpy(test_y)

    confusion = np.zeros((num_classes, num_classes))
    x_test, y_test = Variable(test_x), Variable(test_y)
    output = model(x_test)

    rows = y_test.cpu().numpy()
    cols = output.max(1)[1].cpu().numpy()

    tmpConfusion = np.zeros((num_classes, num_classes))

    loss_test = criterion(output, y_test)

    yhat = torch.argmax(output, dim=1).type(torch.FloatTensor)

    confusion = confusion_matrix(y_test.detach().numpy(), yhat.detach().numpy(), labels=range(num_classes))
    acc_test = np.trace(confusion) / np.sum(confusion)

    testAccuracy = torch.sum((torch.argmax(output, dim=1).type(torch.FloatTensor) == y_test).detach()).float() / \
                   y_test.shape[0]
    print('Test loss :', loss_test, '\ttest accuracy  :', str(testAccuracy))

    yhatPerFold.append(yhat)
    ytsPerFold.append(yts)
    a = classification_report(y_test, yhat, target_names=classNamesLingual, output_dict=True)
    reportsLingual.append(a)
    print("Confusion matrix on the test data")
    cm = confusion_matrix(y_test, yhat, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
    print(cm)
    confusionMatricesLingual.append(cm)
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
reports_12_fn = ('RESULTS/reportsCNN_Lingual_w_' + str(w) + '_.p')
with open(reports_12_fn, 'wb') as fp:
    pickle.dump(reportsLingual, fp)

confusionMatrices_12_fn = ('RESULTS/confusionMatricesCNN_Lingual_w_' + str(w) + '.p')
with open(confusionMatrices_12_fn, 'wb') as fp:
    pickle.dump(confusionMatricesLingual, fp)

normConfusionMatrices_12_fn = ('RESULTS/normConfusionMatricesCNN_Lingual_w_' + str(w) + '.p')
with open(normConfusionMatrices_12_fn, 'wb') as fp:
    pickle.dump(normConfusionMatrices, fp)

predictedLabels_12_fn = ('RESULTS/predLabelsCNN_Lingual_w_' + str(w) + '.p')
with open(predictedLabels_12_fn, 'wb') as fp:
    pickle.dump(yhatPerFold, fp)

actualLabels_12_fn = ('RESULTS/actualLabelsCNN_Lingual_w_' + str(w) + '.p')
with open(actualLabels_12_fn, 'wb') as fp:
    pickle.dump(ytsPerFold, fp)

valLoss_12_fn = ('RESULTS/valLossCNN_Lingual_w_' + str(w) + '_.p')
with open(valLoss_12_fn, 'wb') as fp:
    pickle.dump(valLossPerFold, fp)

valAccuracy_12_fn = ('RESULTS/valAccuracyCNN_Lingual_w_' + str(w) + '.p')
with open(valAccuracy_12_fn, 'wb') as fp:
    pickle.dump(valAccuracyPerFold, fp)

trainLoss_12_fn = ('RESULTS/trainLossCNN_Lingual_w_' + str(w) + '.p')
with open(trainLoss_12_fn, 'wb') as fp:
    pickle.dump(trainLossPerFold, fp)

trainAccuracy_12_fn = ('RESULTS/trainAccuracyCNN_Lingual_w_' + str(w) + '.p')
with open(trainAccuracy_12_fn, 'wb') as fp:
    pickle.dump(trainAccuracyPerFold, fp)

testAccuracy_12_fn = ('RESULTS/testAccuracyCNN_Lingual_w_' + str(w) + '.p')
with open(testAccuracy_12_fn, 'wb') as fp:
    pickle.dump(testAccuracyPerFold, fp)
########################################################################################################################
########################################################################################################################
# GET THE PERFORMANCE REPORTS
########################################################################################################################
classes = list(reportsLingual[0].keys())[0:num_classes]
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
# PLOT THE NORMALIZED CONFUSION MATRIX LINGUAL
########################################################################################################################
numFeatures = len(selectedFeatures)
# normalize confusion matrices
normalizedAvgCM = np.zeros((num_classes, num_classes))
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
plt.savefig('FIGURES/normCM_CNN_LINGUAL_six_regions_manual_' + str(numFeatures) + '_features_w_' + str(w) + '.pdf')
