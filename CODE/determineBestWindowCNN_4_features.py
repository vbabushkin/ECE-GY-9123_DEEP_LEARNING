# Determining the optimal sliding window size for proposed CNN model for 4 features selected with MI
########################################################################################################################
# importing the libraries
########################################################################################################################

import pickle

import tensorflow.compat.v1 as tf
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix

tf.disable_v2_behavior()
import torch.nn as nn
from loadData import loadData
import numpy as np
# for reading and displaying images
import matplotlib.pyplot as plt
# for creating validation set
from sklearn.model_selection import train_test_split
# for evaluating the model
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

classNamesBuccal = ['32_DB', '32_B', '32_MB',
                    '31_DB', '31_B', '31_MB',
                    '30_DB', '30_B', '30_MB',
                    '29_DB', '29_B', '29_MB',
                    '28_DB', '28_B', '28_MB',
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
# create dataset for training/testing
########################################################################################################################
# selectedFeatures = list(range(24))
selectedFeatures = [0, 1, 17, 19]

Xtr = np.vstack(
    (xBuccal1, xBuccal2, xBuccal3, xBuccal4, xBuccal5, xBuccal6, xBuccal7, xBuccal8, xBuccal9, xBuccal10, xBuccal11))
Xts = xBuccal12

Xtr = Xtr[:, selectedFeatures]
Xts = Xts[:, selectedFeatures]

scaler = preprocessing.StandardScaler().fit(Xtr)
Xtr = scaler.transform(Xtr)
Xts = scaler.transform(Xts)

ytr = np.hstack((yBuccal1.astype(int) - 1, yBuccal2.astype(int) - 1, yBuccal3.astype(int) - 1, yBuccal4.astype(int) - 1,
                 yBuccal5.astype(int) - 1, yBuccal6.astype(int) - 1, yBuccal7.astype(int) - 1, yBuccal8.astype(int) - 1,
                 yBuccal9.astype(int) - 1, yBuccal10.astype(int) - 1, yBuccal11.astype(int) - 1))
yts = yBuccal12.astype(int) - 1

########################################################################################################################
# SET PARAMETERS
########################################################################################################################
num_classes = 15  # number of classes
learning_rate = 0.07  # learning rate of ADAM optimizer
n_epochs = 100  # defining the number of epochs
# convolution kernel dimensions
k1 = 2
k2 = 2
########################################################################################################################
# create dataset for training/testing
########################################################################################################################
totalTestAccuracy = []
avgTrainLossTotal = []
avgValLossTotal = []
avgTrainAccuracyTotal = []
avgValAccuracyTotal = []

endTrainLossTotal = []
endValLossTotal = []
endTrainAccuracyTotal = []
endValAccuracyTotal = []

for w in range(2, 200, 2):  # sliding window size from w=2 to w=200
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
    train_x.shape, train_y.shape  # [83569, 1, 8, 4] 83569 timepoints, 1 dummy dimension(channels)

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


    # input size torch.Size([83569, 1, 8, 4])

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
        train_losses.append(float(loss_train))
        val_losses.append(float(loss_val))
        train_accuracies.append(float(trainAccuracy))
        val_accuracies.append(float(valAccuracy))

    avgTrainLossTotal.append(np.mean(train_losses))
    avgValLossTotal.append(np.mean(val_losses))
    avgTrainAccuracyTotal.append(np.mean(train_accuracies))
    avgValAccuracyTotal.append(np.mean(val_accuracies))

    endTrainLossTotal.append(train_losses[-1])
    endValLossTotal.append(val_losses[-1])
    endTrainAccuracyTotal.append(train_accuracies[-1])
    endValAccuracyTotal.append(val_accuracies[-1])
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
    totalTestAccuracy.append(float(testAccuracy))

########################################################################################################################
# SAVE
########################################################################################################################
totalTestAccuracy_fn = ('totalTestAccuracy.p')
with open(totalTestAccuracy_fn, 'wb') as fp:
    pickle.dump(totalTestAccuracy, fp)

avgTrainLoss_fn = ('avgTrainLoss.p')
with open(avgTrainLoss_fn, 'wb') as fp:
    pickle.dump(avgTrainLossTotal, fp)

avgValLoss_fn = ('avgValLoss.p')
with open(avgValLoss_fn, 'wb') as fp:
    pickle.dump(avgValLossTotal, fp)

avgTrainAccuracy_fn = ('avgTrainAccuracy.p')
with open(avgTrainAccuracy_fn, 'wb') as fp:
    pickle.dump(avgTrainAccuracyTotal, fp)

avgValAccuracy_fn = ('avgValAccuracy.p')
with open(avgValAccuracy_fn, 'wb') as fp:
    pickle.dump(avgValAccuracyTotal, fp)

endTrainLoss_fn = ('endTrainLoss.p')
with open(endTrainLoss_fn, 'wb') as fp:
    pickle.dump(endTrainLossTotal, fp)

endValLoss_fn = ('endValLoss.p')
with open(avgValLoss_fn, 'wb') as fp:
    pickle.dump(avgValLossTotal, fp)

endTrainAccuracy_fn = ('endTrainAccuracy.p')
with open(endTrainAccuracy_fn, 'wb') as fp:
    pickle.dump(endTrainAccuracyTotal, fp)

endValAccuracy_fn = ('endValAccuracy.p')
with open(endValAccuracy_fn, 'wb') as fp:
    pickle.dump(endValAccuracyTotal, fp)

########################################################################################################################
# GET THE REPORTS
########################################################################################################################

plt.figure(figsize=(12, 5))
x = np.arange(2, 2 * len(endValAccuracyTotal) + 1, 2)
plt.plot(x, endTrainLossTotal)
plt.plot(x, endValLossTotal)
plt.title('Loss metrics after 100 epochs vs window size')
plt.legend(['train', 'validation'])
plt.grid()
plt.xticks(np.arange(2, 2 * len(endValAccuracyTotal) + 1, 4), fontsize=8)
plt.xlim([2, 2 * len(endValAccuracyTotal)])
plt.xlabel("Window size, w")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig('FIGURES/endLossMetrics_window_4_features_200.pdf')

plt.figure(figsize=(14, 5))
x = np.arange(2, 2 * len(endValAccuracyTotal) + 1, 2)
plt.plot(x, endTrainAccuracyTotal)
plt.plot(x, endValAccuracyTotal)
plt.plot(x, totalTestAccuracy)
plt.title('Train accuracy metrics after 100 epochs and test accuracy vs window size')
plt.legend(['train', 'validation', 'test'])
plt.grid()
plt.xticks(np.arange(2, 2 * len(endValAccuracyTotal) + 1, 4), fontsize=8)
plt.yticks(np.arange(0, 1.1, 0.1), fontsize=8)
plt.xlim([2, 2 * len(endValAccuracyTotal)])
plt.xlabel("Window size, w")
plt.ylabel("Accuracy")
plt.ylim([0, 1])
plt.tight_layout()
plt.savefig('FIGURES/endAccuracyMetrics_window_4_features_200.pdf')
