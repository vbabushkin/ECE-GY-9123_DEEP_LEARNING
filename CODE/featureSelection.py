# Feature selection based on several tests. MI is the most relevant test since it can deal with non-linear data
########################################################################################################################
# importing the libraries
########################################################################################################################
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from loadData import loadData
import pandas as pd
import numpy as np
from sklearn.feature_selection import f_classif
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
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
# CREATE DATASETS
########################################################################################################################

Xtr_Buccal = np.vstack((xBuccal1, xBuccal2, xBuccal3, xBuccal4, xBuccal5, xBuccal6, xBuccal7, xBuccal8, xBuccal9,
                        xBuccal10, xBuccal11, xBuccal12))
ytr_Buccal = np.hstack((yBuccal1, yBuccal2, yBuccal3, yBuccal4, yBuccal5, yBuccal6, yBuccal7, yBuccal8, yBuccal9,
                        yBuccal10, yBuccal11, yBuccal12))
Xtr_Lingual = np.vstack((xLingual1, xLingual2, xLingual3, xLingual4, xLingual5, xLingual6, xLingual7, xLingual8,
                         xLingual9, xLingual10, xLingual11, xLingual12))
ytr_Lingual = np.hstack((yLingual1, yLingual2, yLingual3, yLingual4, yLingual5, yLingual6, yLingual7, yLingual8,
                         yLingual9, yLingual10, yLingual11, yLingual12))

########################################################################################################################
# PERFORM FEATURE SELECTION BUCCAL
########################################################################################################################

########################################################################################################################
# Bagged decision trees like Random Forest and Extra Trees can be used to estimate the importance of features.
########################################################################################################################
# Feature Importance with Extra Trees Classifier
X = Xtr_Buccal
Y = ytr_Buccal
# feature extraction
model = ExtraTreesClassifier(n_estimators=10)
model.fit(X, Y)
print(model.feature_importances_)

importanceCoeffTrees = model.feature_importances_
normImportanceCoeffTrees = (importanceCoeffTrees - np.min(importanceCoeffTrees)) / (
            np.max(importanceCoeffTrees) - np.min(importanceCoeffTrees))

ind = np.arange(1, 25)  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(list(range(1, 25)), normImportanceCoeffTrees)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xticks(ind)
ax.set_xticklabels(featuresNames, rotation=35, fontsize="10", ha='right', va='top')
ax.tick_params(axis='x', which='major', pad=0.1)
plt.title("Feature Selection with Extra Trees Classifier for Buccal", fontsize="14")
ax.set_ylabel('Normalized feature importance coefficient', fontsize="12")
plt.tight_layout()
plt.savefig('FIGURES/features_importances_extra_trees_buccal.pdf')

dfscores = pd.DataFrame(importanceCoeffTrees)
dfcolumns = pd.DataFrame(featuresNames)
# concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
print(featureScores.nlargest(10, 'Score'))  # print 10 best features

########################################################################################################################
# Feature Selection with Univariate Statistical Tests -- Normalized ANOVA F-value
########################################################################################################################
X = Xtr_Buccal
Y = ytr_Buccal
# feature extraction
test = SelectKBest(score_func=f_classif, k='all')
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5, :])
importanceCoeff = np.nan_to_num(fit.scores_)
normImportanceCoeff = (importanceCoeff - np.min(importanceCoeff)) / (np.max(importanceCoeff) - np.min(importanceCoeff))
ind = np.arange(1, 25)  # the x locations for the groups
width = 0.35  # the width of the bars
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(list(range(1, 25)), normImportanceCoeff)
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xticks(ind)
ax.set_xticklabels(featuresNames, rotation=35, fontsize="10", ha='right', va='top')
ax.tick_params(axis='x', which='major', pad=0.1)
plt.title("Feature Selection with Univariate Statistical Tests for Buccal", fontsize="14")
ax.set_ylabel('Normalized ANOVA F-value', fontsize="12")
plt.tight_layout()
plt.savefig('FIGURES/features_importances_f_buccal.pdf')

dfscores = pd.DataFrame(importanceCoeff)
dfcolumns = pd.DataFrame(featuresNames)
# concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
print(featureScores.nlargest(10, 'Score'))  # print 10 best features

########################################################################################################################
# Feature Selection with Univariate Statistical Tests -- Normalized Mutual Info
########################################################################################################################
X = Xtr_Buccal
Y = ytr_Buccal
# feature extraction
test = SelectKBest(score_func=mutual_info_classif, k='all')
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5, :])

importanceCoeff = np.nan_to_num(fit.scores_)
normImportanceCoeff = (importanceCoeff - np.min(importanceCoeff)) / (np.max(importanceCoeff) - np.min(importanceCoeff))
ind = np.arange(1, 25)  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(list(range(1, 25)), normImportanceCoeff)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xticks(ind)
ax.set_xticklabels(featuresNames, rotation=35, fontsize="10", ha='right', va='top')
ax.tick_params(axis='x', which='major', pad=0.1)
plt.title("Feature Selection with Univariate Statistical Tests", fontsize="14")
ax.set_ylabel('Normalized Mutual Info', fontsize="12")
plt.tight_layout()
plt.savefig('FIGURES/features_importances_MI_buccal.pdf')

dfscores = pd.DataFrame(importanceCoeff)
dfcolumns = pd.DataFrame(featuresNames)
# concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
print(featureScores.nlargest(10, 'Score'))  # print 10 best features

########################################################################################################################
# PERFORM FEATURE SELECTION LINGUAL
########################################################################################################################

########################################################################################################################
# Bagged decision trees like Random Forest and Extra Trees can be used to estimate the importance of features.
########################################################################################################################
# Feature Importance with Extra Trees Classifier
X = Xtr_Lingual
Y = ytr_Lingual
# feature extraction
model = ExtraTreesClassifier(n_estimators=10)
model.fit(X, Y)
print(model.feature_importances_)

dfscores = pd.DataFrame(model.feature_importances_)
dfcolumns = pd.DataFrame(featuresNames)
# concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
print(featureScores.nlargest(10, 'Score'))  # print 10 best features
importanceCoeffTrees = model.feature_importances_
normImportanceCoeffTrees = (importanceCoeffTrees - np.min(importanceCoeffTrees)) / (
            np.max(importanceCoeffTrees) - np.min(importanceCoeffTrees))

ind = np.arange(1, 25)  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(list(range(1, 25)), normImportanceCoeffTrees)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xticks(ind)
ax.set_xticklabels(featuresNames, rotation=35, fontsize="10", ha='right', va='top')
ax.tick_params(axis='x', which='major', pad=0.1)
# ax.set_ylim([0,1.1])
# ax.legend()
plt.title("Feature Selection with Extra Trees Classifier for Lingual", fontsize="14")
ax.set_ylabel('Normalized feature importance coefficient', fontsize="12")
plt.tight_layout()
plt.savefig('FIGURES/features_importances_extra_trees_lingual.pdf')

########################################################################################################################
# Feature Selection with Univariate Statistical Tests -- Normalized ANOVA F-value
########################################################################################################################
X = Xtr_Lingual
Y = ytr_Lingual
# feature extraction
test = SelectKBest(score_func=f_classif, k='all')
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5, :])

importanceCoeff = np.nan_to_num(fit.scores_)
normImportanceCoeff = (importanceCoeff - np.min(importanceCoeff)) / (np.max(importanceCoeff) - np.min(importanceCoeff))

ind = np.arange(1, 25)  # the x locations for the groups
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(list(range(1, 25)), normImportanceCoeff)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xticks(ind)
ax.set_xticklabels(featuresNames, rotation=35, fontsize="10", ha='right', va='top')
ax.tick_params(axis='x', which='major', pad=0.1)
plt.title("Feature Selection with Univariate Statistical Tests for Lingual", fontsize="14")
ax.set_ylabel('Normalized ANOVA F-value', fontsize="12")
plt.tight_layout()
plt.savefig('FIGURES/features_importances_f_lingual.pdf')

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(featuresNames)
# concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
print(featureScores.nlargest(10, 'Score'))  # print 10 best features

########################################################################################################################
# Feature Selection with Univariate Statistical Tests -- Normalized Mutual Info
########################################################################################################################
X = Xtr_Lingual
Y = ytr_Lingual
# feature extraction
test = SelectKBest(score_func=mutual_info_classif, k='all')
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(X)
# summarize selected features
print(features[0:5, :])

importanceCoeff = np.nan_to_num(fit.scores_)
normImportanceCoeff = (importanceCoeff - np.min(importanceCoeff)) / (np.max(importanceCoeff) - np.min(importanceCoeff))

ind = np.arange(1, 25)  # the x locations for the groups
width = 0.35  # the width of the bars
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(list(range(1, 25)), normImportanceCoeff)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xticks(ind)
ax.set_xticklabels(featuresNames, rotation=35, fontsize="10", ha='right', va='top')
ax.tick_params(axis='x', which='major', pad=0.1)
plt.title("Feature Selection with Univariate Statistical Tests", fontsize="14")
ax.set_ylabel('Normalized Mutual Info', fontsize="12")
plt.tight_layout()
plt.savefig('FIGURES/features_importances_MI_lingual.pdf')

dfscores = pd.DataFrame(importanceCoeff)
dfcolumns = pd.DataFrame(featuresNames)
# concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
print(featureScores.nlargest(10, 'Score'))  # print 10 best features
