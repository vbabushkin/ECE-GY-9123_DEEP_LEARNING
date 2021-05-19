# for loading the pocket data
# parameters: str mainFolder -- path to the main directory where the NEW_DATA folder is stored
import numpy as np


def loadPocketData(mainFolder):
    ########################################################################################################################
    # IMPORT BUCCAL DATA
    ########################################################################################################################
    with open(mainFolder + 'NEW_DATA/RECORDINGS/BUCCAL/rec1_data_full.npy', 'rb') as f:
        xBuccal1 = np.load(f)
    with open(mainFolder + 'NEW_DATA/RECORDINGS/BUCCAL/rec1_labels_pockets.npy', 'rb') as f:
        yBuccal1 = np.load(f)

    with open(mainFolder + 'NEW_DATA/RECORDINGS/BUCCAL/rec2_data_full.npy', 'rb') as f:
        xBuccal2 = np.load(f)
    with open(mainFolder + 'NEW_DATA/RECORDINGS/BUCCAL/rec2_labels_pockets.npy', 'rb') as f:
        yBuccal2 = np.load(f)

    with open(mainFolder + 'NEW_DATA/RECORDINGS/BUCCAL/rec3_data_full.npy', 'rb') as f:
        xBuccal3 = np.load(f)
    with open(mainFolder + 'NEW_DATA/RECORDINGS/BUCCAL/rec3_labels_pockets.npy', 'rb') as f:
        yBuccal3 = np.load(f)

    with open(mainFolder + 'NEW_DATA/RECORDINGS/BUCCAL/rec4_data_full.npy', 'rb') as f:
        xBuccal4 = np.load(f)
    with open(mainFolder + 'NEW_DATA/RECORDINGS/BUCCAL/rec4_labels_pockets.npy', 'rb') as f:
        yBuccal4 = np.load(f)

    with open(mainFolder + 'NEW_DATA/RECORDINGS/BUCCAL/rec5_data_full.npy', 'rb') as f:
        xBuccal5 = np.load(f)
    with open(mainFolder + 'NEW_DATA/RECORDINGS/BUCCAL/rec5_labels_pockets.npy', 'rb') as f:
        yBuccal5 = np.load(f)

    with open(mainFolder + 'NEW_DATA/RECORDINGS/BUCCAL/rec6_data_full.npy', 'rb') as f:
        xBuccal6 = np.load(f)
    with open(mainFolder + 'NEW_DATA/RECORDINGS/BUCCAL/rec6_labels_pockets.npy', 'rb') as f:
        yBuccal6 = np.load(f)

    with open(mainFolder + 'NEW_DATA/RECORDINGS/BUCCAL/rec7_data_full.npy', 'rb') as f:
        xBuccal7 = np.load(f)
    with open(mainFolder + 'NEW_DATA/RECORDINGS/BUCCAL/rec7_labels_pockets.npy', 'rb') as f:
        yBuccal7 = np.load(f)

    with open(mainFolder + 'NEW_DATA/RECORDINGS/BUCCAL/rec8_data_full.npy', 'rb') as f:
        xBuccal8 = np.load(f)
    with open(mainFolder + 'NEW_DATA/RECORDINGS/BUCCAL/rec8_labels_pockets.npy', 'rb') as f:
        yBuccal8 = np.load(f)

    with open(mainFolder + 'NEW_DATA/RECORDINGS/BUCCAL/rec9_data_full.npy', 'rb') as f:
        xBuccal9 = np.load(f)
    with open(mainFolder + 'NEW_DATA/RECORDINGS/BUCCAL/rec9_labels_pockets.npy', 'rb') as f:
        yBuccal9 = np.load(f)

    with open(mainFolder + 'NEW_DATA/RECORDINGS/BUCCAL/rec10_data_full.npy', 'rb') as f:
        xBuccal10 = np.load(f)
    with open(mainFolder + 'NEW_DATA/RECORDINGS/BUCCAL/rec10_labels_pockets.npy', 'rb') as f:
        yBuccal10 = np.load(f)

    with open(mainFolder + 'NEW_DATA/RECORDINGS/BUCCAL/rec11_data_full.npy', 'rb') as f:
        xBuccal11 = np.load(f)
    with open(mainFolder + 'NEW_DATA/RECORDINGS/BUCCAL/rec11_labels_pockets.npy', 'rb') as f:
        yBuccal11 = np.load(f)

    with open(mainFolder + 'NEW_DATA/RECORDINGS/BUCCAL/rec12_data_full.npy', 'rb') as f:
        xBuccal12 = np.load(f)
    with open(mainFolder + 'NEW_DATA/RECORDINGS/BUCCAL/rec12_labels_pockets.npy', 'rb') as f:
        yBuccal12 = np.load(f)

    ########################################################################################################################
    # IMPORT LINGUAL DATA
    ########################################################################################################################

    with open(mainFolder + 'NEW_DATA/RECORDINGS/LINGUAL/rec1_data_full.npy', 'rb') as f:
        xLingual1 = np.load(f)
    with open(mainFolder + 'NEW_DATA/RECORDINGS/LINGUAL/rec1_labels_pockets.npy', 'rb') as f:
        yLingual1 = np.load(f)

    with open(mainFolder + 'NEW_DATA/RECORDINGS/LINGUAL/rec2_data_full.npy', 'rb') as f:
        xLingual2 = np.load(f)
    with open(mainFolder + 'NEW_DATA/RECORDINGS/LINGUAL/rec2_labels_pockets.npy', 'rb') as f:
        yLingual2 = np.load(f)

    with open(mainFolder + 'NEW_DATA/RECORDINGS/LINGUAL/rec3_data_full.npy', 'rb') as f:
        xLingual3 = np.load(f)
    with open(mainFolder + 'NEW_DATA/RECORDINGS/LINGUAL/rec3_labels_pockets.npy', 'rb') as f:
        yLingual3 = np.load(f)

    with open(mainFolder + 'NEW_DATA/RECORDINGS/LINGUAL/rec4_data_full.npy', 'rb') as f:
        xLingual4 = np.load(f)
    with open(mainFolder + 'NEW_DATA/RECORDINGS/LINGUAL/rec4_labels_pockets.npy', 'rb') as f:
        yLingual4 = np.load(f)

    with open(mainFolder + 'NEW_DATA/RECORDINGS/LINGUAL/rec5_data_full.npy', 'rb') as f:
        xLingual5 = np.load(f)
    with open(mainFolder + 'NEW_DATA/RECORDINGS/LINGUAL/rec5_labels_pockets.npy', 'rb') as f:
        yLingual5 = np.load(f)

    with open(mainFolder + 'NEW_DATA/RECORDINGS/LINGUAL/rec6_data_full.npy', 'rb') as f:
        xLingual6 = np.load(f)
    with open(mainFolder + 'NEW_DATA/RECORDINGS/LINGUAL/rec6_labels_pockets.npy', 'rb') as f:
        yLingual6 = np.load(f)

    with open(mainFolder + 'NEW_DATA/RECORDINGS/LINGUAL/rec7_data_full.npy', 'rb') as f:
        xLingual7 = np.load(f)
    with open(mainFolder + 'NEW_DATA/RECORDINGS/LINGUAL/rec7_labels_pockets.npy', 'rb') as f:
        yLingual7 = np.load(f)

    with open(mainFolder + 'NEW_DATA/RECORDINGS/LINGUAL/rec8_data_full.npy', 'rb') as f:
        xLingual8 = np.load(f)
    with open(mainFolder + 'NEW_DATA/RECORDINGS/LINGUAL/rec8_labels_pockets.npy', 'rb') as f:
        yLingual8 = np.load(f)

    with open(mainFolder + 'NEW_DATA/RECORDINGS/LINGUAL/rec9_data_full.npy', 'rb') as f:
        xLingual9 = np.load(f)
    with open(mainFolder + 'NEW_DATA/RECORDINGS/LINGUAL/rec9_labels_pockets.npy', 'rb') as f:
        yLingual9 = np.load(f)

    with open(mainFolder + 'NEW_DATA/RECORDINGS/LINGUAL/rec10_data_full.npy', 'rb') as f:
        xLingual10 = np.load(f)
    with open(mainFolder + 'NEW_DATA/RECORDINGS/LINGUAL/rec10_labels_pockets.npy', 'rb') as f:
        yLingual10 = np.load(f)

    with open(mainFolder + 'NEW_DATA/RECORDINGS/LINGUAL/rec11_data_full.npy', 'rb') as f:
        xLingual11 = np.load(f)
    with open(mainFolder + 'NEW_DATA/RECORDINGS/LINGUAL/rec11_labels_pockets.npy', 'rb') as f:
        yLingual11 = np.load(f)

    with open(mainFolder + 'NEW_DATA/RECORDINGS/LINGUAL/rec12_data_full.npy', 'rb') as f:
        xLingual12 = np.load(f)
    with open(mainFolder + 'NEW_DATA/RECORDINGS/LINGUAL/rec12_labels_pockets.npy', 'rb') as f:
        yLingual12 = np.load(f)
    xbuccal = (
    xBuccal1, xBuccal2, xBuccal3, xBuccal4, xBuccal5, xBuccal6, xBuccal7, xBuccal8, xBuccal9, xBuccal10, xBuccal11,
    xBuccal12)
    xlingual = (
    xLingual1, xLingual2, xLingual3, xLingual4, xLingual5, xLingual6, xLingual7, xLingual8, xLingual9, xLingual10,
    xLingual11, xLingual12)
    ybuccal = (
    yBuccal1, yBuccal2, yBuccal3, yBuccal4, yBuccal5, yBuccal6, yBuccal7, yBuccal8, yBuccal9, yBuccal10, yBuccal11,
    yBuccal12)
    ylingual = (
    yLingual1, yLingual2, yLingual3, yLingual4, yLingual5, yLingual6, yLingual7, yLingual8, yLingual9, yLingual10,
    yLingual11, yLingual12)
    return (xbuccal, xlingual, ybuccal, ylingual)
