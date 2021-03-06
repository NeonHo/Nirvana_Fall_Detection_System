from __future__ import print_function
from numpy.random import seed

seed(1)
import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import h5py
import scipy.io as sio
import cv2
import glob
import gc

from keras.models import load_model, Model, Sequential
from keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
                          Activation, Dense, Dropout, ZeroPadding2D)
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from keras.layers.advanced_activations import ELU


class Tester:
    def __init__(self, fold_best_model_path, threshold=0.5):
        self.threshold = threshold

        # Classifier Adam to predict fall.
        self.classifier = load_model(fold_best_model_path)  # 分类器是从fold_best_model_path中加载的。

    def extract_features(self):
        """

        :return: tested_features Use Conv1_1 to Conv5_3 to extract features from the only one example.
        """
        pass

    def classify_with_features(self, tested_features):
        """

        :param tested_features: the features extracted from one example.
        :return: Fall or not, if fall, return true, if not return false.
        """
        predicted = self.classifier.predict(tested_features)  # 输出预测向量，单元值为浮点数。
        for i in range(len(self.predicted)):
            if predicted[i] < self.threshold:
                predicted[i] = 0  # 小于阈值则为假
            else:
                predicted[i] = 1  # 大于阈值则为真
