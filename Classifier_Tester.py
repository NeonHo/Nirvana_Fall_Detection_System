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
        # ========================================================================
        # VGG16 model from Conv1_1 to fc6 layer to extract features.
        # ========================================================================

        model = Sequential()  # 多个网络层的线性堆叠

        model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 20)))  # 输入尺寸224×224×20，矩阵四周填充一排0，后面的层会自动推断尺寸。
        model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1'))  # 卷积核个数64，卷积核尺寸3×3，激活函数ReLU，名称convx_y
        model.add(ZeroPadding2D((1, 1)))  # 矩阵四周填充一排0
        model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))  # 池化窗口的大小2×2，滑动步长横向纵向2.

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_1'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_2'))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_3'))
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        model.add(Flatten())  # 从卷积层到全连接层的过渡，把多维的输入一维化。
        model.add(Dense(num_features, name='fc6', kernel_initializer='glorot_uniform'))  # 输出特征向量。

        # ========================================================================
        # WEIGHT INITIALIZATION
        # ========================================================================
        layerscaffe = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1',
                       'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3',
                       'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']
        h5 = h5py.File(vgg_16_weights, 'r')  # 读入VGG16在UCF101下训练的权重

        layer_dict = dict([(layer.name, layer) for layer in model.layers])  # 将每一层与层的名字对应，方便用名字搜索特定曾。

        # Copy the weights stored in the 'vgg_16_weights' file to the
        # feature extractor part of the VGG16
        for layer in layerscaffe[:-3]:  # 所有的卷积层都进行权重赋值。
            w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
            w2 = np.transpose(np.asarray(w2), (2, 3, 1, 0))
            w2 = w2[::-1, ::-1, :, :]
            b2 = np.asarray(b2)
            layer_dict[layer].set_weights((w2, b2))

        # Copy the weights of the first fully-connected layer (fc6)
        layer = layerscaffe[-3]  # 所有的全连接层进行权重赋值。
        w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
        w2 = np.transpose(np.asarray(w2), (1, 0))
        b2 = np.asarray(b2)
        layer_dict[layer].set_weights((w2, b2))
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
