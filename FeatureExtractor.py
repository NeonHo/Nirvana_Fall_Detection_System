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


class FeatureExtractor:
    def __init__(self, network_weight_path, num_features=4096):
        """

        :param network_weight_path: 用于提取特征的神经网络序列模型的参数文件路径。
        :param num_features: 需要提取出的特征维数。
        """
        # ========================================================================
        # VGG16 model from Conv1_1 to fc6 layer to extract features.
        # ========================================================================

        self.vgg_16_weights = network_weight_path
        self.num_features = num_features

        self.model = Sequential()  # 多个网络层的线性堆叠

        self.model.add(ZeroPadding2D((1, 1), input_shape=(224, 224, 20)))  # 输入尺寸224×224×20，矩阵四周填充一排0，后面的层会自动推断尺寸。
        self.model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_1'))  # 卷积核个数64，卷积核尺寸3×3，激活函数ReLU，名称convx_y
        self.model.add(ZeroPadding2D((1, 1)))  # 矩阵四周填充一排0
        self.model.add(Conv2D(64, (3, 3), activation='relu', name='conv1_2'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))  # 池化窗口的大小2×2，滑动步长横向纵向2.

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_1'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(128, (3, 3), activation='relu', name='conv2_2'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_1'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_2'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(256, (3, 3), activation='relu', name='conv3_3'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_1'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_2'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu', name='conv4_3'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_1'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_2'))
        self.model.add(ZeroPadding2D((1, 1)))
        self.model.add(Conv2D(512, (3, 3), activation='relu', name='conv5_3'))
        self.model.add(MaxPooling2D((2, 2), strides=(2, 2)))

        self.model.add(Flatten())  # 从卷积层到全连接层的过渡，把多维的输入一维化。
        self.model.add(Dense(self.num_features, name='fc6', kernel_initializer='glorot_uniform'))  # 输出特征向量。

        # ========================================================================
        # WEIGHT INITIALIZATION
        # ========================================================================
        layers_caffe = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1',
                        'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3',
                        'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']
        h5 = h5py.File(self.vgg_16_weights, 'r')  # 读入VGG16在UCF101下训练的权重

        layer_dict = dict([(layer.name, layer) for layer in self.model.layers])  # 将每一层与层的名字对应，方便用名字搜索特定曾。

        # Copy the weights stored in the 'vgg_16_weights' file to the
        # feature extractor part of the VGG16
        for layer in layers_caffe[:-3]:  # 所有的卷积层都进行权重赋值。
            w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
            w2 = np.transpose(np.asarray(w2), (2, 3, 1, 0))
            w2 = w2[::-1, ::-1, :, :]
            b2 = np.asarray(b2)
            layer_dict[layer].set_weights((w2, b2))

        # Copy the weights of the first fully-connected layer (fc6)
        layer = layers_caffe[-3]  # 所有的全连接层进行权重赋值。
        w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
        w2 = np.transpose(np.asarray(w2), (1, 0))
        b2 = np.asarray(b2)
        layer_dict[layer].set_weights((w2, b2))
