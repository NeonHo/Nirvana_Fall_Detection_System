from __future__ import print_function

import time

import h5py
import matplotlib
import numpy as np
import scipy.io as sio
from keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, ZeroPadding2D)
from keras.models import Sequential
from numpy.random import seed
import os


os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
seed(1)
matplotlib.use('Agg')


class FeatureExtractor:
    def __init__(self, network_weight_path, mean_file_path, optical_frame_path, features_path, width, height, use_qt,
                 num_features=4096, features_key='features'):
        """

        :param network_weight_path: 用于提取特征的神经网络序列模型的参数文件路径，包含文件名。
        :param num_features: 需要提取出的特征维数。
        """
        # ========================================================================
        # VGG16 model from Conv1_1 to fc6 layer to extract features.
        # ========================================================================

        self.ucf101_vgg16_weight_address = network_weight_path + "weights.h5"
        self.mean_file_path = mean_file_path + "flow_mean.mat"
        self.num_features = num_features
        self.stack_length = 10  # RGB图片组成的堆栈的尺寸
        self.features_key = features_key  # 提取的H5特征文件中的键名
        self.optical_frame_path = optical_frame_path
        self.features_path = features_path
        self.use_qt = use_qt
        # the flow array will be zeros at the head
        self.img_count = 0
        self.width = width
        self.height = height
        self.flow_stack = np.zeros((self.width, self.height, 2 * self.stack_length, 1), dtype=np.float64)
        # signal
        if self.use_qt:
            from RgbFlowSignal import RgbFlowSignal
            self.signal = RgbFlowSignal()

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
        h5 = h5py.File(self.ucf101_vgg16_weight_address, 'r')  # 读入VGG16在UCF101下训练的权重

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
        layer = layers_caffe[-3]  # fc6全连接层进行权重赋值。
        w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
        w2 = np.transpose(np.asarray(w2), (1, 0))
        b2 = np.asarray(b2)
        layer_dict[layer].set_weights((w2, b2))

        # Load the mean file to subtract to the images
        d = sio.loadmat(self.mean_file_path)
        self.flow_mean = d['image_mean']  # 用来归一化的参数

    def add_flow_images_couple(self, img_x, img_y):
        """
        append img_x and img_y to the x and y flow numpy.array and update the stack size.
        :param img_x: x 方向上的光流图像
        :param img_y: y 方向上的光流图像
        :return:
        """
        self.img_count += 1
        if self.img_count <= self.stack_length:
            self.flow_stack[:, :, 2 * (self.img_count - 1), 0] = img_x
            self.flow_stack[:, :, 2 * (self.img_count - 1) + 1, 0] = img_y
        else:
            self.flow_stack = np.delete(self.flow_stack, 0, axis=2)
            self.flow_stack = np.delete(self.flow_stack, 0, axis=2)
            img_x = np.expand_dims(img_x, axis=2)
            img_x = np.expand_dims(img_x, axis=3)
            self.flow_stack = np.append(self.flow_stack, img_x, axis=2)
            img_y = np.expand_dims(img_y, axis=2)
            img_y = np.expand_dims(img_y, axis=3)
            self.flow_stack = np.append(self.flow_stack, img_y, axis=2)

    def extract(self, flow_input_queue, feature_output_queue):
        """

        :param feature_output_queue:
        :param flow_input_queue:
        :return:
        """
        self.img_count = 0
        while True:
            if not flow_input_queue.empty():
                flow_x, flow_y = flow_input_queue.get()
                self.add_flow_images_couple(flow_x, flow_y)
                # print("optical flow images:\t" + str(self.img_count) + "压入光流栈。")
                # and (self.img_count % self.stack_length == 0)
                if (self.img_count >= self.stack_length) and (self.img_count % (self.stack_length / 2) == 0):
                    # Subtract mean 减去均值，做到归一化。
                    self.signal.per_stack.emit(self.img_count)
                    self.flow_stack = self.flow_stack - np.tile(self.flow_mean[..., np.newaxis],
                                                                (1, 1, 1, self.flow_stack.shape[3]))
                    flow = np.transpose(self.flow_stack, (3, 0, 1, 2))
                    features = self.model.predict(np.expand_dims(flow[0, ...], 0))  # 进行预测。
                    feature_output_queue.put(features)
