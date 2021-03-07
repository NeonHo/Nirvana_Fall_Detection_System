from __future__ import print_function

import gc

import cv2
from numpy.random import seed
import numpy as np
import matplotlib
import h5py
from keras.models import Sequential
from keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, ZeroPadding2D)
import scipy.io as sio
import glob

seed(1)
matplotlib.use('Agg')


def generator(list1, list2):
    """
    Auxiliary generator: returns the ith element of both given list with each call to next()
    """
    for x, y in zip(list1, list2):  # 将两个列表的对应元素组成一一对应的元组。
        yield x, y


class FeatureExtractor:
    def __init__(self, network_weight_path, mean_file_path, num_features=4096, features_key='features'):
        """

        :param network_weight_path: 用于提取特征的神经网络序列模型的参数文件路径，包含文件名。
        :param num_features: 需要提取出的特征维数。
        """
        # ========================================================================
        # VGG16 model from Conv1_1 to fc6 layer to extract features.
        # ========================================================================

        self.ucf101_vgg16_weight_address = network_weight_path
        self.mean_file_path = mean_file_path
        self.num_features = num_features
        self.stack_length = 10  # RGB图片组成的堆栈的尺寸
        self.features_key = features_key  # 提取的H5特征文件中的键名

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

    def extract(self, optical_frame_path, features_path):
        """

        :param optical_frame_path: 光流图像所在文件路径，不包含图片名称。
        :param features_path: 提取出的特征的文件路径，不包含H5文件名。
        :return: None
        """
        # Load the mean file to subtract to the images
        d = sio.loadmat(self.mean_file_path)
        flow_mean = d['image_mean']  # 用来归一化的参数

        x_images = glob.glob(optical_frame_path + '\\flow_x*.jpg')  # 将每个样本文件夹中的所有光流图片全拿出来。
        y_images = glob.glob(optical_frame_path + '\\flow_y*.jpg')

        nb_stacks = len(x_images) - self.stack_length + 1  # 计算共需要多少个栈

        # File to store the extracted features and datasets to store them
        # IMPORTANT NOTE: 'w' mode totally erases previous data
        h5features = h5py.File(features_path + "\\features.h5", 'w')  # 完全清除特征文件中的内容重新写入
        # 预计在特征数据集中写入nb_total_stacks×4096个特征数据。Shape:(nb_total_stacks, 4096)
        dataset_features = h5features.create_dataset(self.features_key, shape=(nb_stacks, self.num_features),
                                                     dtype='float64')

        # Here nb_stacks optical flow stacks will be stored
        flow = np.zeros(shape=(224, 224, 2 * self.stack_length, nb_stacks), dtype=np.float64)
        gen = generator(x_images, y_images)
        for i in range(len(x_images)):
            flow_x_file, flow_y_file = next(gen)
            img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
            img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
            # Assign an image i to the jth stack in the kth position, but also in the j+1th stack in the k+1th
            # position and so on (for sliding window) 滑动窗口的目的是能从单个录像中密集地采样，并提取多组特征。
            for s in list(reversed(range(min(10, i + 1)))):
                if i - s < nb_stacks:
                    flow[:, :, 2 * s, i - s] = img_x
                    flow[:, :, 2 * s + 1, i - s] = img_y
            del img_x, img_y
            gc.collect()

        # Subtract mean 减去均值，做到归一化。
        flow = flow - np.tile(flow_mean[..., np.newaxis], (1, 1, 1, flow.shape[3]))
        flow = np.transpose(flow, (3, 0, 1, 2))
        predictions = np.zeros((flow.shape[0], self.num_features), dtype=np.float64)  # 创建预测矩阵，nb_stacks×特征数4096
        truth = np.zeros((flow.shape[0], 1), dtype=np.float64)
        for i in range(flow.shape[0]):
            prediction = self.model.predict(np.expand_dims(flow[i, ...], 0))  # 进行预测。
            predictions[i, ...] = prediction  # 预测值放入列表
        dataset_features[0 + flow.shape[0], :] = predictions
        h5features.close()
