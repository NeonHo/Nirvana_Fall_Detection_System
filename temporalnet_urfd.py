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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# CHANGE THESE VARIABLES ---
data_folder = '/home/anunez/URFD_opticalflow/'  # 光流数据集
mean_file = '/home/anunez/flow_mean.mat'  # 归一化参数
vgg_16_weights = 'weights.h5'  # 用UCF101训练得到的VGG-16的权重
save_features = False  # 如果想要训练自己的数据集，那么设为True，这会覆盖原有的特征。
save_plots = True  # ？？？ 是否保存训练后的Accuracy的图像。

# Set to 'True' if you want to restore a previous trained models
# Training is skipped and test is done
use_checkpoint = False  # Set to True or False ？？？
# --------------------------

best_model_path = 'models/'  # 训练出的模型存储的路径
plots_folder = 'plots/'  # 训练过程中的图像的存储路径
checkpoint_path = best_model_path + 'fold_'  # ？？？ 检查点文件的存储路径

saved_files_folder = 'saved_features/'  # URFD数据集中提取的特征和标签的存放路径
features_file = saved_files_folder + 'features_urfd_tf.h5'  # URFD中提取的特征
labels_file = saved_files_folder + 'labels_urfd_tf.h5'  # URFD中提取特征一一对应的标签
features_key = 'features'  # 如果要训练自己的数据集，这些就有用，hdf5文件钟特征一栏的键名
labels_key = 'labels'  # 如果要训练自己的数据集，这些就有用，hdf5文件钟标签一栏的键名

L = 10  # RGB图片组成的堆栈的尺寸
num_features = 4096  # 特征的数量
batch_norm = True  # 是否需要批量归一化
learning_rate = 0.0001  # 拟合过程中的学习率
mini_batch_size = 64  # 最小批的尺寸
weight_0 = 1  # 多分类问题中只有摔倒这一类有独一无二的权重
epochs = 3000  # 世代的数量
use_validation = False  # ？？？ 是否使用验证集
# After the training stops, use train+validation to train for 1 epoch
use_val_for_training = False  # ？？？ 是否使用验证集去训练
val_size = 100  # ？？？ 验证集的数量
# Threshold to classify between positive and negative
threshold = 0.5  # 二值分类的判别阈值，低于阈值为0，高于或等于阈值为1

# Name of the experiment URFD_学习率_批处理尺寸_是否要批处理正则化_摔倒类的权重
exp = 'urfd_lr{}_batchs{}_batchnorm{}_w0_{}'.format(learning_rate, mini_batch_size, batch_norm, weight_0)


def plot_training_info(case, metrics, save, history):
    """
    Function to create plots for train and validation loss and accuracy
    Input:
    * case: name for the plot, an 'accuracy.png' or 'loss.png' will be concatenated after the name.
    * metrics: list of metrics to store: 'loss' and/or 'accuracy'
    * save: boolean to store the plots or only show them.
    * history: History object returned by the Keras fit function.
    """
    val = False
    if 'val_acc' in history and 'val_loss' in history:
        val = True
    plt.ioff()
    if 'accuracy' in metrics:
        fig = plt.figure()
        plt.plot(history['accuracy'])
        if val: plt.plot(history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        if val:
            plt.legend(['train', 'val'], loc='upper left')
        else:
            plt.legend(['train'], loc='upper left')
        if save == True:
            plt.savefig(case + 'accuracy.png')
            plt.gcf().clear()
        else:
            plt.show()
        plt.close(fig)

    # summarize history for loss
    if 'loss' in metrics:
        fig = plt.figure()
        plt.plot(history['loss'])
        if val: plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        # plt.ylim(1e-3, 1e-2)
        plt.yscale("log")
        if val:
            plt.legend(['train', 'val'], loc='upper left')
        else:
            plt.legend(['train'], loc='upper left')
        if save == True:
            plt.savefig(case + 'loss.png')
            plt.gcf().clear()
        else:
            plt.show()
        plt.close(fig)


def generator(list1, lits2):
    """
    Auxiliar generator: returns the ith element of both given list with each call to next()
    """
    for x, y in zip(list1, lits2):  # 将两个列表的对应元素组成一一对应的元组。
        yield x, y


def saveFeatures(feature_extractor, features_file, labels_file, features_key, labels_key):
    """
    Function to
    load the optical flow stacks,
    do a feed-forward through the feature extractor (VGG16)
    and
    store the output feature vectors in the file 'features_file' and the labels in 'labels_file'.
    Input:
    * feature_extractor: model VGG16 until the fc6 layer. 是直到fc6层的VGG16模型
    * features_file: path to the hdf5 file where the extracted features are going to be stored
    * labels_file: path to the hdf5 file where the labels of the features are going to be stored
    * features_key: name of the key for the hdf5 file to store the features
    * labels_key: name of the key for the hdf5 file to store the labels
    """

    class0 = 'Falls'
    class1 = 'NotFalls'

    # Load the mean file to subtract to the images
    d = sio.loadmat(mean_file)
    flow_mean = d['image_mean']  # 用来归一化的参数

    # Fill the folders and classes arrays with all the paths to the data
    folders, classes = [], []  # folders 统计有多少个达标的录像样本，只要通过该样本的x_images数是否超过10个就能判断，没必要再计数y_images.
    fall_videos = [f for f in os.listdir(data_folder + class0)
                   if os.path.isdir(os.path.join(data_folder + class0, f))]  # 将标记为摔倒的文件夹中的众多样本录像文件夹组成一个列表。
    fall_videos.sort()  # 对这些标记摔倒的录像文件夹进行排序
    for fall_video in fall_videos:
        x_images = glob.glob(data_folder + class0 + '/' + fall_video + '/flow_x*.jpg')  # 将每个样本文件夹中的所有光流图片全拿出来。
        if int(len(x_images)) >= 10:  # 光流能组成超过10帧就是符合标准，可以将路径放到文件列表和类列表中。并且类都赋值为0即表示摔倒。
            folders.append(data_folder + class0 + '/' + fall_video)
            classes.append(0)

    not_fall_videos = [f for f in os.listdir(data_folder + class1)
                       if os.path.isdir(os.path.join(data_folder + class1, f))]  # 将标记为非摔倒的文件夹中的众多样本录像文件夹组成一个列表。
    not_fall_videos.sort()  # 对这些标记非摔倒的录像文件夹进行排序
    for not_fall_video in not_fall_videos:
        x_images = glob.glob(data_folder + class1 + '/' + not_fall_video + '/flow_x*.jpg')
        if int(len(x_images)) >= 10:  # 光流能组成超过10帧就是符合标准，可以将路径放到文件列表和类列表中。并且类都赋值为1即表示非摔倒。
            folders.append(data_folder + class1 + '/' + not_fall_video)
            classes.append(1)

    # Total amount of stacks, with sliding window = num_images-L+1
    nb_total_stacks = 0
    for folder in folders:
        x_images = glob.glob(folder + '/flow_x*.jpg')  # 搜索所有的横向光流图
        nb_total_stacks += len(x_images) - L + 1  # 计算共需要多少个栈

    # File to store the extracted features and datasets to store them
    # IMPORTANT NOTE: 'w' mode totally erases previous data
    h5features = h5py.File(features_file, 'w')  # 完全清除特征文件中的内容重新写入
    h5labels = h5py.File(labels_file, 'w')  # 完全清楚标签文件中的内容重新写入
    # 预计在特征数据集中写入nb_total_stacks×4096个特征数据。Shape:(10652, 4096)
    dataset_features = h5features.create_dataset(features_key, shape=(nb_total_stacks, num_features), dtype='float64')
    # 预计在标签数据集中写入nb_total_stacks个特征数据。Shape:(10652, 1)
    dataset_labels = h5labels.create_dataset(labels_key, shape=(nb_total_stacks, 1), dtype='float64')
    cont = 0

    for folder, label in zip(folders, classes):
        x_images = glob.glob(folder + '/flow_x*.jpg')
        x_images.sort()
        y_images = glob.glob(folder + '/flow_y*.jpg')
        y_images.sort()
        nb_stacks = len(x_images) - L + 1
        # Here nb_stacks optical flow stacks will be stored
        flow = np.zeros(shape=(224, 224, 2 * L, nb_stacks), dtype=np.float64)
        gen = generator(x_images, y_images)
        for i in range(len(x_images)):
            flow_x_file, flow_y_file = gen.next()
            img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)
            img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)
            # Assign an image i to the jth stack in the kth position, but also
            # in the j+1th stack in the k+1th position and so on
            # (for sliding window)
            for s in list(reversed(range(min(10, i + 1)))):
                if i - s < nb_stacks:
                    flow[:, :, 2 * s, i - s] = img_x
                    flow[:, :, 2 * s + 1, i - s] = img_y
            del img_x, img_y
            gc.collect()

        # Subtract mean 减去均值，做到归一化。
        flow = flow - np.tile(flow_mean[..., np.newaxis], (1, 1, 1, flow.shape[3]))
        flow = np.transpose(flow, (3, 0, 1, 2))
        predictions = np.zeros((flow.shape[0], num_features), dtype=np.float64)  # 创建预测矩阵，nb_stacks×特征数4096
        truth = np.zeros((flow.shape[0], 1), dtype=np.float64)
        # Process each stack: do the feed-forward pass and store in the hdf5 file the output
        for i in range(flow.shape[0]):
            prediction = feature_extractor.predict(np.expand_dims(flow[i, ...], 0))  # 进行预测。
            predictions[i, ...] = prediction  # 预测值放入列表
            truth[i] = label  # 真实值放入列表
        dataset_features[cont:cont + flow.shape[0], :] = predictions
        dataset_labels[cont:cont + flow.shape[0], :] = truth
        cont += flow.shape[0]
    h5features.close()
    h5labels.close()  # 两个文件的流都关闭，达到写入的效果。


def exam_video(feature_extractor, video_path, ground_truth):
    # Load the mean file to subtract to the images
    d = sio.loadmat(mean_file)
    flow_mean = d['image_mean']

    x_images = glob.glob(video_path + '/flow_x*.jpg')  # 从系统中搜索到的横轴上的光流图像集
    x_images.sort()  # 对搜索到的图像进行排序
    y_images = glob.glob(video_path + '/flow_y*.jpg')  # 从系统中搜索到的纵轴上的光流图像集
    y_images.sort()  # 对搜索到的图像进行排序
    nb_stacks = len(x_images) - L + 1  # 非重叠采样的个数，每个非重叠的采样就是一个长2L的栈
    # Here nb_stacks optical flow stacks will be stored
    flow = np.zeros(shape=(224, 224, 2 * L, nb_stacks), dtype=np.float64)  # nb_stacks个栈 224×224×2L 数据类型是64位浮点数，
    gen = generator(x_images, y_images)  # 产生一个生成<x_image, y_image>元组列表的生成器，用于下面循环的迭代。
    for i in range(len(x_images)):  # 第i帧图像
        flow_x_file, flow_y_file = gen.next()  # 获得第i帧图像的横轴纵轴元组。
        img_x = cv2.imread(flow_x_file, cv2.IMREAD_GRAYSCALE)  # 读入第i张横轴上灰度图片
        img_y = cv2.imread(flow_y_file, cv2.IMREAD_GRAYSCALE)  # 读入第i张纵轴上灰度图片
        # Assign an image i to the jth stack in the kth position, but also
        # in the j+1th stack in the k+1th position and so on
        # (for sliding window)
        for s in list(reversed(range(min(10, i + 1)))):  # L = 10 一个栈的尺寸VS当前图像的索引，取最小值构成一个索引表。
            if i - s < nb_stacks:  # 栈个数限制以内可以赋值。
                # 将横向和纵向的两张图片一前一后放在栈中。
                flow[:, :, 2 * s, i - s] = img_x
                flow[:, :, 2 * s + 1, i - s] = img_y
        del img_x, img_y  # 删除的是变量名，数据还在原来的内存中。
        gc.collect()  # 垃圾回收
    flow = flow - np.tile(flow_mean[..., np.newaxis], (1, 1, 1, flow.shape[3]))
    # 先将归一化矩阵flow_mean重复nb_stacks次，即[224, 224, 2L] × nb_stacks，让flow矩阵减flow_mean得到nb_stacks个被归一化的栈构成矩阵。
    flow = np.transpose(flow, (3, 0, 1, 2))  # 将每个元素的3号坐标提前到0号。
    predictions = np.zeros((flow.shape[0], num_features), dtype=np.float64)  # 创建预测矩阵，nb_stacks×特征数4096
    truth = np.zeros((flow.shape[0], 1), dtype=np.float64)  # 真值矩阵 nb_stacks×1
    # Process each stack: do the feed-forward pass
    for i in range(flow.shape[0]):  # 进行nb_stacks次循环
        prediction = feature_extractor.predict(np.expand_dims(flow[i, ...], 0))  # 输入i号测试样本的4096个特征值，并进行预测
        predictions[i, ...] = prediction  # 预测出结果并赋值给predictions向量。
        truth[i] = ground_truth  # ？？？ground_truth是什么？
    return predictions, truth


def main():
    # ========================================================================
    # VGG-16 ARCHITECTURE
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

    # ========================================================================
    # FEATURE EXTRACTION
    # ========================================================================
    if save_features:
        saveFeatures(model, features_file, labels_file, features_key, labels_key)

    # ========================================================================
    # TRAINING
    # ========================================================================  

    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)  # Adam梯度下降：训练的参数：学习率；β1和β2；ε阈值，
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])  # 多分类损失函数

    h5features = h5py.File(features_file, 'r')
    h5labels = h5py.File(labels_file, 'r')

    # X_full will contain all the feature vectors extracted
    # from optical flow images
    X_full = h5features[features_key]
    _y_full = np.asarray(h5labels[labels_key])

    zeroes_full = np.asarray(np.where(_y_full == 0)[0])
    ones_full = np.asarray(np.where(_y_full == 1)[0])
    zeroes_full.sort()
    ones_full.sort()

    # Use a 5 fold cross-validation
    kf_falls = KFold(n_splits=5, shuffle=True)
    kf_falls.get_n_splits(X_full[zeroes_full, ...])

    kf_nofalls = KFold(n_splits=5, shuffle=True)
    kf_nofalls.get_n_splits(X_full[ones_full, ...])

    sensitivities = []  # 评估性能用的5个参数。
    specificities = []
    fars = []
    mdrs = []
    accuracies = []

    fold_number = 1
    # CROSS-VALIDATION: Stratified partition of the dataset into train/test sets
    for ((train_index_falls, test_index_falls), (train_index_nofalls, test_index_nofalls)) in zip(
        kf_falls.split(X_full[zeroes_full, ...]),
        kf_nofalls.split(X_full[ones_full, ...])
    ):

        train_index_falls = np.asarray(train_index_falls)
        test_index_falls = np.asarray(test_index_falls)
        train_index_nofalls = np.asarray(train_index_nofalls)
        test_index_nofalls = np.asarray(test_index_nofalls)

        X = np.concatenate((
            X_full[zeroes_full, ...][train_index_falls, ...],
            X_full[ones_full, ...][train_index_nofalls, ...]
        ))
        _y = np.concatenate((
            _y_full[zeroes_full, ...][train_index_falls, ...],
            _y_full[ones_full, ...][train_index_nofalls, ...]
        ))
        X_test = np.concatenate((
            X_full[zeroes_full, ...][test_index_falls, ...],
            X_full[ones_full, ...][test_index_nofalls, ...]
        ))
        y_test = np.concatenate((
            _y_full[zeroes_full, ...][test_index_falls, ...],
            _y_full[ones_full, ...][test_index_nofalls, ...]
        ))

        if use_validation:
            # Create a validation subset from the training set
            zeroes = np.asarray(np.where(_y == 0)[0])
            ones = np.asarray(np.where(_y == 1)[0])

            zeroes.sort()
            ones.sort()

            trainval_split_0 = StratifiedShuffleSplit(n_splits=1, test_size=val_size / 2, random_state=7)
            indices_0 = trainval_split_0.split(X[zeroes, ...], np.argmax(_y[zeroes, ...], 1))
            trainval_split_1 = StratifiedShuffleSplit(n_splits=1, test_size=val_size / 2, random_state=7)
            indices_1 = trainval_split_1.split(X[ones, ...], np.argmax(_y[ones, ...], 1))
            train_indices_0, val_indices_0 = indices_0.next()
            train_indices_1, val_indices_1 = indices_1.next()

            X_train = np.concatenate([X[zeroes, ...][train_indices_0, ...], X[ones, ...][train_indices_1, ...]], axis=0)
            y_train = np.concatenate([_y[zeroes, ...][train_indices_0, ...], _y[ones, ...][train_indices_1, ...]], axis=0)
            X_val = np.concatenate([X[zeroes, ...][val_indices_0, ...], X[ones, ...][val_indices_1, ...]], axis=0)
            y_val = np.concatenate([_y[zeroes, ...][val_indices_0, ...], _y[ones, ...][val_indices_1, ...]], axis=0)
        else:
            X_train = X
            y_train = _y

        # Balance the number of positive and negative samples so that
        # there is the same amount of each of them
        all0 = np.asarray(np.where(y_train == 0)[0])
        all1 = np.asarray(np.where(y_train == 1)[0])

        if len(all0) < len(all1):
            all1 = np.random.choice(all1, len(all0), replace=False)
        else:
            all0 = np.random.choice(all0, len(all1), replace=False)
        allin = np.concatenate((all0.flatten(), all1.flatten()))
        allin.sort()
        X_train = X_train[allin, ...]
        y_train = y_train[allin]

        # ==================== CLASSIFIER ========================
        extracted_features = Input(shape=(num_features,), dtype='float32', name='input')
        if batch_norm:  # 批量归一化
            x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(extracted_features)
            x = Activation('relu')(x)
        else:
            x = ELU(alpha=1.0)(extracted_features)

        x = Dropout(0.9)(x)  # 以0.9的概率进行丢弃正则化。
        x = Dense(4096, name='fc2', kernel_initializer='glorot_uniform')(x)
        # 4096 output units, Xavier uniform initializer
        if batch_norm:
            x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
            x = Activation('relu')(x)
        else:
            x = ELU(alpha=1.0)(x)
        x = Dropout(0.8)(x)  # 以0.8的概率进行丢弃正则化。
        x = Dense(1, name='predictions', kernel_initializer='glorot_uniform')(x)
        # 1 output unit, Xavier uniform initializer
        x = Activation('sigmoid')(x)  # sigmoid function.

        classifier = Model(inputs=extracted_features, outputs=x, name='classifier')
        # 分类器，输入提取的特征，输出真值判断。
        fold_best_model_path = best_model_path + 'urfd_fold_{}.h5'.format(fold_number)
        # models/urfd_fold_1.h5 是分类器本身，我已经训练出来了。
        classifier.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
        # Adam优化器， 代价函数二值交叉混合，用精确度度量。

        if not use_checkpoint:
            # ==================== TRAINING ========================     
            # weighting of each class: only the fall class gets a different weight
            class_weight = {0: weight_0, 1: 1}

            callbacks = None
            if use_validation:
                # callback definition
                metric = 'val_loss'
                e = EarlyStopping(monitor=metric, min_delta=0, patience=100,
                                  mode='auto')
                c = ModelCheckpoint(fold_best_model_path, monitor=metric,
                                    save_best_only=True,
                                    save_weights_only=False, mode='auto')
                callbacks = [e, c]
            validation_data = None
            if use_validation:
                validation_data = (X_val, y_val)
            _mini_batch_size = mini_batch_size
            if mini_batch_size == 0:
                _mini_batch_size = X_train.shape[0]

            history = classifier.fit(
                X_train, y_train,
                validation_data=validation_data,
                batch_size=_mini_batch_size,
                epochs=epochs,
                shuffle=True,
                class_weight=class_weight,
                callbacks=callbacks
            )

            if not use_validation:  # 如果不使用验证集。
                classifier.save(fold_best_model_path)

            plot_training_info(plots_folder + exp, ['accuracy', 'loss'], save_plots, history.history)

            if use_validation and use_val_for_training:  # 如果使用验证集去训练。
                classifier = load_model(fold_best_model_path)

                # Use full training set (training+validation)
                X_train = np.concatenate((X_train, X_val), axis=0)
                y_train = np.concatenate((y_train, y_val), axis=0)

                history = classifier.fit(
                    X_train, y_train,
                    validation_data=validation_data,
                    batch_size=_mini_batch_size,
                    epochs=epochs,
                    shuffle='batch',
                    class_weight=class_weight,
                    callbacks=callbacks
                )

                classifier.save(fold_best_model_path)  # 将分类器保存到fold_best_model_path中。

        # ==================== EVALUATION ========================     

        # Load best model
        print('Model loaded from checkpoint')
        classifier = load_model(fold_best_model_path)  # 分类器是从fold_best_model_path中加载的。

        predicted = classifier.predict(np.asarray(X_test))  # 输出预测向量，单元值为浮点数。
        for i in range(len(predicted)):
            if predicted[i] < threshold:
                predicted[i] = 0  # 小于阈值则为假
            else:
                predicted[i] = 1  # 大于阈值则为真
        # Array of predictions 0/1
        predicted = np.asarray(predicted).astype(int)  # 转换为整型
        # Compute metrics and print them
        cm = confusion_matrix(y_test, predicted, labels=[0, 1])
        tp = cm[0][0]
        fn = cm[0][1]
        fp = cm[1][0]
        tn = cm[1][1]
        tpr = tp / float(tp + fn)
        fpr = fp / float(fp + tn)
        fnr = fn / float(fn + tp)
        tnr = tn / float(tn + fp)
        precision = tp / float(tp + fp)
        recall = tp / float(tp + fn)
        specificity = tn / float(tn + fp)
        f1 = 2 * float(precision * recall) / float(precision + recall)
        accuracy = accuracy_score(y_test, predicted)

        print('FOLD {} results:'.format(fold_number))
        print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp, tn, fp, fn))
        print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(tpr, tnr, fpr, fnr))
        print('Sensitivity/Recall: {}'.format(recall))
        print('Specificity: {}'.format(specificity))
        print('Precision: {}'.format(precision))
        print('F1-measure: {}'.format(f1))
        print('Accuracy: {}'.format(accuracy))

        # Store the metrics for this epoch
        sensitivities.append(tp / float(tp + fn))
        specificities.append(tn / float(tn + fp))
        fars.append(fpr)
        mdrs.append(fnr)
        accuracies.append(accuracy)
        fold_number += 1

    print('5-FOLD CROSS-VALIDATION RESULTS ===================')
    print("Sensitivity: %.2f%% (+/- %.2f%%)" % (np.mean(sensitivities) * 100., np.std(sensitivities) * 100.))
    print("Specificity: %.2f%% (+/- %.2f%%)" % (np.mean(specificities) * 100., np.std(specificities) * 100.))
    print("FAR: %.2f%% (+/- %.2f%%)" % (np.mean(fars) * 100., np.std(fars) * 100.))
    print("MDR: %.2f%% (+/- %.2f%%)" % (np.mean(mdrs) * 100., np.std(mdrs) * 100.))
    print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracies) * 100., np.std(accuracies) * 100.))


if __name__ == '__main__':
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    main()
