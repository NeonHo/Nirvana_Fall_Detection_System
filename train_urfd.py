from __future__ import print_function
from numpy.random import seed
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import os
import h5py
from keras.models import load_model, Model, Sequential
from keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, Activation, Dense, Dropout, ZeroPadding2D)
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, StratifiedShuffleSplit
from keras.layers.advanced_activations import ELU

seed(1)
matplotlib.use('Agg')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# CHANGE THESE VARIABLES ---
mean_file = 'flow_mean.mat'  # 归一化参数
vgg_16_weights = 'weights.h5'  # 用UCF101训练得到的VGG-16的权重
save_plots = True  # ？？？ 是否保存训练后的Accuracy的图像。

# Set to 'True' if you want to restore a previous trained models Training is skipped and test is done
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


def main():
    # ========================================================================
    # TRAINING
    # ========================================================================
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)  # Adam梯度下降：训练的参数：学习率；β1和β2；ε阈值，

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

        if use_validation:  # 如果使用验证集，就将训练集和验证集分开，
            # Create a validation subset from the training set
            zeroes = np.asarray(np.where(_y == 0)[0])
            ones = np.asarray(np.where(_y == 1)[0])

            zeroes.sort()
            ones.sort()
            # 将完整的训练集作为一整份，其中50个作为验证集，剩下的都作为训练集。
            trainval_split_0 = StratifiedShuffleSplit(n_splits=1, test_size=val_size / 2, random_state=7)
            indices_0 = trainval_split_0.split(X[zeroes, ...], np.argmax(_y[zeroes, ...], 1))
            trainval_split_1 = StratifiedShuffleSplit(n_splits=1, test_size=val_size / 2, random_state=7)
            indices_1 = trainval_split_1.split(X[ones, ...], np.argmax(_y[ones, ...], 1))
            train_indices_0, val_indices_0 = indices_0.next()
            train_indices_1, val_indices_1 = indices_1.next()

            X_train = np.concatenate([X[zeroes, ...][train_indices_0, ...], X[ones, ...][train_indices_1, ...]], axis=0)
            y_train = np.concatenate([_y[zeroes, ...][train_indices_0, ...], _y[ones, ...][train_indices_1, ...]],
                                     axis=0)
            X_val = np.concatenate([X[zeroes, ...][val_indices_0, ...], X[ones, ...][val_indices_1, ...]], axis=0)
            y_val = np.concatenate([_y[zeroes, ...][val_indices_0, ...], _y[ones, ...][val_indices_1, ...]], axis=0)
        else:  # 不使用验证集，而是直接将整个训练集拿来训练。
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
                e = EarlyStopping(monitor=metric, min_delta=0, patience=100, mode='auto')
                c = ModelCheckpoint(fold_best_model_path, monitor=metric, save_best_only=True, save_weights_only=False,
                                    mode='auto')
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
