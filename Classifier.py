import h5py
from keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
                          Activation, Dense, Dropout, ZeroPadding2D)
from keras.layers.normalization import BatchNormalization
from keras.models import load_model, Model
from keras.optimizers import Adam


class Classifier:
    def __init__(self, fold_best_model_path, features_key='features', threshold=0.5):
        # Classifier Adam to predict fall.
        self.classifier = load_model(fold_best_model_path + "urfd_fold_1.h5")  # 分类器是从fold_best_model_path中加载的。
        self.features_key = features_key  # 提取的H5特征文件中的键名
        self.threshold = threshold  # 判断阈值

    def classify(self, features_path):
        """

        :param features_path: h5 特征文件的路径，包含文件名。
        :return: 对应多组特征的真值向量。
        """
        h5features = h5py.File(features_path + "features.h5", 'r')
        tested_features = h5features[self.features_key]
        predicted = self.classifier.predict(tested_features)  # 输出预测向量，单元值为浮点数。
        for i in range(len(predicted)):
            if predicted[i] < self.threshold:
                predicted[i] = 0  # 小于阈值则为假
                print("您所看护的老人摔倒了！")
            else:
                predicted[i] = 1  # 大于阈值则为真
        return predicted
