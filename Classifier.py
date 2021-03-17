import threading
import h5py
import numpy as np
from keras.layers import (Input, Conv2D, MaxPooling2D, Flatten,
                          Activation, Dense, Dropout, ZeroPadding2D)
from keras.layers.normalization import BatchNormalization
from keras.models import load_model, Model
from keras.optimizers import Adam


class Classifier:
    def __init__(self, fold_best_model_path, features_path, features_key='features', threshold=0.5):
        # Classifier Adam to predict fall.
        self.classifier = load_model(fold_best_model_path + "urfd_fold_1.h5")  # 分类器是从fold_best_model_path中加载的。
        self.features_path = features_path
        self.features_key = features_key  # 提取的H5特征文件中的键名
        self.threshold = threshold  # 判断阈值

    def classify(self):
        h5features = h5py.File(self.features_path + "features.h5", 'r')
        tested_features = h5features[self.features_key]
        predicted = self.classifier.predict(tested_features)  # 输出预测向量，单元值为浮点数。
        for i in range(len(predicted)):
            if predicted[i] < self.threshold:
                predicted[i] = 0  # 小于阈值则为假
                print("第" + str(i) + "栈时，我发现您所看护的老人摔倒了！")
            else:
                predicted[i] = 1  # 大于阈值则为真
        return predicted

    def classify_single(self, sample_feature):
        predicted = self.classifier.predict(sample_feature)
        if predicted < self.threshold:
            predicted = 0  # 小于阈值则为假，摔倒了。
            print("我发现您所看护的老人摔倒了!!!!!!!")
        else:
            predicted = 1  # 大于阈值则为真，没摔到。
            print("正常活动中。")
        return predicted
