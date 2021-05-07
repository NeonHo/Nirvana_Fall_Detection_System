from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox
from keras.models import load_model


class Classifier:
    def __init__(self, features_path, use_qt, features_key='features', threshold=0.5):
        # Classifier Adam to predict fall.
        self.classifier = None
        self.features_path = features_path
        self.features_key = features_key  # 提取的H5特征文件中的键名
        self.threshold = threshold  # 判断阈值
        self.use_qt = use_qt
        self.music_signal = None
        if self.use_qt:
            from RgbFlowSignal import RgbFlowSignal
            self.music_signal = RgbFlowSignal()

    def select_model(self, fold_best_model_path):
        """

        :param fold_best_model_path: h5 file path.
        :return:
        """
        self.classifier = load_model(fold_best_model_path)
        # 分类器是从fold_best_model_path中加载的。

    def classify_single(self, feature_input_queue):
        if self.classifier is None:
            QMessageBox.critical(QtWidgets.QWidget(), "模型问题", "您的模型未载入成功！", QMessageBox.Yes)
            return
        while True:
            # if not feature_input_queue.empty():
            sample_feature = feature_input_queue.get()
            predicted = self.classifier.predict(sample_feature)
            if predicted < self.threshold:
                # predicted = 0  # 小于阈值则为假，摔倒了。
                if self.use_qt:
                    self.music_signal.music.emit(True)
                    self.music_signal.judge_message.emit(True)
            else:
                # predicted = 1  # 大于阈值则为真，没摔到。
                if self.use_qt:
                    self.music_signal.judge_message.emit(False)
