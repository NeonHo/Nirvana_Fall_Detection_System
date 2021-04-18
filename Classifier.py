from keras.models import load_model
# import winsound

from RgbFlowSignal import RgbFlowSignal


class Classifier:
    def __init__(self, fold_best_model_path, features_path, features_key='features', threshold=0.5):
        # Classifier Adam to predict fall.
        self.classifier = load_model(fold_best_model_path + "combined_final_model.h5")  # 分类器是从fold_best_model_path中加载的。
        self.features_path = features_path
        self.features_key = features_key  # 提取的H5特征文件中的键名
        self.threshold = threshold  # 判断阈值
        self.duration = 1000  # millisecond
        self.freq = 500  # Hz
        self.music_signal = RgbFlowSignal()

    def classify_single(self, feature_input_queue):
        while True:
            sample_feature = feature_input_queue.get()
            predicted = self.classifier.predict(sample_feature)
            if predicted < self.threshold:
                predicted = 0  # 小于阈值则为假，摔倒了。
                # winsound.Beep(self.freq, self.duration)
                self.music_signal.music.emit(True)
                print("被看护者摔倒了!!!!!!!!!")
            else:
                predicted = 1  # 大于阈值则为真，没摔到。
                print("正常。")
