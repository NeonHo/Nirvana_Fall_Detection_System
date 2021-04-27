import time
from queue import Queue
from threading import Thread

import timer
from PyQt5 import uic, QtMultimedia
from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QImage, QPixmap
import cv2
from PyQt5.QtWidgets import QApplication

from Classifier import Classifier
from FeatureExtractor import FeatureExtractor
from OpticalGenerator import OpticalGenerator
from VideoGrapher import Videographer


class MonitorWindow:
    def __init__(self):
        super(MonitorWindow, self).__init__()
        # parameters
        self.width = 224
        self.height = 224
        self.bound = 20
        self.sound_level = 50
        self.threshold = 0.5

        # paths
        self.video_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\videos\\"
        self.flow_image_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\flow\\"
        self.features_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\features\\"
        self.weight_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\"
        self.mean_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\"
        self.model_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\URFD_results\\"
        self.ui0_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\windows\\untitled.ui"
        self.jpg_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\alarm.jpg"
        self.sound_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\alarm.mp3"
        self.avi_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\test_ground\\cam7_15.avi"
        self.ends_jpg_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\ends.jpg"
        self.ui = uic.loadUi(self.ui0_path)
        self.rgb = None
        self.flow = None
        alarm_img = cv2.imread(self.jpg_path)
        alarm_img = cv2.resize(alarm_img, (112, 112))
        alarm_img = QImage(alarm_img, 112, 112, QImage.Format_RGB888)
        alarm_img = QPixmap.fromImage(alarm_img)
        self.ui.label_sound.setPixmap(alarm_img)
        self.ui.label_sound.show()
        url = QUrl.fromLocalFile(self.sound_path)
        content = QtMultimedia.QMediaContent(url)
        self.player = QtMultimedia.QMediaPlayer()
        self.player.setMedia(content)
        self.ui.label_Threshold_2.setText(str(self.threshold))

        # component
        self.videographer = Videographer(self.video_path, self.width, self.height)
        self.feature_extractor = FeatureExtractor(self.weight_path, self.mean_path, self.flow_image_path,
                                                  self.features_path, self.width, self.height)
        self.optical_generator = OpticalGenerator(self.video_path, self.flow_image_path, self.bound, self.width,
                                                  self.height, self.feature_extractor.stack_length)
        self.classifier = Classifier(self.model_path, self.features_path, threshold=self.threshold)

        # queues
        self.flow_queue = Queue(1)
        self.feature_queue = Queue(1)

        # signals
        self.optical_generator.rgb_flow_signal.ends.connect(self.show_ends)
        self.optical_generator.rgb_flow_signal.frames.connect(self.show_frame)
        self.classifier.music_signal.music.connect(self.play_music)
        self.ui.horizontalSlider_sound.valueChanged.connect(self.update_music_volume)
        self.ui.horizontalSlider_threshold.valueChanged.connect(self.update_predict_threshold)
        self.ui.pushButton.clicked.connect(self.play_stop)
        self.ui.frame_fall.setStyleSheet("QFrame { background-color: Green }")
        self.exam()

    def show_frame(self, frames):
        self.rgb = frames[0]
        self.flow = frames[1]
        frame_left = QImage(self.rgb, self.width, self.height, QImage.Format_RGB888)
        frame_right = QImage(self.flow, self.width, self.height, QImage.Format_RGB888)
        pix_left = QPixmap.fromImage(frame_left)
        pix_right = QPixmap.fromImage(frame_right)
        self.ui.label_rgb.setPixmap(pix_left)
        self.ui.label_fl.setPixmap(pix_right)
        self.ui.label_rgb.show()
        self.ui.label_fl.show()

    def play_music(self, fall):
        if fall:
            self.player.setVolume(self.sound_level)
            self.player.play()
            self.blink_light()

    def show_ends(self, ends):
        if not ends:
            img = cv2.imread(self.ends_jpg_path)
            img = cv2.resize(img, (self.width, self.height))
            frame = QImage(img, self.width, self.height, QImage.Format_RGB888)
            pix = QPixmap.fromImage(frame)
            self.ui.label_rgb.setPixmap(pix)
            self.ui.label_fl.setPixmap(pix)

    def blink_light(self):
        self.ui.frame_fall.setStyleSheet("QFrame { background-color: Red }")

    def play_stop(self):
        self.player.stop()
        self.ui.frame_fall.setStyleSheet("QFrame { background-color: Green }")

    def update_predict_threshold(self):
        self.threshold = self.ui.horizontalSlider_threshold.value() / 10.0
        self.classifier.threshold = self.threshold
        self.ui.label_Threshold_2.setText(str(self.threshold))
        self.ui.label_Threshold_2.show()

    def update_music_volume(self):
        self.sound_level = self.ui.horizontalSlider_sound.value()
        self.player.setVolume(self.sound_level)

    def exam(self) -> None:
        # Thread(target=self.videographer.capture_video, args=(self.frame_queue,)).start()
        Thread(target=self.optical_generator.generate_optical_flow_tvl1, args=(self.avi_path, self.flow_queue)).start()
        Thread(target=self.feature_extractor.extract, args=(self.flow_queue, self.feature_queue,)).start()
        Thread(target=self.classifier.classify_single, args=(self.feature_queue,)).start()
