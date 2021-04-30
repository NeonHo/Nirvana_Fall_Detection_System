import os
from queue import Queue
from threading import Thread

import cv2
from PyQt5 import uic
from PyQt5.QtCore import QDir
from PyQt5.QtGui import QImage, QPixmap, QTextCursor
from PyQt5.QtWidgets import QFileDialog

from Classifier import Classifier
from FeatureExtractor import FeatureExtractor
from MusicPlayer import MusicPlayer
from OpticalGenerator import OpticalGenerator
from VideoGrapher import Videographer


class MonitorWindow:
    def __init__(self):
        super(MonitorWindow, self).__init__()
        # parameters
        self.is_windows = True
        self.width = 224
        self.height = 224
        self.bound = 20
        self.sound_level = 50
        self.threshold = 0.5

        # paths
        separator = "\\" if self.is_windows else "/"
        work_path = os.getcwd()
        self.video_path = work_path + separator + "otherFiles" + separator + "videos" + separator
        self.flow_image_path = work_path + separator + "otherFiles" + separator + "flow" + separator
        self.features_path = work_path + separator + "otherFiles" + separator + "features" + separator
        self.weight_path = work_path + separator + "otherFiles" + separator
        self.mean_path = work_path + separator + "otherFiles" + separator
        self.model_path = work_path + separator + "otherFiles" + separator + "URFD_results" + separator
        self.ui0_path = work_path + separator + "windows" + separator + "untitled.ui"
        self.jpg_path = work_path + separator + "otherFiles" + separator + "alarm.jpg"
        self.sound_path = work_path + separator + "otherFiles" + separator + "alarm.mp3"
        self.avi_directory_path = work_path + separator + "test_ground"
        self.avi_path = work_path + separator + "test_ground" + separator + "cam7_2.avi"
        self.ends_jpg_path = work_path + separator + "otherFiles" + separator + "ends.jpg"
        self.ui = uic.loadUi(self.ui0_path)
        self.rgb = None
        self.flow = None
        alarm_img = cv2.imread(self.jpg_path)
        alarm_img = cv2.resize(alarm_img, (112, 112))
        alarm_img = QImage(alarm_img, 112, 112, QImage.Format_RGB888)
        alarm_img = QPixmap.fromImage(alarm_img)
        self.ui.label_sound.setPixmap(alarm_img)
        self.ui.label_sound.show()
        self.player = MusicPlayer(self.sound_path)
        self.ui.label_Threshold_2.setText(str(self.threshold))

        # component
        self.videographer = Videographer(self.video_path, self.width, self.height)
        self.feature_extractor = FeatureExtractor(self.weight_path, self.mean_path, self.flow_image_path,
                                                  self.features_path, self.width, self.height, use_qt=True)
        self.optical_generator = OpticalGenerator(self.video_path, self.flow_image_path, self.bound, self.width,
                                                  self.height, self.feature_extractor.stack_length, use_qt=True,
                                                  is_windows=self.is_windows)
        self.classifier = Classifier(self.model_path, self.features_path, use_qt=True, threshold=self.threshold)

        # queues
        self.flow_queue = Queue(1)
        self.feature_queue = Queue(1)

        # signals
        self.optical_generator.rgb_flow_signal.not_ends.connect(self.show_ends)
        self.optical_generator.rgb_flow_signal.frames.connect(self.show_frame)
        self.optical_generator.rgb_flow_signal.per_flow.connect(self.update_flow_index)
        self.feature_extractor.signal.per_stack.connect(self.update_stack_index)
        self.classifier.music_signal.music.connect(self.play_music)
        self.classifier.music_signal.judge_message.connect(self.show_fall_message)
        self.ui.horizontalSlider_sound.valueChanged.connect(self.update_music_volume)
        self.ui.horizontalSlider_threshold.valueChanged.connect(self.update_predict_threshold)
        self.ui.pushButton.clicked.connect(self.play_stop)
        self.ui.frame_fall.setStyleSheet("QFrame { background-color: Green }")
        self.ui.radioButton.toggled.connect(self.show_select_mp4)

        # threads
        self.optical_thread = Thread(target=self.optical_generator.generate_optical_flow_tvl1, args=(self.avi_path, self.flow_queue))
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

    def update_stack_index(self, index):
        self.ui.FlowStackTextEdit.moveCursor(QTextCursor.End)
        self.ui.FlowStackTextEdit.insertPlainText('\n' + str(index))

    def update_flow_index(self, index):
        self.ui.FlowProcessTextEdit.moveCursor(QTextCursor.End)
        self.ui.FlowProcessTextEdit.insertPlainText('\n' + str(index))

    def play_music(self, fall):
        if fall:
            self.player.player.setVolume(self.sound_level)
            self.player.player.play()

    def show_fall_message(self, fall):
        if fall:
            self.ui.frame_fall.setStyleSheet("QFrame { background-color: Red }")
            self.ui.label_fall.setText("Fall!!!")
            self.ui.label_fall.show()
        else:
            self.ui.label_fall.setText("Not Fall.")
            self.ui.label_fall.show()

    def show_ends(self, not_ends):
        if not not_ends:
            img = cv2.imread(self.ends_jpg_path)
            img = cv2.resize(img, (self.width, self.height))
            frame = QImage(img, self.width, self.height, QImage.Format_RGB888)
            pix = QPixmap.fromImage(frame)
            self.ui.label_rgb.setPixmap(pix)
            self.ui.label_fl.setPixmap(pix)
            self.optical_thread.join()

    def play_stop(self):
        self.player.player.stop()
        self.ui.frame_fall.setStyleSheet("QFrame { background-color: Green }")
        self.ui.label_fall.setText("Solved.")
        self.ui.label_fall.show()

    def show_select_mp4(self):
        if self.ui.radioButton.isChecked():
            dig = QFileDialog()
            dig.setFileMode(QFileDialog.AnyFile)
            dig.setFilter(QDir.Files)
            if dig.exec():
                file_paths = dig.selectedFiles()
                self.avi_path = file_paths[0]
                self.optical_thread.start()

    def update_predict_threshold(self):
        self.threshold = self.ui.horizontalSlider_threshold.value() / 10.0
        self.classifier.threshold = self.threshold
        self.ui.label_Threshold_2.setText(str(self.threshold))
        self.ui.label_Threshold_2.show()

    def update_music_volume(self):
        self.sound_level = self.ui.horizontalSlider_sound.value()
        self.player.player.setVolume(self.sound_level)

    def exam(self) -> None:
        # Thread(target=self.videographer.capture_video, args=(self.frame_queue,)).start()
        Thread(target=self.feature_extractor.extract, args=(self.flow_queue, self.feature_queue,)).start()
        Thread(target=self.classifier.classify_single, args=(self.feature_queue,)).start()
