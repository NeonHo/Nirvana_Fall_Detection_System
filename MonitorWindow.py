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
        self.video_num = 0
        self.avi_directory_path = work_path + separator + "test_ground"
        self.video_path = self.avi_directory_path + separator
        self.flow_image_path = work_path + separator + "otherFiles" + separator + "flow" + separator
        self.features_path = work_path + separator + "otherFiles" + separator + "features" + separator
        self.weight_path = work_path + separator + "otherFiles" + separator
        self.mean_path = work_path + separator + "otherFiles" + separator
        self.model_path = work_path + separator + "otherFiles" + separator + "URFD_results" + separator
        self.ui0_path = work_path + separator + "windows" + separator + "MainWindow.ui"
        self.ui1_path = work_path + separator + "windows" + separator + "Camera.ui"
        self.jpg_path = work_path + separator + "otherFiles" + separator + "alarm.jpg"
        self.sound_path = work_path + separator + "otherFiles" + separator + "alarm.mp3"
        self.avi_path = None
        self.ends_jpg_path = work_path + separator + "otherFiles" + separator + "ends.jpg"
        self.ui = uic.loadUi(self.ui0_path)
        self.ui_cam = uic.loadUi(self.ui1_path)
        self.ui_cam.setVisible(False)
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
        self.ui.radioButton.toggled.connect(self.show_select_avi)
        self.ui.radioButton_2.toggled.connect(self.show_camera)
        self.ui_cam.pushButton.clicked.connect(self.finish_video)
        self.videographer.signal.per_rgb.connect(self.show_rgb)

        # threads
        self.optical_thread = None
        self.video_thread = None
        self.feature_thread = Thread(target=self.feature_extractor.extract,
                                     args=(self.flow_queue, self.feature_queue,))
        self.classify_thread = Thread(target=self.classifier.classify_single,
                                      args=(self.feature_queue,))
        self.feature_thread.start()
        self.classify_thread.start()

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
            self.feature_extractor.img_count = 0
            self.threads_ends()
            if self.ui.radioButton.isChecked():
                self.ui.radioButton.setAutoExclusive(False)
                self.ui.radioButton.setChecked(False)
                self.ui.radioButton.setAutoExclusive(True)

    def play_stop(self):
        self.player.player.stop()
        self.ui.frame_fall.setStyleSheet("QFrame { background-color: Green }")
        self.ui.label_fall.setText("Solved.")
        self.ui.label_fall.show()

    def show_select_avi(self):
        if self.ui.radioButton.isChecked():
            dig = QFileDialog(directory=self.avi_directory_path)
            dig.setFileMode(QFileDialog.AnyFile)
            dig.setFilter(QDir.Files)
            if dig.exec():
                file_paths = dig.selectedFiles()
                self.avi_path = file_paths[0]
                self.threads_init()
                self.threads_start()

    def show_camera(self):
        self.video_num = self.video_num + 1
        self.ui_cam.setVisible(True)
        self.videographer.video_path = self.video_path + str(self.video_num)
        self.video_thread = Thread(target=self.videographer.capture_video)
        self.video_thread.start()

    def show_rgb(self, frames):
        rgb_frame = frames[0]
        frame_left = QImage(rgb_frame, self.width, self.height, QImage.Format_RGB888)
        pix_left = QPixmap.fromImage(frame_left)
        self.ui_cam.label.setPixmap(pix_left)
        self.ui_cam.label.show()

    def finish_video(self):
        self.videographer.terminal()
        self.video_thread.join()
        self.videographer.capture.release()
        self.videographer.out.release()
        if self.ui.radioButton_2.isChecked():
            self.ui.radioButton_2.setAutoExclusive(False)
            self.ui.radioButton_2.setChecked(False)
            self.ui.radioButton_2.setAutoExclusive(True)
        self.ui_cam.setVisible(False)

    def update_predict_threshold(self):
        self.threshold = self.ui.horizontalSlider_threshold.value() / 10.0
        self.classifier.threshold = self.threshold
        self.ui.label_Threshold_2.setText(str(self.threshold))
        self.ui.label_Threshold_2.show()

    def update_music_volume(self):
        self.sound_level = self.ui.horizontalSlider_sound.value()
        self.player.player.setVolume(self.sound_level)

    def threads_init(self) -> None:
        # Thread(target=self.videographer.capture_video, args=(self.frame_queue,)).start()
        # threads
        self.optical_thread = Thread(target=self.optical_generator.generate_optical_flow_tvl1,
                                     args=(self.avi_path, self.flow_queue))

    def threads_start(self):
        if (self.optical_thread is not None) and \
                (self.feature_thread is not None) and \
                (self.classify_thread is not None):
            self.optical_thread.start()

    def threads_ends(self):
        if self.optical_thread is not None:
            self.optical_thread.join()
            self.optical_thread = None