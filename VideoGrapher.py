import cv2
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox

from RgbFlowSignal import RgbFlowSignal


class Videographer:
    def __init__(self, video_path, width, height):
        self.capture = None
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_path = video_path
        self.width = width
        self.height = height
        self.out = None
        self.previous_frame = None
        self.captured_frame_number = 0
        self.signal = RgbFlowSignal()
        self.__running = True

    def terminal(self):
        self.__running = False

    def capture_video_frame(self):
        """
        capture 1 RGB frame, show it and store into the video.
        :return:
        """
        ret, current_frame = self.capture.read()
        if current_frame is None:
            QMessageBox.critical(QtWidgets.QWidget(), "摄像头问题", "您在USB接口1上可能没有接摄像头！并未捕获到录像帧！", QMessageBox.Yes)
            return False
        self.captured_frame_number += 1
        current_frame = cv2.resize(current_frame, (self.width, self.height))
        self.signal.per_rgb.emit([current_frame])
        self.out.write(current_frame)
        cv2.waitKey(1)
        return ret

    def capture_video(self):
        self.capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        self.out = cv2.VideoWriter(self.video_path + "_video.avi", self.fourcc, 20.0, (self.width, self.height))
        ret = True
        while self.__running and ret:
            ret = self.capture_video_frame()
