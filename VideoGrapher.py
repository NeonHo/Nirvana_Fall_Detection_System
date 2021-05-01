import threading
import time

import cv2
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QMessageBox

from RgbFlowSignal import RgbFlowSignal


class Videographer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.captured_frame_number = 0
        self.signal = RgbFlowSignal()
        self.__running = True
        self.lock = threading.Lock()

    def terminal(self):
        self.lock.acquire()
        self.__running = False
        self.lock.release()

    def begin(self):
        self.lock.acquire()
        self.__running = True
        self.lock.release()

    def capture_video_frame(self, capture, out):
        """
        capture 1 RGB frame, show it and store into the video.
        :return:
        """
        ret, current_frame = capture.read()
        if current_frame is None:
            QMessageBox.critical(QtWidgets.QWidget(), "摄像头问题", "您在USB接口1上可能没有接摄像头！并未捕获到录像帧！", QMessageBox.Yes)
            return False
        self.captured_frame_number += 1
        current_frame = cv2.resize(current_frame, (self.width, self.height))
        self.signal.per_rgb.emit([current_frame])
        out.write(current_frame)
        cv2.waitKey(1)
        return ret

    def capture_video(self, video_path):
        capture = None
        out = None
        self.lock.acquire()
        if self.__running:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)
            out = cv2.VideoWriter(video_path + "video.avi", fourcc, 20.0, (self.width, self.height))
        ret = True
        self.lock.release()
        while self.__running:
            if ret:
                ret = self.capture_video_frame(capture, out)
        if out is not None:
            out.release()
        if capture is not None:
            capture.release()
