from PyQt5.QtCore import pyqtSignal, QObject


class RgbFlowSignal(QObject):
    not_ends = pyqtSignal(bool)
    frames = pyqtSignal(list)
    music = pyqtSignal(bool)
    judge_message = pyqtSignal(bool)
    per_stack = pyqtSignal(int)
    per_flow = pyqtSignal(int)
    per_rgb = pyqtSignal(list)
