from PyQt5.QtCore import pyqtSignal, QObject


class RgbFlowSignal(QObject):
    not_ends = pyqtSignal(bool)
    frames = pyqtSignal(list)
    music = pyqtSignal(bool)
