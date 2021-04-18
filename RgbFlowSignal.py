from PyQt5.QtCore import pyqtSignal, QObject


class RgbFlowSignal(QObject):
    frames = pyqtSignal(list)
    music = pyqtSignal(bool)
