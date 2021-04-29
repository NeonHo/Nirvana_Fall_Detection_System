from PyQt5 import QtMultimedia
from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaPlaylist
from PyQt5.QtWidgets import QWidget


class MusicPlayer(QWidget):
    def __init__(self, sound_path):
        super(MusicPlayer, self).__init__()
        self.sound_path = sound_path
        url = QUrl.fromLocalFile(self.sound_path)
        content = QtMultimedia.QMediaContent(url)
        self.playlist = QMediaPlaylist(self)
        self.player = QtMultimedia.QMediaPlayer()
        self.player.setPlaylist(self.playlist)
        self.playlist.addMedia(content)
        self.playlist.setPlaybackMode(QMediaPlaylist.CurrentItemInLoop)
        self.playlist.setCurrentIndex(0)
