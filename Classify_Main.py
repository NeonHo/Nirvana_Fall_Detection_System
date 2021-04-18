from PyQt5.QtWidgets import QApplication
from MonitorWindow import MonitorWindow


def main():
    app = QApplication([])
    monitor_window = MonitorWindow()
    monitor_window.ui.show()
    app.exec()


if __name__ == "__main__":
    main()
