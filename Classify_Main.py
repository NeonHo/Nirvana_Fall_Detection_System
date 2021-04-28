def main():
    use_qt = False
    if use_qt:
        from PyQt5.QtWidgets import QApplication
        from MonitorWindow import MonitorWindow
        app = QApplication([])
        monitor_window = MonitorWindow()
        monitor_window.ui.show()
        app.exec()
    else:
        from JupyterWindow import JupyterWindow
        jupyter_window = JupyterWindow()
        jupyter_window.exam()


if __name__ == "__main__":
    main()
