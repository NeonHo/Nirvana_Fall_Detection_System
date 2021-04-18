from queue import Queue
from threading import Thread

from PyQt5 import uic
from PyQt5.QtGui import QImage, QPixmap

from Classifier import Classifier
from FeatureExtractor import FeatureExtractor
from OpticalGenerator import OpticalGenerator
from VideoGrapher import Videographer


class MonitorWindow:
    def __init__(self):
        super(MonitorWindow, self).__init__()
        # paths
        self.video_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\videos\\"
        self.flow_image_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\flow\\"
        self.features_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\features\\"
        self.weight_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\"
        self.mean_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\"
        self.model_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\URFD_results\\"
        self.ui0_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\windows\\untitled.ui"
        self.ui = uic.loadUi(self.ui0_path)
        self.rgb = None
        self.flow = None
        self.width = None
        self.height = None

        # parameters
        self.width = 224
        self.height = 224
        self.bound = 20

        # component
        self.videographer = Videographer(self.video_path, self.width, self.height)
        self.feature_extractor = FeatureExtractor(self.weight_path, self.mean_path, self.flow_image_path,
                                                  self.features_path, self.width, self.height)
        self.optical_generator = OpticalGenerator(self.video_path, self.flow_image_path, self.bound, self.width,
                                                  self.height, self.feature_extractor.stack_length)
        self.classifier = Classifier(self.model_path, self.features_path)

        self.flow_queue = Queue(1)
        self.feature_queue = Queue(1)
        self.avi_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\test_ground\\cam7.avi"

        # signals
        self.optical_generator.rgb_flow_signal.frames.connect(self.show_frame)
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

    def exam(self) -> None:
        # Thread(target=self.videographer.capture_video, args=(self.frame_queue,)).start()
        Thread(target=self.optical_generator.generate_optical_flow_tvl1, args=(self.avi_path, self.flow_queue)).start()
        Thread(target=self.feature_extractor.extract, args=(self.flow_queue, self.feature_queue,)).start()
        Thread(target=self.classifier.classify_single, args=(self.feature_queue,)).start()
