from multiprocessing import Process
from threading import Thread
import time
from Classifier import Classifier
from FeatureExtractor import FeatureExtractor
from OpticalGenerator import OpticalGenerator
from VideoGrapher import Videographer
from queue import Queue


class Tester:
    def __init__(self):
        # parameters
        self.width = 224
        self.height = 224
        self.bound = 20
        # paths
        self.video_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\videos\\"
        self.flow_image_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\flow\\"
        self.features_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\features\\"
        self.weight_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\"
        self.mean_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\"
        self.model_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\URFD_results\\"
        # component
        self.videographer = Videographer(self.video_path, self.width, self.height)
        self.feature_extractor = FeatureExtractor(self.weight_path, self.mean_path, self.flow_image_path,
                                                  self.features_path, self.width, self.height)
        self.optical_generator = OpticalGenerator(self.video_path, self.flow_image_path, self.bound, self.width,
                                                  self.height)
        self.classifier = Classifier(self.model_path, self.features_path)

    def exam(self) -> None:
        frame_queue = Queue()
        flow_queue = Queue()
        feature_queue = Queue()
        Thread(target=self.videographer.capture_video, args=(frame_queue,)).start()
        Thread(target=self.optical_generator.generate_flow_couple, args=(frame_queue, flow_queue,)).start()
        Thread(target=self.feature_extractor.extract, args=(flow_queue, feature_queue,)).start()
        Thread(target=self.classifier.classify_single, args=(feature_queue,)).start()
