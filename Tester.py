import threading
import time
from Classifier import Classifier
from FeatureExtractor import FeatureExtractor
from OpticalGenerator import OpticalGenerator
from VideoGrapher import Videographer


class Tester(threading.Thread):
    def __init__(self):
        super().__init__()
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
        self.optical_generator = OpticalGenerator(self.video_path, self.flow_image_path, self.bound, self.width,
                                                  self.height)
        self.feature_extractor = FeatureExtractor(self.weight_path, self.mean_path, self.flow_image_path,
                                                  self.features_path)
        self.classifier = Classifier(self.model_path, self.features_path)

    def start_test(self):
        # self.videographer.start()
        # while not self.videographer.v_start_og[0]:
        #     pass
        # time.sleep(10)  # delay 5 second to avoid empty.
        # self.optical_generator.start()
        # while not self.optical_generator.or_start_fe[0]:
        #     pass
        # time.sleep(5)
        self.start()

    def end_test(self):
        pass

    def run(self) -> None:
        while True:
            self.feature_extractor.extract(self.classifier)
