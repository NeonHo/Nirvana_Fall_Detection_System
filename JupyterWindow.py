from queue import Queue
from threading import Thread

from Classifier import Classifier
from FeatureExtractor import FeatureExtractor
from OpticalGenerator import OpticalGenerator


class JupyterWindow:
    def __init__(self):
        super(JupyterWindow, self).__init__()
        # parameters
        self.is_windows = False
        self.width = 224
        self.height = 224
        self.bound = 20
        self.sound_level = 50
        self.threshold = 0.5

        # paths
        separator = "\\" if self.is_windows else "/"
        work_path = "/content/drive/MyDrive/app"
        self.video_path = work_path + separator + "otherFiles" + separator + "videos" + separator
        self.flow_image_path = work_path + separator + "otherFiles" + separator + "flow" + separator
        self.features_path = work_path + separator + "otherFiles" + separator + "features" + separator
        self.weight_path = work_path + separator + "otherFiles" + separator
        self.mean_path = work_path + separator + "otherFiles" + separator
        self.model_path = work_path + separator + "otherFiles" + separator + "URFD_results" + separator
        self.avi_path = work_path + separator + "test_ground" + separator + "cam7_2.avi"

        self.rgb = None
        self.flow = None

        # component
        self.feature_extractor = FeatureExtractor(self.weight_path, self.mean_path, self.flow_image_path,
                                                  self.features_path, self.width, self.height)
        self.optical_generator = OpticalGenerator(self.video_path, self.flow_image_path, self.bound, self.width,
                                                  self.height, self.feature_extractor.stack_length, use_qt=False,
                                                  is_windows=self.is_windows)
        self.classifier = Classifier(self.model_path, self.features_path, use_qt=False, threshold=self.threshold)

        # queues
        self.flow_queue = Queue(1)
        self.feature_queue = Queue(1)

    def exam(self) -> None:
        Thread(target=self.optical_generator.generate_optical_flow_tvl1, args=(self.avi_path, self.flow_queue)).start()
        Thread(target=self.feature_extractor.extract, args=(self.flow_queue, self.feature_queue,)).start()
        Thread(target=self.classifier.classify_single, args=(self.feature_queue,)).start()
