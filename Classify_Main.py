import cv2

from VideoGrapher import Videographer
from OpticalGenerator import OpticalGenerator
from FeatureExtractor import FeatureExtractor
from Classifier import Classifier


def main():
    # Directories path
    video_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\videos\\"
    flow_image_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\flow\\"
    features_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\features\\"
    weight_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\"
    mean_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\"
    fold_best_model_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\URFD_results\\"
    # Videographer
    videographer = Videographer()
    videographer.get_video_constantly(video_path, 224, 224)
    # Optical Generator
    optical_generator = OpticalGenerator()
    optical_generator.generate(video_path, flow_image_path, 20, 224, 224)
    # Feature extractor
    feature_extractor = FeatureExtractor(weight_path, mean_path)
    feature_extractor.extract(flow_image_path, features_path)
    # Classifier
    classifier = Classifier(fold_best_model_path)
    classifier.classify(features_path)


if __name__ == "__main__":
    main()
