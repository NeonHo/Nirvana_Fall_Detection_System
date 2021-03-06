from FeatureExtractor import FeatureExtractor
from Classifier import Classifier


def main():
    # weight_path = "F:\\fsociety\\graduation_project\\Fall-Detection-with-CNNs-and-Optical-Flow-master\\otherFiles" \
    #               "\\weights.h5 "
    # mean_path = "F:\\fsociety\\graduation_project\\Fall-Detection-with-CNNs-and-Optical-Flow-master\\otherFiles" \
    #             "\\flow_mean.mat "
    # feature_extractor = FeatureExtractor(weight_path, mean_path)
    # optical_frame_path = "F:\\fsociety\\graduation_project\\Project\\TestExampleInTrainingSet\\fall"
    features_path = "F:\\fsociety\\graduation_project\\Project\\TestExampleInTrainingSet\\fall_features"
    # feature_extractor.extract(optical_frame_path, features_path)
    fold_best_model_path = "F:\\fsociety\\graduation_project\\Project\\Copy\\urfd_fold_1.h5"
    classifier = Classifier(fold_best_model_path)
    classifier.classify(features_path + "\\features.h5")


if __name__ == "__main__":
    main()
