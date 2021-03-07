from FeatureExtractor import FeatureExtractor
from Classifier import Classifier


def main():
    weight_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\weights.h5 "
    mean_path = "F:\\fsociety\\graduation_project\\Nirvana_Fall_Detection_System\\otherFiles\\flow_mean.mat "
    feature_extractor = FeatureExtractor(weight_path, mean_path)
    optical_frame_path = "F:\\fsociety\\graduation_project\\Project\\TestClassifier\\Test1Package\\notfall"
    features_path = "F:\\fsociety\\graduation_project\\Project\\TestExampleInTrainingSet\\notfall_features"
    feature_extractor.extract(optical_frame_path, features_path)
    fold_best_model_path = "F:\\fsociety\\graduation_project\\Project\\Copy\\urfd_fold_1.h5"
    classifier = Classifier(fold_best_model_path)
    classifier.classify(features_path + "\\features.h5")


if __name__ == "__main__":
    main()
