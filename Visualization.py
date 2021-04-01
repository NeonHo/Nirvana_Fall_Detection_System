import h5py
import numpy as np

# F:\fsociety\graduation_project\Project\Copy\feature_label\labels_urfd_tf.h5
class VisualizerDiy:
    def visualize_data(self, file_path):
        h5_file = h5py.File(file_path, "r")  # open h5 file
        print(h5_file.keys())
        for key in h5_file.keys():
            print("Shape:" + str(h5_file[key].shape))


def main():
    visualizer = VisualizerDiy()
    h5_file_path = input("Where's your hdf5 file?")
    visualizer.visualize_data(str(h5_file_path))


if __name__ == "__main__":
    # execute only if run as a script
    main()
