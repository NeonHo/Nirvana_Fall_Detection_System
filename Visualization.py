import h5py


class VisualizerDiy:
    def visualize_data(self, file_path):
        h5_file = h5py.File(file_path, "r")  # open h5 file
        print(h5_file.keys())
        for key in h5_file.keys():
            print(h5_file[key].shape)
            for index_row in range(0, h5_file[key].shape[0]):
                for index_column in range(0, h5_file[key].shape[1]):
                    print(h5_file[key].value[index_row][index_column], end = "\t")
                print()


def main():
    visualizer = VisualizerDiy()
    visualizer.visualize_data("F:\\fsociety\\graduation_project\\Fall-Detection-with-CNNs-and-Optical-Flow-master"
                              "\\otherFiles\\URFD_results\\features_urfd_tf.h5")


if __name__ == "__main__":
    # execute only if run as a script
    main()
