from __future__ import print_function

import os

import h5py
import matplotlib
import numpy as np
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (Input, Activation, Dense, Dropout)
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from numpy.random import seed
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import StratifiedShuffleSplit, KFold

seed(1)
matplotlib.use('Agg')
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# CHANGE THESE VARIABLES ---
vgg_16_weights = 'weights.h5'
save_features = False
save_plots = True

# Set to 'True' if you want to restore a previous trained models
# Training is skipped and test is done
use_checkpoint = False
# --------------------------
# 'models\\''models/'
# 'plots\\''plots/'
best_model_path = 'models\\'
plots_folder = 'plots\\'
checkpoint_path = best_model_path + 'fold_'
# 'F:\\fsociety\\graduation_project\\Project\\train\\train_mark2\\saved_features\\'
# '/content/drive/MyDrive/train/saved_features/'
saved_files_folder = 'F:\\fsociety\\graduation_project\\Project\\train\\train_mark2\\saved_features\\'
features_file = saved_files_folder + 'features_multicam.h5'
labels_file = saved_files_folder + 'labels_multicam.h5'
features_key = 'features'
labels_key = 'labels'

num_cameras = 8
fold_num = 6
L = 10
num_features = 4096
batch_norm = True
learning_rate = 0.001
mini_batch_size = 0
weight_0 = 1
epochs = 2000  # 6000
use_validation = True
use_early_stop = False
hidden_layer_units_num = 4096
hidden_lambda = 0.01
output_lambda = 0.01
val_size = 100
# Threshold to classify between positive and negative
threshold = 0.5
# dropout prob
dropout_l1 = 0.1
dropout_l2 = 0.2

# Name of the experiment
exp = 'multicam_lr{}_batchs{}_batchnorm{}_w0_{}'.format(learning_rate, mini_batch_size, batch_norm, weight_0)


def plot_training_info(case, metrics, save, history):
    """
    Function to create plots for train and validation loss and accuracy
    Input:
    * case: name for the plot, an 'accuracy.png' or 'loss.png' will be concatenated after the name.
    * metrics: list of metrics to store: 'loss' and/or 'accuracy'
    * save: boolean to store the plots or only show them.
    * history: History object returned by the Keras fit function.
    """
    val = False
    if 'val_accuracy' in history and 'val_loss' in history:
        val = True
    plt.ioff()
    if 'accuracy' in metrics:
        fig = plt.figure()
        plt.plot(history['accuracy'])
        if val:
            plt.plot(history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        if val:
            plt.legend(['train', 'val'], loc='upper left')
        else:
            plt.legend(['train'], loc='upper left')
        if save:
            plt.savefig(case + 'accuracy.png')
            plt.gcf().clear()
        else:
            plt.show()
        plt.close(fig)

    # summarize history for loss
    if 'loss' in metrics:
        fig = plt.figure()
        plt.plot(history['loss'])
        if val:
            plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        # plt.ylim(1e-3, 1e-2)
        plt.yscale("log")
        if val:
            plt.legend(['train', 'val'], loc='upper left')
        else:
            plt.legend(['train'], loc='upper left')
        if save:
            plt.savefig(case + 'loss.png')
            plt.gcf().clear()
        else:
            plt.show()
        plt.close(fig)


def load_dataset():
    h5features = h5py.File(features_file, 'r')
    h5labels = h5py.File(labels_file, 'r')

    # Load the data separated by cameras for cross-validation
    stages = []
    for i in range(1, 5):
        stages.append('chute{:02}'.format(i))
    cams_x = []
    cams_y = []
    for stage, nb_stage in zip(stages, range(len(stages))):
        for cam, nb_cam in zip(h5features[stage].keys(), range(8)):
            temp_x = []
            temp_y = []
            for key in h5features[stage][cam].keys():
                temp_x.append(np.asarray(h5features[stage][cam][key]))
                temp_y.append(np.asarray(h5labels[stage][cam][key]))
            temp_x = np.concatenate(temp_x, axis=0)
            temp_y = np.concatenate(temp_y, axis=0)
            temp_0_indexes = np.where(temp_y == 0)[0]
            temp_1_indexes = np.where(temp_y == 1)[0]
            # Balance the positive and negative samples
            temp_1_indexes = np.random.choice(temp_1_indexes, len(temp_0_indexes), replace=False)
            temp_indexes = np.concatenate((temp_0_indexes, temp_1_indexes))
            temp_x = np.asarray(temp_x[temp_indexes, ...])
            temp_y = np.asarray(temp_y[temp_indexes])
            if nb_stage == 0:
                cams_x.append(temp_x)
                cams_y.append(temp_y)
            else:
                cams_x[nb_cam] = np.concatenate([cams_x[nb_cam], temp_x], axis=0)
                cams_y[nb_cam] = np.concatenate([cams_y[nb_cam], temp_y], axis=0)
            del temp_x
            del temp_y

    return cams_x, cams_y


def divide_k_fold(cams_x, cams_y, fold_num):
    # every cam gives (val_size/num_cameras)/2 zero-sample and (val_size/num_cameras)/2 one-sample.

    cams_x_blocks_list = []  # row = cams_num, column = k, element is fold
    cams_y_blocks_list = []  # row = cams_num, column = k, element is fold
    for cam in range(num_cameras):
        cam_x = cams_x[cam]
        cam_y = cams_y[cam]
        zeroes = np.asarray(np.where(cam_y == 0)[0])
        ones = np.asarray(np.where(cam_y == 1)[0])
        k_fold = KFold(n_splits=fold_num, shuffle=True)
        k_fold0_indexes = k_fold.split(zeroes)
        k_fold1_indexes = k_fold.split(ones)
        cam_x_blocks_list = []  # row = k, element is fold
        cam_y_blocks_list = []  # row = k, element is fold
        # divide zeros and ones into 5 blocks.
        for fold in range(fold_num):
            _, test0_indexes = next(k_fold0_indexes)
            _, test1_indexes = next(k_fold1_indexes)
            indexes_0 = zeroes[test0_indexes]
            indexes_1 = ones[test1_indexes]
            indexes = np.concatenate([indexes_0, indexes_1], axis=0)
            fold_x = cam_x[indexes]
            fold_y = cam_y[indexes]
            cam_x_blocks_list.append(fold_x)
            cam_y_blocks_list.append(fold_y)
        cams_x_blocks_list.append(cam_x_blocks_list)
        cams_y_blocks_list.append(cam_y_blocks_list)

    return cams_x_blocks_list, cams_y_blocks_list


def main():
    # ========================================================================
    # TRAINING
    # =======================================================================
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0005)

    cams_x, cams_y = load_dataset()

    sensitivities = []
    specificities = []
    auc_s = []
    accuracies = []
    cams_x_blocks_list, cams_y_blocks_list = divide_k_fold(cams_x, cams_y, fold_num=fold_num)
    # row = cams_num, column = k, element is fold transform into row = k, column = cams_num
    folds_x = []
    folds_y = []
    for fold_index in range(fold_num):
        # a list which every element comes from all cameras.
        cams_x_folds = [cam_x_folds[fold_index] for cam_x_folds in cams_x_blocks_list]
        cams_y_folds = [cam_y_folds[fold_index] for cam_y_folds in cams_y_blocks_list]
        cams_x_folds = np.concatenate(cams_x_folds)
        cams_y_folds = np.concatenate(cams_y_folds)
        folds_x.append(cams_x_folds)
        folds_y.append(cams_y_folds)
        del cams_x_folds
        del cams_y_folds
    del cams_x
    del cams_y
    del cams_x_blocks_list
    del cams_y_blocks_list

    # 8-fold cross validation
    for fold_index in range(fold_num):
        print('=' * 30)
        print(str(fold_num) + '-fold cross validation : {} fold'.format(fold_index))
        print('=' * 30)

        test_x = folds_x[fold_index]
        test_y = folds_y[fold_index]
        train_x = folds_x[0: fold_index] + folds_x[fold_index + 1:]
        train_y = folds_y[0: fold_index] + folds_y[fold_index + 1:]

        # Flatten to 1D arrays
        train_x = np.asarray([train_x[i][j] for i in range(len(train_x)) for j in range(len(train_x[i]))])
        train_y = np.asarray([train_y[i][j] for i in range(len(train_y)) for j in range(len(train_y[i]))])

        # Create a validation subset from the training set
        zeroes = np.asarray(np.where(train_y == 0)[0])
        ones = np.asarray(np.where(train_y == 1)[0])
        train_val_split_0 = StratifiedShuffleSplit(n_splits=1, test_size=int(val_size / 2), random_state=7)
        indices_0 = train_val_split_0.split(train_x[zeroes, ...], np.argmax(train_y[zeroes, ...], 1))
        train_val_split_1 = StratifiedShuffleSplit(n_splits=1, test_size=int(val_size / 2), random_state=7)
        indices_1 = train_val_split_1.split(train_x[ones, ...], np.argmax(train_y[ones, ...], 1))
        train_indices_0, val_indices_0 = next(indices_0)
        train_indices_1, val_indices_1 = next(indices_1)

        x_train = np.concatenate([train_x[zeroes, ...][train_indices_0, ...], train_x[ones, ...][train_indices_1, ...]],
                                 axis=0)
        y_train = np.concatenate([train_y[zeroes, ...][train_indices_0, ...], train_y[ones, ...][train_indices_1, ...]],
                                 axis=0)
        x_val = np.concatenate([train_x[zeroes, ...][val_indices_0, ...], train_x[ones, ...][val_indices_1, ...]],
                               axis=0)
        y_val = np.concatenate([train_y[zeroes, ...][val_indices_0, ...], train_y[ones, ...][val_indices_1, ...]],
                               axis=0)

        del train_x
        del train_y

        # ==================== CLASSIFIER ========================
        # input layer
        extracted_features = Input(shape=(num_features,), dtype='float32', name='input')
        if batch_norm:
            x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(extracted_features)
            x = Activation('relu')(x)
        else:
            x = ELU(alpha=1.0)(extracted_features)
        # hidden layer
        x = Dropout(dropout_l1)(x)
        x = Dense(hidden_layer_units_num, name='fc2', kernel_initializer='glorot_uniform',
                  kernel_regularizer=regularizers.l2(hidden_lambda))(x)
        if batch_norm:
            x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
            x = Activation('relu')(x)
        else:
            x = ELU(alpha=1.0)(x)
        # output layer
        x = Dropout(dropout_l2)(x)
        x = Dense(1, name='predictions', kernel_initializer='glorot_uniform',
                  kernel_regularizer=regularizers.l2(output_lambda))(x)
        x = Activation('sigmoid')(x)

        classifier = Model(inputs=extracted_features, outputs=x, name='classifier')
        fold_best_model_path = best_model_path + 'multicam_fold_{}.h5'.format(fold_index)
        classifier.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

        if not use_checkpoint:
            # ==================== TRAINING ========================
            # weighting of each class: only the fall class gets a different weight
            class_weight = {0: weight_0, 1: 1}

            callbacks = None
            if use_validation:
                # callback definition
                metric = 'val_loss'
                if use_early_stop:
                    e = EarlyStopping(monitor=metric, min_delta=0, patience=100, mode='auto')
                    c = ModelCheckpoint(fold_best_model_path, monitor=metric, save_best_only=True,
                                        save_weights_only=True, mode='auto')
                    callbacks = [e, c]
                else:
                    c = ModelCheckpoint(fold_best_model_path, monitor=metric, save_best_only=True,
                                        save_weights_only=True, mode='auto')
                    callbacks = [c]
            validation_data = None
            if use_validation:
                validation_data = (x_val, y_val)

            _mini_batch_size = mini_batch_size
            if mini_batch_size == 0:
                _mini_batch_size = x_train.shape[0]

            history = classifier.fit(
                x_train,
                y_train,
                validation_data=validation_data,
                batch_size=_mini_batch_size,
                epochs=epochs,
                shuffle=True,
                class_weight=class_weight,
                callbacks=callbacks
            )
            if not use_validation:
                classifier.save_weights(fold_best_model_path)

            plot_training_info(plots_folder + exp + '_fold' + str(fold_index), ['accuracy', 'loss'], save_plots,
                               history.history)

        # ==================== EVALUATION ========================
        # Load best model
        print('Model loaded from checkpoint')
        classifier.load_weights(fold_best_model_path)

        predicted = classifier.predict(test_x)
        for i in range(len(predicted)):
            if predicted[i] < threshold:
                predicted[i] = 0
            else:
                predicted[i] = 1
        # Array of predictions 0/1
        predicted = np.asarray(predicted).astype(int)

        # Compute metrics and print them
        cm = confusion_matrix(test_y, predicted, labels=[0, 1])
        tp = cm[0][0]
        fn = cm[0][1]
        fp = cm[1][0]
        tn = cm[1][1]
        tpr = tp / float(tp + fn)
        fpr = fp / float(fp + tn)
        fnr = fn / float(fn + tp)
        tnr = tn / float(tn + fp)
        precision = tp / float(tp + fp)
        recall = tp / float(tp + fn)
        specificity = tn / float(tn + fp)
        f1 = 2 * float(precision * recall) / float(precision + recall)
        accuracy = accuracy_score(test_y, predicted)
        fpr, tpr, _ = roc_curve(test_y, predicted)
        roc_auc = auc(fpr, tpr)

        print('FOLD {} results:'.format(fold_index))
        print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp, tn, fp, fn))
        print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(tpr, tnr, fpr, fnr))
        print('Sensitivity/Recall: {}'.format(recall))
        print('Specificity: {}'.format(specificity))
        print('Precision: {}'.format(precision))
        print('F1-measure: {}'.format(f1))
        print('Accuracy: {}'.format(accuracy))
        print('AUC: {}'.format(roc_auc))

        # Store the metrics for this epoch
        sensitivities.append(tp / float(tp + fn))
        specificities.append(tn / float(tn + fp))
        auc_s.append(roc_auc)
        accuracies.append(accuracy)

    print('LEAVE-ONE-OUT RESULTS ===================')
    print("Sensitivity: %.2f%% (+/- %.2f%%)" % (np.mean(sensitivities) * 100., np.std(sensitivities) * 100.))
    print("Specificity: %.2f%% (+/- %.2f%%)" % (np.mean(specificities) * 100., np.std(specificities) * 100.))
    print("Accuracy: %.2f%% (+/- %.2f%%)" % (np.mean(accuracies) * 100., np.std(accuracies) * 100.))
    print("AUC: %.2f%% (+/- %.2f%%)" % (np.mean(auc_s) * 100., np.std(auc_s) * 100.))


if __name__ == '__main__':
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    main()
