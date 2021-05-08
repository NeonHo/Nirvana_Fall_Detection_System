from __future__ import print_function

import os

import h5py
import numpy as np
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import (Input, Activation, Dense, Dropout)
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from numpy.random import seed
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold, StratifiedShuffleSplit

seed(1)
# os.environ["TF_XLA_FLAGS"] = "XLA_FLAGS"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# CHANGE THESE VARIABLES ---
save_plots = True

# Set to 'True' if you want to restore a previous trained models Training is skipped and test is done
use_checkpoint = False
# --------------------------

best_model_path = '/content/models/'
plots_folder = '/content/plots/'
checkpoint_path = best_model_path + 'fold_'
# 'F:\\fsociety\\graduation_project\\Project\\train\\train_mark2\\saved_features\\'
saved_files_folder = '/content/drive/MyDrive/train/saved_features/'
features_file = {
    'urfd': saved_files_folder + 'features_urfd_tf.h5',
    'multicam': saved_files_folder + 'features_multicam.h5',
    'fdd': saved_files_folder + 'features_fdd.h5',
}
labels_file = {
    'urfd': saved_files_folder + 'labels_urfd_tf.h5',
    'multicam': saved_files_folder + 'labels_multicam.h5',
    'fdd': saved_files_folder + 'labels_fdd.h5',
}
features_key = 'features'
labels_key = 'labels'

# Hyper parameters
folds_num = 5
L = 10
num_features = 4096
batch_norm = True
learning_rate = 0.001
mini_batch_size = 512
weight_0 = 2
epochs = 2000
use_validation = True
use_early_stop = False
# hidden layer units' number
hidden_layer_units_num = 3072
hidden_lambda = 0.01
output_lambda = 0.01
# validation
val_size = 66
val_size_stages = 710
val_size_ur = 66
val_size_fdd = 66
# Threshold to classify between positive and negative
threshold = 0.5
# dropout pro
dropout_l1 = 0.1
dropout_l2 = 0.2

# Name of the experiment
exp = 'combine_lr{}_batchs{}_batchnorm{}_w0_{}'.format(learning_rate, mini_batch_size, batch_norm, weight_0)

# Load features and labels per dataset
h5features_multicam = h5py.File(features_file['multicam'], 'r')
h5labels_multicam = h5py.File(labels_file['multicam'], 'r')
h5features_urfd = h5py.File(features_file['urfd'], 'r')
h5labels_urfd = h5py.File(labels_file['urfd'], 'r')
h5features_fdd = h5py.File(features_file['fdd'], 'r')
h5labels_fdd = h5py.File(labels_file['fdd'], 'r')
size = 0


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


def sample_from_dataset(x, y, zeroes, ones):
    """
    Samples from X and y using the indices obtained from the arrays all0 and all1 taking slices that depend on the
    fold, the slice_size the mode.
    Input:
    * X: array of features
    * y: array of labels
    * all0: indices of sampled labelled as class 0 in y
    * all1: indices of sampled labelled as class 1 in y
    * fold: integer, fold number (from the cross-validation)
    * slice_size: integer, half of the size of a fold
    * mode: 'train' or 'test', used to choose how to slice
    """
    indices = np.concatenate([zeroes, ones], axis=0)
    sampled_x = x[indices]
    sampled_y = y[indices]
    return sampled_x, sampled_y


def divide_train_val(zeroes, ones, validation_size):
    """
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size/2, random_state=7)
    indices_0 = sss.split(np.zeros(len(zeroes)), zeroes)
    indices_1 = sss.split(np.zeros(len(ones)), ones)
    train_indices_0, val_indices_0 = indices_0.next()
    train_indices_1, val_indices_1 = indices_1.next()
    """
    rand0 = np.random.permutation(len(zeroes))
    train_indices_0 = zeroes[rand0[validation_size // 2:]]
    val_indices_0 = zeroes[rand0[:validation_size // 2]]
    rand1 = np.random.permutation(len(ones))
    train_indices_1 = ones[rand1[validation_size // 2:]]
    val_indices_1 = ones[rand1[:validation_size // 2]]
    return train_indices_0, train_indices_1, val_indices_0, val_indices_1


def reload_multiple_cameras_dataset():
    # Load the data separated by cameras for cross-validation
    stages = []
    for i in range(1, 25):
        stages.append('chute{:02}'.format(i))
    cams_x = []
    cams_y = []
    for stage, nb_stage in zip(stages, range(len(stages))):
        for cam, nb_cam in zip(h5features_multicam[stage].keys(), range(8)):
            temp_x = []
            temp_y = []
            for key in h5features_multicam[stage][cam].keys():
                temp_x.append(np.asarray(h5features_multicam[stage][cam][key]))
                temp_y.append(np.asarray(h5labels_multicam[stage][cam][key]))
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


def divide_k_fold(cams_x, cams_y, folds_num):
    # every cam gives (val_size/num_cameras)/2 zero-sample and (val_size/num_cameras)/2 one-sample.

    cams_x_blocks_list = []  # row = cams_num, column = k, element is fold
    cams_y_blocks_list = []  # row = cams_num, column = k, element is fold
    for cam in range(8):
        cam_x = cams_x[cam]
        cam_y = cams_y[cam]
        zeroes = np.asarray(np.where(cam_y == 0)[0])
        ones = np.asarray(np.where(cam_y == 1)[0])
        k_fold = KFold(n_splits=folds_num, shuffle=True)
        k_fold0_indexes = k_fold.split(zeroes)
        k_fold1_indexes = k_fold.split(ones)
        cam_x_blocks_list = []  # row = k, element is fold
        cam_y_blocks_list = []  # row = k, element is fold
        # divide zeros and ones into 5 blocks.
        for fold in range(folds_num):
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

    # row = cams_num, column = k, element is fold transform into row = k, column = cams_num
    stages_x = []
    stages_y = []
    for fold_index in range(folds_num):
        # a list which every element comes from all cameras.
        cams_x_folds = [cam_x_folds[fold_index] for cam_x_folds in cams_x_blocks_list]
        cams_y_folds = [cam_y_folds[fold_index] for cam_y_folds in cams_y_blocks_list]
        cams_x_folds = np.concatenate(cams_x_folds)
        cams_y_folds = np.concatenate(cams_y_folds)
        stages_x.append(cams_x_folds)
        stages_y.append(cams_y_folds)
        del cams_x_folds
        del cams_y_folds
    del cams_x
    del cams_y
    del cams_x_blocks_list
    del cams_y_blocks_list

    return stages_x, stages_y


def reload_ur_fall_dataset(limit_size=False):
    global size

    x_ur = np.asarray(h5features_urfd['features'])
    y_ur = np.asarray(h5labels_urfd['labels'])
    size = np.asarray(np.where(y_ur == 0)[0]).shape[0]
    # step1
    all0_ur = np.asarray(np.where(y_ur == 0)[0])
    all1_ur = np.asarray(np.where(y_ur == 1)[0])
    # step2 under-sample
    if limit_size:
        all0_ur = np.random.choice(all0_ur, size, replace=False)
        all1_ur = np.random.choice(all1_ur, size, replace=False)
        x_ur, y_ur = sample_from_dataset(x_ur, y_ur, all0_ur, all1_ur)

    # delete
    del all1_ur
    del all0_ur
    return x_ur, y_ur


def reload_fall_detection_dataset(limit_size=False):
    global size

    x_fdd = np.asarray(h5features_fdd['features'])
    y_fdd = np.asarray(h5labels_fdd['labels'])

    # step1
    all0_fdd = np.asarray(np.where(y_fdd == 0)[0])
    all1_fdd = np.asarray(np.where(y_fdd == 1)[0])
    # step2 under-sample
    if limit_size:
        all0_fdd = np.random.choice(all0_fdd, size, replace=False)
        all1_fdd = np.random.choice(all1_fdd, size, replace=False)
        x_fdd, y_fdd = sample_from_dataset(x_fdd, y_fdd, all0_fdd, all1_fdd)

    # delete
    del all1_fdd
    del all0_fdd
    return x_fdd, y_fdd


def main():
    global threshold
    global size
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    threshold = 0.5

    # load dataset as the size of UR
    x_ur, y_ur = reload_ur_fall_dataset(True)
    x_stages, y_stages = reload_multiple_cameras_dataset()
    x_fdd, y_fdd = reload_fall_detection_dataset(True)

    # all 0 and all 1 indices from different datasets.
    all0_ur = np.asarray(np.where(y_ur == 0)[0])
    all1_ur = np.asarray(np.where(y_ur == 1)[0])
    all0_fdd = np.asarray(np.where(y_fdd == 0)[0])
    all1_fdd = np.asarray(np.where(y_fdd == 1)[0])

    # arrays to save the results
    sensitivities = {'combined': [], 'multicam': [], 'urfd': [], 'fdd': []}
    specificities = {'combined': [], 'multicam': [], 'urfd': [], 'fdd': []}

    # Use a 5 fold cross-validation
    stages_x, stages_y = divide_k_fold(x_stages, y_stages, folds_num=folds_num)
    k_fold = KFold(n_splits=5, shuffle=True)
    k_fold0_ur = k_fold.split(all0_ur)
    k_fold1_ur = k_fold.split(all1_ur)
    k_fold0_fdd = k_fold.split(all0_fdd)
    k_fold1_fdd = k_fold.split(all1_fdd)

    # CROSS-VALIDATION: Stratified partition of the dataset into train/test sets.
    for fold_index in range(folds_num):
        # stages
        test_x_stages = stages_x[fold_index]
        test_y_stages = stages_y[fold_index]
        train_x_stages = stages_x[0: fold_index] + stages_x[fold_index + 1:]
        train_y_stages = stages_y[0: fold_index] + stages_y[fold_index + 1:]
        # Flatten to 1D arrays
        train_x_stages = np.asarray(
            [train_x_stages[i][j] for i in range(len(train_x_stages)) for j in range(len(train_x_stages[i]))])
        train_y_stages = np.asarray(
            [train_y_stages[i][j] for i in range(len(train_y_stages)) for j in range(len(train_y_stages[i]))])
        # ur
        _train0_ur, _test0_ur = next(k_fold0_ur)
        _train1_ur, _test1_ur = next(k_fold1_ur)
        train0_ur = all0_ur[_train0_ur]
        train1_ur = all1_ur[_train1_ur]
        test0_ur = all0_ur[_test0_ur]
        test1_ur = all1_ur[_test1_ur]
        # fdd
        _train0_fdd, _test0_fdd = next(k_fold0_fdd)
        _train1_fdd, _test1_fdd = next(k_fold1_fdd)
        train0_fdd = all0_fdd[_train0_fdd]
        train1_fdd = all1_fdd[_train1_fdd]
        test0_fdd = all0_fdd[_test0_fdd]
        test1_fdd = all1_fdd[_test1_fdd]

        x_val = None
        y_val = None
        x_train_stages = None
        y_train_stages = None
        if use_validation:
            # stages
            # Create a validation subset from the training set
            zeroes = np.asarray(np.where(train_y_stages == 0)[0])
            ones = np.asarray(np.where(train_y_stages == 1)[0])
            train_val_split_0 = StratifiedShuffleSplit(n_splits=1, test_size=int(val_size_stages / 2), random_state=7)
            indices_0 = train_val_split_0.split(train_x_stages[zeroes, ...], np.argmax(train_y_stages[zeroes, ...], 1))
            train_val_split_1 = StratifiedShuffleSplit(n_splits=1, test_size=int(val_size_stages / 2), random_state=7)
            indices_1 = train_val_split_1.split(train_x_stages[ones, ...], np.argmax(train_y_stages[ones, ...], 1))
            train_indices_0, val_indices_0 = next(indices_0)
            train_indices_1, val_indices_1 = next(indices_1)
            x_train_stages = np.concatenate(
                [train_x_stages[zeroes, ...][train_indices_0, ...], train_x_stages[ones, ...][train_indices_1, ...]],
                axis=0)
            y_train_stages = np.concatenate(
                [train_y_stages[zeroes, ...][train_indices_0, ...], train_y_stages[ones, ...][train_indices_1, ...]],
                axis=0)
            x_val_stages = np.concatenate(
                [train_x_stages[zeroes, ...][val_indices_0, ...], train_x_stages[ones, ...][val_indices_1, ...]],
                axis=0)
            y_val_stages = np.concatenate(
                [train_y_stages[zeroes, ...][val_indices_0, ...], train_y_stages[ones, ...][val_indices_1, ...]],
                axis=0)
            del train_x_stages
            del train_y_stages
            # ur
            (train0_ur, train1_ur, val0_ur, val1_ur) = divide_train_val(train0_ur, train1_ur, val_size_ur)
            val_index = np.concatenate((val0_ur, val1_ur))
            x_val_ur = x_ur[val_index]
            y_val_ur = y_ur[val_index]
            # fdd
            (train0_fdd, train1_fdd, val0_fdd, val1_fdd) = divide_train_val(train0_fdd, train1_fdd, val_size_fdd)
            val_index = np.concatenate((val0_fdd, val1_fdd))
            x_val_fdd = x_fdd[val_index]
            y_val_fdd = y_fdd[val_index]

            # join
            x_val = np.concatenate((x_val_stages, x_val_ur, x_val_fdd), axis=0)
            y_val = np.concatenate((y_val_stages, y_val_ur, y_val_fdd), axis=0)

        # sampling
        x_train_ur, y_train_ur = sample_from_dataset(x_ur, y_ur, train0_ur, train1_ur)
        x_train_fdd, y_train_fdd = sample_from_dataset(x_fdd, y_fdd, train0_fdd, train1_fdd)
        # create the evaluation folds for each dataset
        x_test_stages, y_test_stages = test_x_stages, test_y_stages
        x_test_ur, y_test_ur = sample_from_dataset(x_ur, y_ur, test0_ur, test1_ur)
        x_test_fdd, y_test_fdd = sample_from_dataset(x_fdd, y_fdd, test0_fdd, test1_fdd)

        # join train
        x_train = np.concatenate((x_train_stages, x_train_ur, x_train_fdd), axis=0)
        y_train = np.concatenate((y_train_stages, y_train_ur, y_train_fdd), axis=0)
        # join test
        x_test = np.concatenate((x_test_stages, x_test_ur, x_test_fdd), axis=0)
        y_test = np.concatenate((y_test_stages, y_test_ur, y_test_fdd), axis=0)

        # classifier
        # input layer
        extracted_features = Input(shape=(num_features,), dtype='float32', name='input')
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(extracted_features)
        x = Activation('relu')(x)
        # hidden layer
        x = Dropout(dropout_l1)(x)
        x = Dense(hidden_layer_units_num, name='fc2', kernel_initializer='glorot_uniform',
                  kernel_regularizer=regularizers.l2(hidden_lambda))(x)
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
        x = Activation('relu')(x)
        # output layer
        x = Dropout(dropout_l2)(x)
        x = Dense(1, name='predictions', kernel_initializer='glorot_uniform',
                  kernel_regularizer=regularizers.l2(output_lambda))(x)
        x = Activation('sigmoid')(x)

        classifier = Model(inputs=extracted_features, outputs=x, name='classifier')
        fold_best_model_path = best_model_path + 'combined_fold_{}.h5'.format(fold_index)
        classifier.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

        if not use_checkpoint:
            # training
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

        # evaluation
        print('Model loaded from checkpoint.')
        classifier.load_weights(fold_best_model_path)

        # evaluate for the combined test set.
        predicted = classifier.predict(x_test)
        for i in range(len(predicted)):
            if predicted[i] < threshold:
                predicted[i] = 0
            else:
                predicted[i] = 1
        predicted = np.asarray(predicted).astype(int)
        cm = confusion_matrix(y_test, predicted, labels=[0, 1])
        tp = cm[0][0]
        fn = cm[0][1]
        fp = cm[1][0]
        tn = cm[1][1]
        tpr = tp / float(tp + fn)
        fpr = fp / float(fp + tn)
        fnr = fn / float(fn + tp)
        print('Combined test set')
        print('-' * 10)
        tnr = tn / float(tn + fp)
        print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp, tn, fp, fn))
        print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(tpr, tnr, fpr, fnr))
        print('Sensitivity/Recall: {}'.format(tp / float(tp + fn)))
        print('Specificity: {}'.format(tn / float(tn + fp)))
        print('Accuracy: {}'.format(accuracy_score(y_test, predicted)))
        sensitivities['combined'].append(tp / float(tp + fn))
        specificities['combined'].append(tn / float(tn + fp))

        # evaluate for the ur test set.
        predicted = classifier.predict(x_test_ur)
        for i in range(len(predicted)):
            if predicted[i] < threshold:
                predicted[i] = 0
            else:
                predicted[i] = 1
        predicted = np.asarray(predicted).astype(int)
        cm = confusion_matrix(y_test_ur, predicted, labels=[0, 1])
        tp = cm[0][0]
        fn = cm[0][1]
        fp = cm[1][0]
        tn = cm[1][1]
        tpr = tp / float(tp + fn)
        fpr = fp / float(fp + tn)
        fnr = fn / float(fn + tp)
        tnr = tn / float(tn + fp)
        print('URFD test set')
        print('-' * 10)
        print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp, tn, fp, fn))
        print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(tpr, tnr, fpr, fnr))
        print('Sensitivity/Recall: {}'.format(tp / float(tp + fn)))
        print('Specificity: {}'.format(tn / float(tn + fp)))
        print('Accuracy: {}'.format(accuracy_score(y_test_ur, predicted)))
        sensitivities['urfd'].append(tp / float(tp + fn))
        specificities['urfd'].append(tn / float(tn + fp))

        # evaluate for the stages test set.
        predicted = classifier.predict(x_test_stages)
        for i in range(len(predicted)):
            if predicted[i] < threshold:
                predicted[i] = 0
            else:
                predicted[i] = 1
        predicted = np.asarray(predicted).astype(int)
        cm = confusion_matrix(y_test_stages, predicted, labels=[0, 1])
        tp = cm[0][0]
        fn = cm[0][1]
        fp = cm[1][0]
        tn = cm[1][1]
        tpr = tp / float(tp + fn)
        fpr = fp / float(fp + tn)
        fnr = fn / float(fn + tp)
        tnr = tn / float(tn + fp)
        print('Multicam test set')
        print('-' * 10)
        print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp, tn, fp, fn))
        print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(tpr, tnr, fpr, fnr))
        print('Sensitivity/Recall: {}'.format(tp / float(tp + fn)))
        print('Specificity: {}'.format(tn / float(tn + fp)))
        print('Accuracy: {}'.format(accuracy_score(y_test_stages, predicted)))
        sensitivities['multicam'].append(tp / float(tp + fn))
        specificities['multicam'].append(tn / float(tn + fp))

        # evaluate for the fdd test set.
        predicted = classifier.predict(x_test_fdd)
        for i in range(len(predicted)):
            if predicted[i] < threshold:
                predicted[i] = 0
            else:
                predicted[i] = 1
        predicted = np.asarray(predicted).astype(int)
        cm = confusion_matrix(y_test_fdd, predicted, labels=[0, 1])
        tp = cm[0][0]
        fn = cm[0][1]
        fp = cm[1][0]
        tn = cm[1][1]
        tpr = tp / float(tp + fn)
        fpr = fp / float(fp + tn)
        fnr = fn / float(fn + tp)
        tnr = tn / float(tn + fp)
        print('FDD test set')
        print('-' * 10)
        print('TP: {}, TN: {}, FP: {}, FN: {}'.format(tp, tn, fp, fn))
        print('TPR: {}, TNR: {}, FPR: {}, FNR: {}'.format(tpr, tnr, fpr, fnr))
        print('Sensitivity/Recall: {}'.format(tp / float(tp + fn)))
        print('Specificity: {}'.format(tn / float(tn + fp)))
        print('Accuracy: {}'.format(accuracy_score(y_test_fdd, predicted)))
        sensitivities['fdd'].append(tp / float(tp + fn))
        specificities['fdd'].append(tn / float(tn + fp))

    # End of the Cross-Validation
    print('CROSS-VALIDATION RESULTS ===================')

    print("Sensitivity Combined: {:.2f}% (+/- {:.2f}%)".format(np.mean(sensitivities['combined']) * 100.,
                                                               np.std(sensitivities['combined']) * 100.))
    print("Specificity Combined: {:.2f}% (+/- {:.2f}%)\n".format(np.mean(specificities['combined']) * 100.,
                                                                 np.std(specificities['combined']) * 100.))
    print("Sensitivity URFD: {:.2f}% (+/- {:.2f}%)".format(np.mean(sensitivities['urfd']) * 100.,
                                                           np.std(sensitivities['urfd']) * 100.))
    print("Specificity URFD: {:.2f}% (+/- {:.2f}%)\n".format(np.mean(specificities['urfd']) * 100.,
                                                             np.std(specificities['urfd']) * 100.))
    print("Sensitivity Multicam: {:.2f}% (+/- {:.2f}%)".format(np.mean(sensitivities['multicam']) * 100.,
                                                               np.std(sensitivities['multicam']) * 100.))
    print("Specificity Multicam: {:.2f}% (+/- {:.2f}%)\n".format(np.mean(specificities['multicam']) * 100.,
                                                                 np.std(specificities['multicam']) * 100.))
    print("Sensitivity FDDs: {:.2f}% (+/- {:.2f}%)".format(np.mean(sensitivities['fdd']) * 100.,
                                                           np.std(sensitivities['fdd']) * 100.))
    print("Specificity FDDs: {:.2f}% (+/- {:.2f}%)".format(np.mean(specificities['fdd']) * 100.,
                                                           np.std(specificities['fdd']) * 100.))


if __name__ == '__main__':
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    main()
