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
from sklearn.model_selection import KFold

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
L = 10
num_features = 4096
batch_norm = True
learning_rate = 0.001
mini_batch_size = 512
weight_0 = 2
epochs = 2000
use_validation = True
use_early_stop = False
cam_times = 6  # Multi-cam dataset is how many times of the ur fall dataset.
# hidden layer units' number
hidden_layer_units_num = 3072
hidden_lambda = 0.01
output_lambda = 0.01
# validation
val_size = 528
val_size_stages = val_size // 8
val_size_ur = (val_size // 8) * 6
val_size_fdd = val_size // 8
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


def reload_multiple_cameras_dataset(stage_head, stage_tail, limit_size=False):
    """
    cut out the multiple cameras dataset from stage_head to stage tail,
    and if we use limit_size, take out samples with specific size to train.
    :param stage_head: int
    :param stage_tail: int
    :param limit_size: bool
    :return:
    """
    global size
    global cam_times

    # stages from 1 to 10.
    stages = []
    for i in range(stage_head, stage_tail):
        stages.append('chute{:02}'.format(i))

    _x_stages = []
    _y_stages = []
    for nb_stage, stage in enumerate(stages):
        for nb_cam, cam in enumerate(h5features_multicam[stage].keys()):
            for key in h5features_multicam[stage][cam].keys():
                _x_stages.extend([x for x in h5features_multicam[stage][cam][key]])
                _y_stages.extend([x for x in h5labels_multicam[stage][cam][key]])
    # multiple cameras dataset from head to tail.
    x_stages = np.asarray(_x_stages)
    y_stages = np.asarray(_y_stages)
    # Step 1
    all0_stages = np.asarray(np.where(y_stages == 0)[0])
    all1_stages = np.asarray(np.where(y_stages == 1)[0])
    # Step 2 under-sample
    if limit_size:
        temp_size = (size // 24) * (stage_tail - stage_head) * cam_times
        all0_stages = np.random.choice(all0_stages, temp_size, replace=False)
        all1_stages = np.random.choice(all1_stages, temp_size, replace=False)
        x_stages, y_stages = sample_from_dataset(x_stages, y_stages, all0_stages, all1_stages)

    # delete
    del all1_stages
    del all0_stages
    del _x_stages
    del _y_stages
    return x_stages, y_stages


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
    x_stages_1_10, y_stages_1_10 = reload_multiple_cameras_dataset(1, 11, True)
    x_stages_11_20, y_stages_11_20 = reload_multiple_cameras_dataset(11, 21, True)
    x_stages_21_24, y_stages_21_24 = reload_multiple_cameras_dataset(21, 25, True)
    x_fdd, y_fdd = reload_fall_detection_dataset(True)
    # join all multiple cameras datasets.
    x_stages = np.concatenate((x_stages_1_10, x_stages_11_20, x_stages_21_24), axis=0)
    y_stages = np.concatenate((y_stages_1_10, y_stages_11_20, y_stages_21_24), axis=0)

    # all 0 and all 1 indices from different datasets.
    all0_stages = np.asarray(np.where(y_stages == 0)[0])
    all1_stages = np.asarray(np.where(y_stages == 1)[0])
    all0_ur = np.asarray(np.where(y_ur == 0)[0])
    all1_ur = np.asarray(np.where(y_ur == 1)[0])
    all0_fdd = np.asarray(np.where(y_fdd == 0)[0])
    all1_fdd = np.asarray(np.where(y_fdd == 1)[0])

    # arrays to save the results
    sensitivities = {'combined': [], 'multicam': [], 'urfd': [], 'fdd': []}
    specificities = {'combined': [], 'multicam': [], 'urfd': [], 'fdd': []}

    # Use a 5 fold cross-validation
    k_fold = KFold(n_splits=5, shuffle=True)
    k_fold0_stages = k_fold.split(all0_stages)
    k_fold1_stages = k_fold.split(all1_stages)
    k_fold0_ur = k_fold.split(all0_ur)
    k_fold1_ur = k_fold.split(all1_ur)
    k_fold0_fdd = k_fold.split(all0_fdd)
    k_fold1_fdd = k_fold.split(all1_fdd)

    # CROSS-VALIDATION: Stratified partition of the dataset into train/test sets.
    for fold in range(5):
        # stages
        _train0_stages, _test0_stages = next(k_fold0_stages)
        _train1_stages, _test1_stages = next(k_fold1_stages)
        train0_stages = all0_stages[_train0_stages]
        train1_stages = all1_stages[_train1_stages]
        test0_stages = all0_stages[_test0_stages]
        test1_stages = all1_stages[_test1_stages]
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
        if use_validation:
            # stages
            (train0_stages, train1_stages, val0_stages, val1_stages) = divide_train_val(train0_stages, train1_stages,
                                                                                        val_size_stages)
            val_index = np.concatenate((val0_stages, val1_stages))
            x_val_stages = x_stages[val_index]
            y_val_stages = y_stages[val_index]
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
        x_train_stages, y_train_stages = sample_from_dataset(x_stages, y_stages, train0_stages, train1_stages)
        x_train_ur, y_train_ur = sample_from_dataset(x_ur, y_ur, train0_ur, train1_ur)
        x_train_fdd, y_train_fdd = sample_from_dataset(x_fdd, y_fdd, train0_fdd, train1_fdd)
        # create the evaluation folds for each dataset
        x_test_stages, y_test_stages = sample_from_dataset(x_stages, y_stages, test0_stages, test1_stages)
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
        fold_best_model_path = best_model_path + 'combined_fold_{}.h5'.format(fold)
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

            plot_training_info(plots_folder + exp + '_fold' + str(fold), ['accuracy', 'loss'], save_plots,
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
