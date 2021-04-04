from __future__ import print_function

import os

import h5py
import numpy as np
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
mean_file = 'flow_mean.mat'
vgg_16_weights = 'weights.h5'
save_plots = True

# Set to 'True' if you want to restore a previous trained models Training is skipped and test is done
use_checkpoint = False
# --------------------------

best_model_path = '/content/models/'
plots_folder = '/content/plots/'
checkpoint_path = best_model_path + 'fold_'

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

L = 10
num_features = 4096
batch_norm = True
learning_rate = 0.01
mini_batch_size = 32
weight_0 = 2
epochs = 1000
use_validation = False
# After the training stops, use train+validation to train for 1 epoch
use_val_for_training = False
val_size = 200
# Threshold to classify between positive and negative
threshold = 0.5

# Name of the experiment
exp = 'combine_lr{}_batchs{}_batchnorm{}_w0_{}'.format(learning_rate, mini_batch_size, batch_norm, weight_0)

# Load features and labels per dataset
h5features_multicam = h5py.File(features_file['multicam'], 'r')
h5labels_multicam = h5py.File(labels_file['multicam'], 'r')
h5features_urfd = h5py.File(features_file['urfd'], 'r')
h5labels_urfd = h5py.File(labels_file['urfd'], 'r')
h5features_fdd = h5py.File(features_file['fdd'], 'r')
h5labels_fdd = h5py.File(labels_file['fdd'], 'r')
kfold = KFold(n_splits=5, shuffle=True)
k_fold0_stages_3list = [None, None, None]
k_fold1_stages_3list = [None, None, None]
k_fold0_ur = None
k_fold1_ur = None
k_fold0_fdd = None
k_fold1_fdd = None
x_test_list = []
y_test_list = []
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
    if 'val_acc' in history and 'val_loss' in history:
        val = True
    plt.ioff()
    if 'accuracy' in metrics:
        fig = plt.figure()
        plt.plot(history['accuracy'])
        if val:
            plt.plot(history['val_acc'])
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


def train_multiple_cameras(classifier, class_weight, callbacks, stage_head, stage_tail, fold_index, limit_size=False):
    """
    cut out the multiple cameras dataset from stage_head to stage tail,
    and if we use limit_size, take out samples with specific size to train.
    :param callbacks:
    :param class_weight:
    :param classifier:
    :param fold_index:
    :param stage_head: int
    :param stage_tail: int
    :param limit_size: bool
    :return:
    """
    global k_fold0_stages_3list
    global k_fold1_stages_3list
    global y_test_list
    global x_test_list
    global size

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
        all0_stages = np.random.choice(all0_stages, size, replace=False)
        all1_stages = np.random.choice(all1_stages, size, replace=False)
    # Use a 5 fold cross-validation
    _train0_stages = None
    _test0_stages = None
    _train1_stages = None
    _test1_stages = None

    if stage_head == 1:
        if fold_index == 0:
            # At first time, we split the dataset.
            k_fold0_stages_3list[0] = kfold.split(all0_stages)
            k_fold1_stages_3list[0] = kfold.split(all1_stages)
        _train0_stages, _test0_stages = next(k_fold0_stages_3list[0])
        _train1_stages, _test1_stages = next(k_fold1_stages_3list[0])
    elif stage_head == 11:
        if fold_index == 0:
            k_fold0_stages_3list[1] = kfold.split(all0_stages)
            k_fold1_stages_3list[1] = kfold.split(all1_stages)
        _train0_stages, _test0_stages = next(k_fold0_stages_3list[1])
        _train1_stages, _test1_stages = next(k_fold1_stages_3list[1])
    else:
        if fold_index == 0:
            k_fold0_stages_3list[2] = kfold.split(all0_stages)
            k_fold1_stages_3list[2] = kfold.split(all1_stages)
        _train0_stages, _test0_stages = next(k_fold0_stages_3list[2])
        _train1_stages, _test1_stages = next(k_fold1_stages_3list[2])
    # CROSS-VALIDATION: Stratified partition of the dataset into train/test sets
    # Get the train and test indices, then get the actual indices
    train0_stages = all0_stages[_train0_stages]
    train1_stages = all1_stages[_train1_stages]
    test0_stages = all0_stages[_test0_stages]
    test1_stages = all1_stages[_test1_stages]

    x_val_stages = None
    y_val_stages = None
    if use_validation:
        (train0_stages, train1_stages, val0_stages, val1_stages) = divide_train_val(train0_stages, train1_stages,
                                                                                    val_size // 3)
        temp = np.concatenate((val0_stages, val1_stages))
        x_val_stages = x_stages[temp]
        y_val_stages = y_stages[temp]

    # Sampling
    x_train_stages, y_train_stages = sample_from_dataset(x_stages, y_stages, train0_stages, train1_stages)
    # Create the evaluation folds for each dataset
    x_test_stages, y_test_stages = sample_from_dataset(x_stages, y_stages, test0_stages, test1_stages)
    x_test_list.append(x_test_stages)
    y_test_list.append(y_test_stages)

    # Training
    validation_data = None
    if use_validation:
        validation_data = (x_val_stages, y_val_stages)
    _mini_batch_size = mini_batch_size
    if mini_batch_size == 0:
        _mini_batch_size = x_train_stages.shape[0]
    history = classifier.fit(
        x_train_stages,
        y_train_stages,
        validation_data=validation_data,
        batch_size=_mini_batch_size,
        epochs=epochs,
        shuffle='batch',
        class_weight=class_weight,
        callbacks=callbacks
    )

    plot_training_info(plots_folder + exp + '_multiple_cameras' + str(stage_head) + '-' + str(stage_tail),
                       ['accuracy', 'loss'], save_plots, history.history)

    # delete
    del _train0_stages
    del _train1_stages
    del train0_stages
    del train1_stages
    # del val0_stages
    # del val1_stages
    del all1_stages
    del all0_stages
    del _x_stages
    del _y_stages
    del x_stages
    del y_stages
    del x_train_stages
    del y_train_stages
    # del x_val_stages
    # del y_val_stages
    # del validation_data


def train_ur_fall(classifier, class_weight, callbacks, fold_index, limit_size=False):
    global k_fold0_ur
    global k_fold1_ur
    global x_test_list
    global y_test_list
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
    # Use a 5 fold cross-validation
    if fold_index == 0:
        # At first time, we split the dataset.
        k_fold0_ur = kfold.split(all0_ur)
        k_fold1_ur = kfold.split(all1_ur)
    # CROSS-VALIDATION: Stratified partition of the dataset into train/test sets
    _train0_ur, _test0_ur = next(k_fold0_ur)
    _train1_ur, _test1_ur = next(k_fold1_ur)
    train0_ur = all0_ur[_train0_ur]
    train1_ur = all1_ur[_train1_ur]
    test0_ur = all0_ur[_test0_ur]
    test1_ur = all1_ur[_test1_ur]

    x_val_ur = None
    y_val_ur = None
    if use_validation:
        (train0_ur, train1_ur, val0_ur, val1_ur) = divide_train_val(train0_ur, train1_ur, val_size // 3)
        temp = np.concatenate((val0_ur, val1_ur))
        x_val_ur = x_ur[temp]
        y_val_ur = y_ur[temp]

    # Sampling
    x_train_ur, y_train_ur = sample_from_dataset(x_ur, y_ur, train0_ur, train1_ur)
    # Create the evaluation folds for each dataset
    x_test_ur, y_test_ur = sample_from_dataset(x_ur, y_ur, test0_ur, test1_ur)

    x_test_list.append(x_test_ur)
    y_test_list.append(y_test_ur)

    # Training
    validation_data = None
    if use_validation:
        validation_data = (x_val_ur, y_val_ur)

    _mini_batch_size = mini_batch_size
    if mini_batch_size == 0:
        _mini_batch_size = x_train_ur.shape[0]

    history = classifier.fit(
        x_train_ur,
        y_train_ur,
        validation_data=validation_data,
        batch_size=_mini_batch_size,
        epochs=epochs,
        shuffle='batch',
        class_weight=class_weight,
        callbacks=callbacks
    )

    plot_training_info(plots_folder + exp + '_ur_fall', ['accuracy', 'loss'], save_plots, history.history)

    # delete
    del _train0_ur
    del _train1_ur
    del train0_ur
    del train1_ur
    # del val0_ur
    # del val1_ur
    del all1_ur
    del all0_ur
    del x_ur
    del y_ur
    del x_train_ur
    del y_train_ur
    # del x_val_ur
    # del y_val_ur
    # del validation_data


def train_fall_detection(classifier, class_weight, callbacks, fold_index, limit_size=False):
    global k_fold0_fdd
    global k_fold1_fdd
    global x_test_list
    global y_test_list
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
    # Use a 5 fold cross-validation
    if fold_index == 0:
        # At first time, we split the dataset.
        k_fold0_fdd = kfold.split(all0_fdd)
        k_fold1_fdd = kfold.split(all1_fdd)
    # CROSS-VALIDATION: Stratified partition of the dataset into train/test sets
    _train0_fdd, _test0_fdd = next(k_fold0_fdd)
    _train1_fdd, _test1_fdd = next(k_fold1_fdd)
    train0_fdd = all0_fdd[_train0_fdd]
    train1_fdd = all1_fdd[_train1_fdd]
    test0_fdd = all0_fdd[_test0_fdd]
    test1_fdd = all1_fdd[_test1_fdd]

    x_val_fdd = None
    y_val_fdd = None
    if use_validation:
        (train0_fdd, train1_fdd, val0_fdd, val1_fdd) = divide_train_val(train0_fdd, train1_fdd, val_size // 3)
        temp = np.concatenate((val0_fdd, val1_fdd))
        x_val_fdd = x_fdd[temp]
        y_val_fdd = y_fdd[temp]

    # Sampling
    x_train_fdd, y_train_fdd = sample_from_dataset(x_fdd, y_fdd, train0_fdd, train1_fdd)
    # Create the evaluation folds for each dataset
    x_test_fdd, y_test_fdd = sample_from_dataset(x_fdd, y_fdd, test0_fdd, test1_fdd)
    x_test_list.append(x_test_fdd)
    y_test_list.append(y_test_fdd)

    # Training
    validation_data = None
    if use_validation:
        validation_data = (x_val_fdd, y_val_fdd)

    _mini_batch_size = mini_batch_size
    if mini_batch_size == 0:
        _mini_batch_size = x_train_fdd.shape[0]

    history = classifier.fit(
        x_train_fdd,
        y_train_fdd,
        validation_data=validation_data,
        batch_size=_mini_batch_size,
        epochs=epochs,
        shuffle='batch',
        class_weight=class_weight,
        callbacks=callbacks
    )

    plot_training_info(plots_folder + exp + '_fall_detection', ['accuracy', 'loss'], save_plots, history.history)

    # delete
    del _train0_fdd
    del _train1_fdd
    del train0_fdd
    del train1_fdd
    # del val0_fdd
    # del val1_fdd
    del all1_fdd
    del all0_fdd
    del x_fdd
    del y_fdd
    del x_train_fdd
    del y_train_fdd
    # del x_val_fdd
    # del y_val_fdd
    # del validation_data


def main():
    global threshold
    global x_test_list, y_test_list
    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    threshold = 0.5

    # Arrays to save the results
    sensitivities = []
    specificities = []

    for fold in range(5):
        # CLASSIFIER
        extracted_features = Input(shape=(num_features,), dtype='float32', name='input')
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(extracted_features)
        x = Activation('relu')(x)
        x = Dropout(0.9)(x)
        x = Dense(4096, name='fc2', kernel_initializer='glorot_uniform')(x)
        x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
        x = Activation('relu')(x)
        x = Dropout(0.8)(x)
        x = Dense(1, name='predictions', kernel_initializer='glorot_uniform')(x)
        x = Activation('sigmoid')(x)

        classifier = Model(inputs=extracted_features, outputs=x, name='classifier')
        fold_best_model_path = best_model_path + 'combined_fold_{}.h5'.format(fold)
        classifier.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])

        # weighting of each class: only the fall class gets a different weight
        class_weight = {0: weight_0, 1: 1}
        callbacks = None
        if use_validation:
            # callback definition
            metric = 'val_accuracy'
            e = EarlyStopping(monitor=metric, min_delta=0, patience=100, mode='auto')
            c = ModelCheckpoint(fold_best_model_path, monitor=metric, save_best_only=True, save_weights_only=True,
                                mode='auto')
            callbacks = [e, c]

        # train multiple cameras dataset from 1st stage to 10th stage.
        train_ur_fall(classifier, class_weight, callbacks, fold, limit_size=True)
        train_multiple_cameras(classifier, class_weight, callbacks, 1, 10, fold, limit_size=True)
        train_multiple_cameras(classifier, class_weight, callbacks, 11, 20, fold, limit_size=True)
        train_multiple_cameras(classifier, class_weight, callbacks, 21, 25, fold, limit_size=True)
        train_fall_detection(classifier, class_weight, callbacks, fold, limit_size=True)

        if not use_validation:
            classifier.save_weights(fold_best_model_path)
        # EVALUATION
        classifier.load_weights(fold_best_model_path)
        x_test = np.concatenate((x_test_list[0], x_test_list[1], x_test_list[2], x_test_list[3], x_test_list[4]),
                                axis=0)
        y_test = np.concatenate((y_test_list[0], y_test_list[1], y_test_list[2], y_test_list[3], y_test_list[4]),
                                axis=0)
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

        x_test_list = []
        y_test_list = []

        sensitivities.append(tp / float(tp + fn))
        specificities.append(tn / float(tn + fp))

    # End of the Cross-Validation
    print('CROSS-VALIDATION RESULTS ===================')
    print("Sensitivity Combined: {:.2f}% (+/- {:.2f}%)".format(np.mean(sensitivities) * 100.,
                                                               np.std(sensitivities) * 100.))
    print("Specificity Combined: {:.2f}% (+/- {:.2f}%)".format(np.mean(specificities) * 100.,
                                                               np.std(specificities) * 100.))


if __name__ == '__main__':
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    if not os.path.exists(plots_folder):
        os.makedirs(plots_folder)

    main()
