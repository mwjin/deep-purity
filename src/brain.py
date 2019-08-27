#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
Make, train, and test our deep learning model
"""
import h5py
import random
import os
import sys
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import functools
import keras

from keras import backend as kb
from keras.models import Model, load_model
from keras.layers import Dense, Flatten, concatenate, BatchNormalization
from keras.layers import Conv2D, Input, MaxPool2D
from keras.applications import ResNet50
from keras.callbacks import ModelCheckpoint, EarlyStopping, History
from keras import regularizers
from data_generator import DataGenerator

# constants
BATCH_SIZE = 32
MAX_EPOCH = 500


def make_base_model(base_model_path):
    """
    make our model
    """
    # build a CNN model for VAF-LRR plots
    vaf_lrr_input = Input(shape=(401, 501, 1), name='vaf_lrr_image')
    vaf_lrr_cnn_model = _build_cnn_model(vaf_lrr_input)

    # build a fully connected layer for VAF histograms of somatic mutations
    """
    vaf_hist_input = Input(shape=(101,), name='vaf_hist_array')
    full_conn_layer = Dense(512, kernel_initializer='he_uniform', activation='relu',
                            kernel_regularizer=regularizers.l2(0.0))(vaf_hist_input)
    full_conn_out = Dense(512, kernel_initializer='he_uniform', activation='relu',
                          kernel_regularizer=regularizers.l2(0.0))(full_conn_layer)
    """
    # concatenate two models
    # concat_layer = concatenate([vaf_lrr_cnn_model.output])
    pred_out_layer = Dense(1, activation=None, name='output')(vaf_lrr_cnn_model.output)

    model = Model(inputs=[vaf_lrr_input], outputs=pred_out_layer)
    model.compile(loss='mean_squared_error', optimizer='adam')

    print(model.summary())
    model.save(base_model_path)


def train_model(train_model_path, base_model_path,
                train_data_list_path, valid_data_list_path, draw_learning_curve=True):
    """
    Train our model
    """
    print('[LOG] Load a base model', time.ctime())
    model = load_model(base_model_path)

    print('[LOG] Start training the model', time.ctime())
    with open(train_data_list_path, 'r') as train_data_list_file:
        train_data_paths = train_data_list_file.read().splitlines()

    with open(valid_data_list_path, 'r') as valid_data_list_file:
        valid_data_paths = valid_data_list_file.read().splitlines()

    # down-sample the images to reduce training time consuming
    train_data_paths = random.sample(train_data_paths, int(len(train_data_paths) * 0.6))
    valid_data_paths = random.sample(valid_data_paths, int(len(valid_data_paths) * 0.4))

    params = {
        'batch_size': BATCH_SIZE,
        'input_keys': ['vaf_hist_array', 'vaf_lrr_image'],
        'output_key': 'tumor_purity',
        'shuffle': True
    }

    train_data_generator = DataGenerator(train_data_paths, **params)
    valid_data_generator = DataGenerator(valid_data_paths, **params)

    # train the model
    model_ckpt = ModelCheckpoint(train_model_path, monitor='val_loss', save_best_only=True, mode='min')
    early_stop_cond = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
    history = History()
    model.fit_generator(train_data_generator, epochs=MAX_EPOCH, callbacks=[model_ckpt, early_stop_cond, history],
                        validation_data=valid_data_generator, verbose=1)

    if draw_learning_curve:
        print('[LOG] Draw learning curve', time.ctime())
        model_dir, model_filename = os.path.split(train_model_path)
        model_filename_wo_ext = os.path.splitext(model_filename)[0]

        plot_dir = f'{model_dir}/learning-curve'
        plot_path = f'{plot_dir}/{model_filename_wo_ext}.png'
        plot_title = f'Model loss ({model_filename_wo_ext})'
        os.makedirs(plot_dir, exist_ok=True)

        _draw_learning_curve(plot_path, plot_title, history)

    print('[LOG] Training is terminated.', time.ctime())


def test_model(test_result_path, train_model_path, test_data_list_path):
    """
    Test the trained model by predicting purities for each each data
    """
    print('[LOG] Load the trained model', time.ctime())
    model = load_model(train_model_path)

    print('[LOG] Start purity prediction', time.ctime())
    with open(test_data_list_path, 'r') as test_data_list_file:
        test_data_paths = test_data_list_file.read().splitlines()

    params = {
        'batch_size': 1,
        'input_keys': ['vaf_hist_array', 'vaf_lrr_image'],
        'output_key': 'tumor_purity',
        'shuffle': False
    }
    test_data_generator = DataGenerator(test_data_paths, **params)
    predicted_values = model.predict_generator(test_data_generator)
    actual_values = []

    for test_data_path in test_data_paths:
        with h5py.File(test_data_path, 'r') as test_data:
            actual_values.append(test_data['tumor_purity'].value)

    with open(test_result_path, 'w') as outfile:
        for test_data_path, actual_value, predicted_value in zip(test_data_paths, actual_values, predicted_values):
            print(*[test_data_path, actual_value, *predicted_value], sep='\t', file=outfile)

    print('[LOG] Testing is terminated.', time.ctime())


def see_model_weights(model_path):
    model = load_model(model_path)

    for layer in model.layers[1:]:
        weights, _ = layer.get_weights()
        print(weights)


def _build_cnn_model(input_tensor):
    """
    Inspired by LeNet-5
    # TODO: ResNet
    """
    """
    conv = \
        Conv2D(8, 2, kernel_initializer='he_uniform', activation='relu', padding="same", name='conv_1')(input_tensor)
    conv = MaxPool2D(2, 2, name='pool_1')(conv)
    conv = Conv2D(4, 2, kernel_initializer='he_uniform', activation='relu', padding="same", name='conv_2')(conv)
    conv = MaxPool2D(2, 2, name='pool_2')(conv)
    conv = Flatten(name='flatten')(conv)
    conv = Dense(128, kernel_initializer='he_uniform', activation='relu', name='fc_1')(conv)
    model = Model(inputs=[input_tensor], outputs=conv)
    """
    conv = \
        Conv2D(3, 1, kernel_initializer='he_uniform', activation='relu', padding='valid', name='conv_1')(input_tensor)
    res_model = ResNet50(weights='imagenet')
    res_model = res_model(conv)
    model = Model(inputs=[input_tensor], outputs=res_model)
    return model


def _make_keras_metric_func(method):
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        kb.get_session().run(tf.local_variables_initializer())

        with tf.control_dependencies([update_op]):
            value = tf.identity(value)

        return value

    return wrapper


def _draw_learning_curve(plot_path, plot_title, history):
    """
    Draw learning curve using history of keras callbacks
    Ref: https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras
    """
    train_losses = history.history['loss']
    val_losses = history.history['val_loss']

    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title(plot_title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.ylim(0, 0.05)
    plt.legend(['train', 'validation'], loc='upper left')
    # plt.yscale('log')
    plt.savefig(plot_path)
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.exit(f'[LOG] Please enter a function name and corresponding arguments')
    else:
        function_name = sys.argv[1]
        function_args = sys.argv[2:]

        if function_name in locals().keys():
            locals()[function_name](*function_args)
        else:
            sys.exit(f'[ERROR]: The function \"{function_name}\" is unavailable.')
