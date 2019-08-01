#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
Make, train, and test our deep learning model
"""
import pickle
import random
import os
import sys
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import functools

from keras import backend as kb
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, Input
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
    # build a fully connected layer and an output layer
    vaf_hist_layer = Input(shape=(101,), name='vaf_hist_array')
    # concat_cnn_output = concatenate([cnn_model_outputs[0], vaf_hist_layer])
    full_conn_layer = Dense(512, kernel_initializer='he_uniform', activation='relu',
                            kernel_regularizer=regularizers.l2(0.0))(vaf_hist_layer)
    full_conn_layer = Dense(512, kernel_initializer='he_uniform', activation='relu',
                            kernel_regularizer=regularizers.l2(0.0))(full_conn_layer)
    pred_out_layer = Dense(1, activation=None, name='output')(full_conn_layer)

    model = Model(inputs=[vaf_hist_layer], outputs=pred_out_layer)
    model.compile(loss='mean_squared_error', optimizer='adam')

    print(model.summary())
    model.save(base_model_path)


def train_model(train_model_path, base_model_path, train_img_set_path, valid_img_set_path, draw_learning_curve=True):
    """
    Train our model
    """
    print('[LOG] Load a base model', time.ctime())
    model = load_model(base_model_path)

    print('[LOG] Start training the model', time.ctime())
    with open(train_img_set_path, 'r') as train_img_set_file:
        train_image_paths = train_img_set_file.read().splitlines()

    with open(valid_img_set_path, 'r') as valid_img_set_file:
        valid_image_paths = valid_img_set_file.read().splitlines()

    # down-sample the images to reduce training time consuming
    train_image_paths = random.sample(train_image_paths, int(len(train_image_paths) * 0.6))
    valid_image_paths = random.sample(valid_image_paths, int(len(valid_image_paths) * 0.4))

    params = {
        'batch_size': BATCH_SIZE,
        'num_labels': 1,
        'num_channels': 9,
        'shuffle': True
    }

    train_data_generator = DataGenerator(train_image_paths, **params)
    valid_data_generator = DataGenerator(valid_image_paths, **params)

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


def test_model(predict_result_path, train_model_path, test_img_set_path):
    """
    Test our trained model by predicting numerical values corresponding to each image
    """
    print('[LOG] Load the trained model', time.ctime())
    model = load_model(train_model_path)

    print('[LOG] Start prediction', time.ctime())
    with open(test_img_set_path, 'r') as test_img_set_file:
        test_image_paths = test_img_set_file.read().splitlines()

    batch_size = 1
    params = {
        'batch_size': batch_size,
        'num_labels': 1,
        'num_channels': 9,
        'shuffle': False
    }
    test_data_generator = DataGenerator(test_image_paths, **params)
    predict_values = model.predict_generator(test_data_generator)
    real_values = [pickle.load(open(image_path, 'rb'))['tumor_purity'] for image_path in test_image_paths]

    with open(predict_result_path, 'w') as pred_out_file:
        for image_path, real_value, predict_value in zip(test_image_paths, real_values, predict_values):
            print(*[image_path, real_value, *predict_value], sep='\t', file=pred_out_file)

    print('[LOG] Prediction is terminated.', time.ctime())


def see_model_weights(model_path):
    model = load_model(model_path)

    for layer in model.layers[1:]:
        weights, _ = layer.get_weights()
        print(weights)


def _build_cnn_model(input_tensor, input_shape, input_name):
    conv = Conv2D(8, (2, 1), padding="valid", input_shape=input_shape, name='conv_1')(input_tensor)
    conv = Conv2D(16, (3, 1), padding="same", name='conv_3')(conv)
    conv = Dropout(0.3, name='DO_2')(conv)
    conv = Flatten(name='flatten')(conv)
    conv_out = Dense(128, kernel_initializer='he_uniform', activation='relu', name='fc_1')(conv)

    if input_name == 'var_allele_image':
        conv_out = Dense(3, kernel_initializer='he_uniform', activation='relu', name='fc_read')(conv_out)

    model = Model(inputs=[input_tensor], outputs=conv_out)

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
