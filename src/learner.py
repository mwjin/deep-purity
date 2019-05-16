#!/extdata6/Doyeon/anaconda3/envs/deep-purity/bin/python3.6
"""
Make, train, and test our deep learning model
"""
import pickle
import random
import sys
import time
import numpy as np
import tensorflow as tf
import keras
import functools

from keras import backend as kb
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, Input, concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping

# constants
BATCH_SIZE = 32
MAX_EPOCH = 500


class DataGenerator(keras.utils.Sequence):
    """
    it generates a data sequence to train and test our model
    """
    def __init__(self, data_paths, batch_size=16, num_channels=9, num_labels=1, shuffle=True):
        self.batch_size = batch_size
        self.data_paths = data_paths
        self.indices = np.arange(len(self.data_paths))
        self.num_channels = num_channels
        self.num_labels = num_labels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        """
        denotes the number of batches per epoch
        """
        return int(np.floor(len(self.data_paths) / self.batch_size))

    def __getitem__(self, batch_idx):
        """
        generates one batch of data
        """
        indices_in_batch = self.indices[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        batch_data_paths = [self.data_paths[k] for k in indices_in_batch]
        batch_data_dict, batch_labels = self._generate_data(batch_data_paths)

        return batch_data_dict, batch_labels

    def on_epoch_end(self):
        """
        updates indices after each epoch
        """
        self.indices = np.arange(len(self.data_paths))

        if self.shuffle:
            np.random.shuffle(self.indices)

    def _generate_data(self, data_paths):
        """
        generates data containing batch_size samples
        you should modify this function to fit in your data
        """
        data_keys = ['var_allele_image', 'vaf_hist_image']
        label_key = 'tumor_purity'

        batch_data_dict = {
            'var_allele_image': np.empty((self.batch_size, 50, 1000, self.num_channels)),
            'vaf_hist_image': np.empty((self.batch_size, 100, 101, 1))
        }
        batch_labels = np.empty((self.batch_size, self.num_labels), dtype=float)

        # load images from each path of image files
        for i, image_path in enumerate(data_paths):
            with open(image_path, 'rb') as image_file:
                data_dict = pickle.load(image_file)

            for key in data_keys:
                batch_data_dict[key][i, ] = data_dict[key]

            batch_labels[i, ] = data_dict[label_key]

        return batch_data_dict, batch_labels


def make_base_model(base_model_path):
    """
    make our model
    """
    var_allele_img_shape = (50, 1000, 9)
    vaf_hist_img_shape = (100, 101, 1)

    var_allele_img_name = 'var_allele_image'
    vaf_hist_img_name = 'vaf_hist_image'

    var_allele_img_in_tensor = Input(shape=var_allele_img_shape, name=var_allele_img_name)
    vaf_hist_img_in_tensor = Input(shape=vaf_hist_img_shape, name=vaf_hist_img_name)

    img_info_list = [
        [var_allele_img_in_tensor, var_allele_img_shape, var_allele_img_name],
        [vaf_hist_img_in_tensor, vaf_hist_img_shape, vaf_hist_img_name]
    ]

    # build a CNN model for each image
    cnn_model_outputs = []  # a CNN model for each image

    for input_tensor, shape, name in img_info_list:
        cnn_model = _build_cnn_model(input_tensor=input_tensor, input_shape=shape, input_name=name)

        for layer in cnn_model.layers[1:]:
            layer.name = layer.name + name

        cnn_model_outputs.append(cnn_model.output)

    # build a fully connected layer and an output layer
    concat_cnn_output = concatenate(cnn_model_outputs)
    full_conn_layer = Dense(16, kernel_initializer='he_uniform', activation='relu')(concat_cnn_output)
    pred_out_layer = Dense(1, activation=None, name='output')(full_conn_layer)

    model = Model(inputs=[var_allele_img_in_tensor, vaf_hist_img_in_tensor], outputs=pred_out_layer)
    model.compile(loss='mean_squared_error', optimizer='adam')

    print(model.summary())
    model.save(base_model_path)


def train_model(train_model_path, base_model_path, train_img_set_path, valid_img_set_path):
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
    train_image_paths = random.sample(train_image_paths, int(len(train_image_paths) * 0.7))
    valid_image_paths = random.sample(valid_image_paths, int(len(valid_image_paths) * 0.3))

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
    early_stop_cond = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
    model.fit_generator(train_data_generator, epochs=MAX_EPOCH, callbacks=[model_ckpt, early_stop_cond],
                        validation_data=valid_data_generator, verbose=1)

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


if __name__ == '__main__':
    if len(sys.argv) == 1:
        sys.exit(f'[LOG] Please enter a function name and corresponding arguments')
    else:
        function_name = sys.argv[1]
        function_args = sys.argv[2:]

        if function_name in locals().keys():
            locals()[function_name](*function_args)
        else:
            sys.exit(f'[ERROR] Unavailable function {function_name}')
