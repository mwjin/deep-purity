"""
This is a data generator for learning our model.
"""
import keras
import numpy as np
import h5py


class DataGenerator(keras.utils.Sequence):
    """
    it generates a data sequence to train and test our model
    """
    def __init__(self, data_paths, input_keys, output_key, batch_size, shuffle=True):
        self.batch_size = batch_size
        self.data_paths = data_paths
        self.indices = np.arange(len(self.data_paths))
        self.input_keys = input_keys
        self.output_key = output_key
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

    def _generate_data(self, batch_data_paths):
        """
        generates data containing batch_size samples
        you should modify this function to fit in your data
        """
        batch_data_dict = {
            'vaf_hist_array': np.empty((self.batch_size, 101)),
            'vaf_lrr_image': np.empty((self.batch_size, 1000, 4, 1))
        }
        batch_labels = np.empty((self.batch_size, 1), dtype=float)

        # load images from each path of image files
        for i, data_path in enumerate(batch_data_paths):
            with h5py.File(data_path, 'r') as data:
                data_dict = {}

                for data_key in data.keys():
                    data_dict[data_key] = data[data_key].value

            for input_key in self.input_keys:
                if batch_data_dict[input_key] is None:
                    batch_data_dict[input_key] = np.empty((self.batch_size, *data_dict[input_key].shape))

                batch_data_dict[input_key][i, ] = data_dict[input_key]

            batch_labels[i, ] = data_dict[self.output_key]

        return batch_data_dict, batch_labels
