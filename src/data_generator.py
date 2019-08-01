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
        # data_keys = ['var_allele_image', 'vaf_hist_image']
        data_keys = ['vaf_hist_array']
        label_key = 'tumor_purity'

        batch_data_dict = {
            'vaf_hist_array': np.empty((self.batch_size, 101))
        }
        batch_labels = np.empty((self.batch_size, self.num_labels), dtype=float)

        # load images from each path of image files
        for i, data_path in enumerate(data_paths):
            with h5py.File(data_path, 'r') as infile:
                data_dict = {}

                for key in infile.keys():
                    data_dict[key] = infile[key].value

            for key in data_keys:
                batch_data_dict[key][i, ] = data_dict[key]

            batch_labels[i, ] = data_dict[label_key]

        return batch_data_dict, batch_labels
