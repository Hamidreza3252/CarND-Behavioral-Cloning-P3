import keras
import numpy as np


class DataGenerator4Classification(keras.utils.Sequence):
    def __init__(self, list_ids, labels, batch_size=32, dim=(32, 32, 3), n_channels=1, n_classes=10, shuffle=True):
        """

        Initialization:

        :param list_ids:
        :param labels:
        :param batch_size:
        :param dim:
        :param n_channels:
        :param n_classes:
        :param shuffle:
        """

        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_ids = list_ids
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

        self.indexes = None
        self.shuffle = True

    def __len__(self):

        """
        Denotes the number of batches per epoch

        :return:
        """

        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        """

        Generate one batch of data

        :param index:
        :return:
        """

        # Generate indexes of the batch
        from_index = index * self.batch_size
        indexes = self.indexes[from_index:from_index + self.batch_size]
        # indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        x_data, y_data = self.__data_generation(list_ids_temp)

        return x_data, y_data

    def on_epoch_end(self):
        """

        Updates indexes after each epoch

        :return:
        """

        self.indexes = np.arange(len(self.list_ids))

        if (self.shuffle == True):
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        """

        Generates data containing batch_size samples # x_data : (n_samples, *dim, n_channels)

        :param list_ids_temp:
        :return:
        """

        # Initialization
        x_data = np.empty((self.batch_size, *self.dim, self.n_channels))
        y_data = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, id in enumerate(list_ids_temp):
            # Store sample
            x_data[i, ] = np.load("data/" + id + ".npy")

            # Store class
            y_data[i] = self.labels[id]

        return x_data, keras.utils.to_categorical(y_data, num_classes=self.n_classes)


class DataGenerator4Regression(keras.utils.Sequence):
    def __init__(self, list_ids, values, batch_size=32, dim=(32, 32, 3), n_channels=1, shuffle=True):
        """

        Initialization:

        :param list_ids:
        :param values:
        :param batch_size:
        :param dim:
        :param n_channels:
        :param shuffle:
        """

        self.dim = dim
        self.batch_size = batch_size
        self.values = values
        self.list_ids = list_ids
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

        self.indexes = None
        self.shuffle = True

    def __len__(self):

        """
        Denotes the number of batches per epoch

        :return:
        """

        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        """

        Generate one batch of data

        :param index:
        :return:
        """

        # Generate indexes of the batch
        from_index = index * self.batch_size
        indexes = self.indexes[from_index:from_index + self.batch_size]
        # indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_ids_temp = [self.list_ids[k] for k in indexes]

        # Generate data
        x_data, y_data = self.__data_generation(list_ids_temp)

        return x_data, y_data

    def on_epoch_end(self):
        """

        Updates indexes after each epoch

        :return:
        """

        self.indexes = np.arange(len(self.list_ids))

        if (self.shuffle == True):
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_ids_temp):
        """

        Generates data containing batch_size samples # x_data : (n_samples, *dim, n_channels)

        :param list_ids_temp:
        :return:
        """

        # Initialization
        x_data = np.empty((self.batch_size, *self.dim, self.n_channels))
        y_data = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, id in enumerate(list_ids_temp):
            # Store sample
            x_data[i, ] = np.load("data/" + id + ".npy")

            # Store class
            y_data[i] = self.labels[id]

        return x_data, keras.utils.to_categorical(y_data, num_classes=self.n_classes)
