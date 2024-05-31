import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold, train_test_split


class DataSplit:
    """
    A class that provides various methods for splitting datasets into training and test sets.
    """

    @staticmethod
    def random_split(X, y, test_size=0.3):
        """
        Randomly split the data into training and test sets.

        Args:
            X (numpy.ndarray): Input data array. shape (n_samples, n_features)
            y (numpy.ndarray): Target data array. shape (n_samples,)
            test_size (float, optional): If float, should be between 0.0 and 1.0 and represent
                                        the proportion of the dataset to include in the test split.
                                        If int, represents the absolute number of test samples.

        Returns:
            tuple: (X_train, X_test, y_train, y_test) where each element is a numpy.ndarray.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        return X_train, X_test, y_train, y_test

    @staticmethod
    def last_n_split(X, y, test_size):
        """
        Split the data such that the last 'test_size' samples are used as the test set.

        Args:
            X (numpy.ndarray): Input data array. shape (n_samples, n_features)
            y (numpy.ndarray): Target data array. shape (n_samples,)
            test_size (int): The number of samples to include in the test split.

        Returns:
            tuple: (X_train, X_test, y_train, y_test) where each element is a numpy.ndarray.
        """
        if len(X) != len(y):
            raise ValueError("The length of X and y must be the same")

        if test_size > len(X):
            raise ValueError(
                "test_size cannot be larger than the total number of samples"
            )

        X_train = X[:-test_size]
        X_test = X[-test_size:]
        y_train = y[:-test_size]
        y_test = y[-test_size:]

        return X_train, X_test, y_train, y_test

    @staticmethod
    def kfold_split(X, y, n_splits):
        """
        Split the data into 'n_splits' folds for cross-validation.

        Args:
            X (numpy.ndarray): Input data array. shape (n_samples, n_features)
            y (numpy.ndarray): Target data array. shape (n_samples,)
            n_splits (int): The number of folds.

        Returns:
            list: A list of tuples, each containing the training and test splits of X and y for a fold.
        """
        if len(X) != len(y):
            raise ValueError("The length of X and y must be the same")

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            splits.append((X_train, X_test, y_train, y_test))

        return splits

    @staticmethod
    def _max_min_distance_split(distance, train_size):
        """
        Perform a max-min distance split.

        Args:
            distance (numpy.ndarray): Pairwise distance matrix. shape (n_samples, n_samples)
            train_size (int): Number of samples to include in the train split.

        Returns:
            tuple: (index_train, index_test) where each element is a list of indices
        """
        i_train = []
        i_test = [i for i in range(distance.shape[0])]

        first_2_points = np.unravel_index(np.argmax(distance), distance.shape)

        i_train.append(first_2_points[0])
        i_train.append(first_2_points[1])

        i_test.remove(first_2_points[0])
        i_test.remove(first_2_points[1])

        for _ in range(train_size - 2):
            max_min_dist_idx = np.argmax(np.min(distance[i_train, :], axis=0))

            i_train.append(max_min_dist_idx)
            i_test.remove(max_min_dist_idx)

        return i_train, i_test

    @staticmethod
    def ks_split(X, y, test_size):
        """
        Perform Kennard-Stone (KS) train-test split.

        Args:
            X (numpy.ndarray): Input data array. shape (n_samples, n_features)
            y (numpy.ndarray): Target data array. shape (n_samples,)
            test_size (float or int): If float, should be between 0.0 and 1.0 and represent
                                    the proportion of the dataset to include in the test split.
                                    If int, represents the absolute number of test samples.

        Returns:
            tuple: (x_train, x_test, y_train, y_test) where each element is a numpy.ndarray
        """
        distance = cdist(X, X)
        train_size = (
            int(X.shape[0] * (1 - test_size))
            if 0 < test_size < 1
            else X.shape[0] - test_size
        )
        index_train, index_test = DataSplit._max_min_distance_split(
            distance, train_size
        )
        x_train, x_test, y_train, y_test = (
            X[index_train],
            X[index_test],
            y[index_train],
            y[index_test],
        )
        return x_train, x_test, y_train, y_test

    @staticmethod
    def spxy_split(X, y, test_size):
        """
        Perform SPXY (Sample set Partitioning based on joint X-Y distances) train-test split.

        Args:
            X (numpy.ndarray): Input data array. shape (n_samples, n_features)
            y (numpy.ndarray): Target data array. shape (n_samples,)
            test_size (float or int): If float, should be between 0.0 and 1.0 and represent
                                    the proportion of the dataset to include in the test split.
                                    If int, represents the absolute number of test samples.

        Returns:
            tuple: (x_train, x_test, y_train, y_test) where each element is a numpy.ndarray
        """
        y = np.expand_dims(y, axis=-1)

        distance_x = cdist(X, X)
        distance_y = cdist(y, y)

        distance_x /= distance_x.max()
        distance_y /= distance_y.max()

        distance = distance_x + distance_y

        train_size = (
            int(X.shape[0] * (1 - test_size))
            if 0 < test_size < 1
            else X.shape[0] - test_size
        )
        index_train, index_test = DataSplit._max_min_distance_split(
            distance, train_size
        )
        x_train, x_test, y_train, y_test = (
            X[index_train],
            X[index_test],
            y[index_train],
            y[index_test],
        )

        y_train = y_train.flatten()
        y_test = y_test.flatten()

        return x_train, x_test, y_train, y_test
