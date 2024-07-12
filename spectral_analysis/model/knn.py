from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


class KNNRegression:
    """
    A regression model using k-Nearest Neighbors (k-NN).

    Methods:
        train(X, y): Train the regression model with input data X and target values y.
        predict(X): Predict the target values for the input data X.
    """

    def __init__(self, n_neighbors=5, weights='uniform'):
        """
        Initialize the KNNRegression model.

        Args:
            n_neighbors (int, optional): Number of neighbors to use. Defaults to 5.
            weights (str, optional): Weight function used in prediction. Defaults to 'uniform'.
        """
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights)
    
    def train(self, X, y):
        """
        Train the regression model with input data and target values.

        Args:
            X (array-like): Training data.
            y (array-like): Target values.
        """
        self.model.fit(X, y)
    
    def predict(self, X):
        """
        Predict the target values for the input data.

        Args:
            X (array-like): Input data for which to predict target values.

        Returns:
            array: Predicted target values for the input data.
        """
        return self.model.predict(X)

class KNNClassifier:
    """
    A classification model using k-Nearest Neighbors (k-NN).

    Methods:
        train(X, y): Train the classification model with input data X and target labels y.
        predict(X): Predict the labels for the input data X.
    """

    def __init__(self, n_neighbors=5, weights='uniform'):
        """
        Initialize the KNNClassifier model.

        Args:
            n_neighbors (int, optional): Number of neighbors to use. Defaults to 5.
            weights (str, optional): Weight function used in prediction. Defaults to 'uniform'.
        """
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
    
    def train(self, X, y):
        """
        Train the classification model with input data and target labels.

        Args:
            X (array-like): Training data.
            y (array-like): Target labels.
        """
        self.model.fit(X, y)
    
    def predict(self, X):
        """
        Predict the labels for the input data.

        Args:
            X (array-like): Input data for which to predict labels.

        Returns:
            array: Predicted labels for the input data.
        """
        return self.model.predict(X)
