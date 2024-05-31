from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class RFRegression(RandomForestRegressor):
    """
    A wrapper class for RandomForestRegressor that provides simplified methods for training and prediction.

    This class extends the RandomForestRegressor class from scikit-learn, inheriting all its attributes and methods.

    Args:
        n_estimators (int, optional): The number of trees in the forest. Default is 50.
        **kwargs: Arbitrary keyword arguments passed to the RandomForestRegressor constructor.
    
    Methods:
        train(X, y): Train the RandomForestRegressor model using the provided data.
        predict(X): Predict using the trained RandomForestRegressor model.
    """

    def __init__(self, n_estimators=50, **kwargs):
        """
        Initialize the RFRegression model.

        Args:
            n_estimators (int, optional): The number of trees in the forest. Default is 50.
            **kwargs: Arbitrary keyword arguments passed to the RandomForestRegressor constructor.
        """
        super().__init__(n_estimators=n_estimators, **kwargs)

    def train(self, X, y):
        """
        Train the RandomForestRegressor model.

        Args:
            X (array-like): Training data, shape (n_samples, n_features).
            y (array-like): Target values, shape (n_samples,).
        """
        self.fit(X, y)

    def predict(self, X):
        """
        Predict using the trained RandomForestRegressor model.

        Args:
            X (array-like): Test data, shape (n_samples, n_features).

        Returns:
            array-like: Predicted values, shape (n_samples,).
        """
        return super().predict(X)


class RFClassifier(RandomForestClassifier):
    """
    A wrapper class for RandomForestClassifier that provides simplified methods for training and prediction.

    This class extends the RandomForestClassifier class from scikit-learn, inheriting all its attributes and methods.

    Args:
        n_estimators (int, optional): The number of trees in the forest. Default is 50.
        **kwargs: Arbitrary keyword arguments passed to the RandomForestClassifier constructor.
    
    Methods:
        train(X, y): Train the RandomForestClassifier model using the provided data.
        predict(X): Predict using the trained RandomForestClassifier model.
    """

    def __init__(self, n_estimators=50, **kwargs):
        """
        Initialize the RFClassifier model.

        Args:
            n_estimators (int, optional): The number of trees in the forest. Default is 50.
            **kwargs: Arbitrary keyword arguments passed to the RandomForestClassifier constructor.
        """
        super().__init__(n_estimators=n_estimators, **kwargs)

    def train(self, X, y):
        """
        Train the RandomForestClassifier model.

        Args:
            X (array-like): Training data, shape (n_samples, n_features).
            y (array-like): Target values, shape (n_samples,).
        """
        self.fit(X, y)

    def predict(self, X):
        """
        Predict using the trained RandomForestClassifier model.

        Args:
            X (array-like): Test data, shape (n_samples, n_features).

        Returns:
            array-like: Predicted labels, shape (n_samples,).
        """
        return super().predict(X)
