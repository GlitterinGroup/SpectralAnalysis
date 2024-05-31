from sklearn.linear_model import BayesianRidge


class BAYESRegression(BayesianRidge):
    """
    A wrapper class for Bayesian Ridge Regression that provides
    simplified methods for training and prediction.

    This class extends the BayesianRidge class from scikit-learn,
    inheriting all its attributes and methods.

    Methods:
        train(X, y): Train the Bayesian Ridge Regression model using the provided data.
        predict(X): Predict using the trained Bayesian Ridge Regression model.
    """

    def __init__(self, **kwargs):
        """
        Initialize the BAYESRegression model.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the BayesianRidge constructor.
        """
        super().__init__(**kwargs)

    def train(self, X, y):
        """
        Train the Bayesian Ridge Regression model.

        Args:
            X (array-like): Training data, shape (n_samples, n_features).
            y (array-like): Target values, shape (n_samples,).
        """
        self.fit(X, y)

    def predict(self, X):
        """
        Predict using the trained Bayesian Ridge Regression model.

        Args:
            X (array-like): Test data, shape (n_samples, n_features).

        Returns:
            array-like: Predicted values, shape (n_samples,).
        """
        return super().predict(X)
