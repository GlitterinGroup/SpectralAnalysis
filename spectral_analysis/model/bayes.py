from sklearn.linear_model import BayesianRidge
from sklearn.naive_bayes import GaussianNB


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

class BAYESClassifier(GaussianNB):
    """
    A classifier that extends the Gaussian Naive Bayes classifier.

    Methods:
        train(X, y): Train the classifier with input data X and labels y.
        predict(X): Predict the labels for the input data X.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the BAYESClassifier.

        Args:
            **kwargs: Additional keyword arguments passed to the GaussianNB initializer.
        """
        super().__init__(**kwargs)
    
    def train(self, X, y):
        """
        Train the classifier with input data and labels.

        Args:
            X (array-like): Training data.
            y (array-like): Target labels.
        """
        self.fit(X, y)
    
    def predict(self, X):
        """
        Predict the labels for the input data.

        Args:
            X (array-like): Input data for which to predict labels.

        Returns:
            array: Predicted labels for the input data.
        """
        return super().predict(X)
