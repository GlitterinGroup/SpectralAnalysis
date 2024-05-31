from sklearn.linear_model import LinearRegression, LogisticRegression


class LRRegression(LinearRegression):
    """
    A wrapper class for Linear Regression that provides
    simplified methods for training and prediction.

    This class extends the LinearRegression class from scikit-learn,
    inheriting all its attributes and methods.

    Methods:
        train(X, y): Train the Linear Regression model using the provided data.
        predict(X): Predict using the trained Linear Regression model.
    """

    def __init__(self, **kwargs):
        """
        Initialize the LRRegression model.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the LinearRegression constructor.
        """
        super().__init__(**kwargs)

    def train(self, X, y):
        """
        Train the Linear Regression model.

        Args:
            X (array-like): Training data, shape (n_samples, n_features).
            y (array-like): Target values, shape (n_samples,).
        """
        self.fit(X, y)

    def predict(self, X):
        """
        Predict using the trained Linear Regression model.

        Args:
            X (array-like): Test data, shape (n_samples, n_features).

        Returns:
            array-like: Predicted values, shape (n_samples,).
        """
        return super().predict(X)

class LRClassifier(LogisticRegression):
    """
    A wrapper class for Logistic Regression that provides
    simplified methods for training and prediction.

    This class extends the LogisticRegression class from scikit-learn,
    inheriting all its attributes and methods.

    Methods:
        train(X, y): Train the Logistic Regression model using the provided data.
        predict(X): Predict using the trained Logistic Regression model.
    """

    def __init__(self, **kwargs):
        """
        Initialize the LRClassifier model.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the LogisticRegression constructor.
        """
        super().__init__(**kwargs)

    def train(self, X, y):
        """
        Train the Logistic Regression model.

        Args:
            X (array-like): Training data, shape (n_samples, n_features).
            y (array-like): Target values, shape (n_samples,).
        """
        self.fit(X, y)

    def predict(self, X):
        """
        Predict using the trained Logistic Regression model.

        Args:
            X (array-like): Test data, shape (n_samples, n_features).

        Returns:
            array-like: Predicted class labels, shape (n_samples,).
        """
        return super().predict(X)
