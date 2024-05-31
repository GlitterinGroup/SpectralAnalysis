from sklearn.cross_decomposition import PLSRegression


class PLSRegression(PLSRegression):
    """
    A wrapper class for Partial Least Squares (PLS) Regression that provides
    simplified methods for training and prediction.

    This class extends the PLSRegression class from scikit-learn,
    inheriting all its attributes and methods.

    Args:
        n_components (int, optional): Number of components to keep. Default is 2.
        scale (bool, optional): Whether to scale the data. Default is True.
        **kwargs: Arbitrary keyword arguments passed to the PLSRegression constructor.
    
    Methods:
        train(X, y): Train the PLS Regression model using the provided data.
        predict(X): Predict using the trained PLS Regression model.
    """

    def __init__(self, n_components=2, scale=True, **kwargs):
        """
        Initialize the PLSRegression model.

        Args:
            n_components (int, optional): Number of components to keep. Default is 2.
            scale (bool, optional): Whether to scale the data. Default is True.
            **kwargs: Arbitrary keyword arguments passed to the PLSRegression constructor.
        """
        super().__init__(
            n_components=n_components,
            scale=scale,
            **kwargs
        )

    def train(self, X, y):
        """
        Train the PLS Regression model.

        Args:
            X (array-like): Training data, shape (n_samples, n_features).
            y (array-like): Target values, shape (n_samples,).
        """
        self.fit(X, y)

    def predict(self, X):
        """
        Predict using the trained PLS Regression model.

        Args:
            X (array-like): Test data, shape (n_samples, n_features).

        Returns:
            array-like: Predicted values, shape (n_samples, n_components).
        """
        return super().predict(X)
