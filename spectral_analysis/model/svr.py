from sklearn.svm import SVR


class SVRRegression(SVR):
    """
    A wrapper class for the Support Vector Regressor (SVR) from scikit-learn.

    This class extends the SVR class from scikit-learn, inheriting all its attributes and methods, and provides simplified methods for training and prediction.

    Args:
        kernel (str, optional): Specifies the kernel type to be used in the algorithm. Default is 'rbf'.
        C (float, optional): Regularization parameter. Default is 1.0.
        gamma (str or float, optional): Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. Default is 'scale'.
        **kwargs: Arbitrary keyword arguments passed to the SVR constructor.
    
    Methods:
        train(X, y): Train the SVR model using the provided data.
        predict(X): Predict using the trained SVR model.
    """

    def __init__(self, kernel="rbf", C=1.0, gamma='scale', **kwargs):
        """
        Initialize the SVRRegression model.

        Args:
            kernel (str, optional): Specifies the kernel type to be used in the algorithm. Default is 'rbf'.
            C (float, optional): Regularization parameter. Default is 1.0.
            gamma (str or float, optional): Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. Default is 'scale'.
            **kwargs: Arbitrary keyword arguments passed to the SVR constructor.
        """
        super().__init__(kernel=kernel, C=C, gamma=gamma, **kwargs)

    def train(self, X, y):
        """
        Train the SVR model.

        Args:
            X (array-like): Training data, shape (n_samples, n_features).
            y (array-like): Target values, shape (n_samples,).
        """
        self.fit(X, y)

    def predict(self, X):
        """
        Predict using the trained SVR model.

        Args:
            X (array-like): Test data, shape (n_samples, n_features).

        Returns:
            array-like: Predicted values, shape (n_samples,).
        """
        return super().predict(X)
