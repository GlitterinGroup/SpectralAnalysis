import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import (linear_kernel, polynomial_kernel,
                                      rbf_kernel, sigmoid_kernel)


class LSSVMRegression(BaseEstimator, RegressorMixin):
    """
    Least Squares Support Vector Machine (LS-SVM) regression model.

    Args:
        C (float): Regularization parameter. Default is 1000.
        kernel (str): Kernel type to be used in the algorithm. It must be one of 'linear', 'poly', 'rbf', 'sigmoid'. Default is 'rbf'.
        gamma (str or float): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If 'scale', uses 1 / (n_features * X.var()). If 'auto', uses 1 / n_features. Default is 'scale'.
        degree (int): Degree of the polynomial kernel function ('poly'). Ignored by other kernels. Default is 3.
        coef0 (float): Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'. Default is 0.0.
        tol (float): Tolerance for stopping criterion. Default is 1e-3.
        epsilon (float): Epsilon in the epsilon-SVR model. Default is 0.1.
        max_iter (int): Hard limit on iterations within solver, or -1 for no limit. Default is -1.
    """

    def __init__(
        self,
        C=1000,
        kernel="rbf",
        gamma="scale",
        degree=3,
        coef0=0.0,
        tol=1e-3,
        epsilon=0.1,
        max_iter=-1,
    ):
        """
        Initializes the LSSVMRegression model with the given parameters.

        Args:
            C (float): Regularization parameter. Default is 1000.
            kernel (str): Kernel type to be used in the algorithm. It must be one of 'linear', 'poly', 'rbf', 'sigmoid'. Default is 'rbf'.
            gamma (str or float): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If 'scale', uses 1 / (n_features * X.var()). If 'auto', uses 1 / n_features. Default is 'scale'.
            degree (int): Degree of the polynomial kernel function ('poly'). Ignored by other kernels. Default is 3.
            coef0 (float): Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'. Default is 0.0.
            tol (float): Tolerance for stopping criterion. Default is 1e-3.
            epsilon (float): Epsilon in the epsilon-SVR model. Default is 0.1.
            max_iter (int): Hard limit on iterations within solver, or -1 for no limit. Default is -1.
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.epsilon = epsilon
        self.max_iter = max_iter

    def _compute_kernel(self, X, Y=None):
        """
        Computes the kernel between X and Y.

        Args:
            X (numpy.ndarray): Input data.
            Y (numpy.ndarray, optional): Second input data. If None, computes the kernel with X itself.

        Returns:
            numpy.ndarray: The computed kernel matrix.
        """
        if self.kernel == "linear":
            return linear_kernel(X, Y)
        elif self.kernel == "poly":
            return polynomial_kernel(
                X, Y, degree=self.degree, gamma=self.gamma, coef0=self.coef0
            )
        elif self.kernel == "rbf":
            return rbf_kernel(X, Y, gamma=self.gamma)
        elif self.kernel == "sigmoid":
            return sigmoid_kernel(X, Y, gamma=self.gamma, coef0=self.coef0)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel}")

    def train(self, X, y):
        """
        Trains the LS-SVM model on the given data.

        Args:
            X (numpy.ndarray): Training data.
            y (numpy.ndarray): Target values.

        Returns:
            LSSVMRegression: The trained model.
        """
        n_samples, n_features = X.shape

        if self.gamma == "scale":
            self.gamma = 1 / (n_features * X.var())
        elif self.gamma == "auto":
            self.gamma = 1 / n_features

        K = self._compute_kernel(X)
        one = np.ones((n_samples, 1))
        O = np.zeros((1, 1))
        K = K + np.eye(n_samples) / self.C

        A = np.block([[O, one.T], [one, K]])
        b = np.concatenate([[0], y])

        sol = np.linalg.inv(A).dot(b)
        self.bias_ = sol[0]
        self.alpha_ = sol[1:]
        self.X_train_ = X

        return self

    def predict(self, X):
        """
        Predicts target values for the given data.

        Args:
            X (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Predicted target values.
        """
        K = self._compute_kernel(X, self.X_train_)
        return np.dot(K, self.alpha_) + self.bias_
