import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class Metrics:
    """
    A class to encapsulate various evaluation metrics for regression models.
    """

    @staticmethod
    def calculate_r2(y_true, y_pred):
        """
        Calculate the R-squared (coefficient of determination) regression score.

        Args:
            y_true (array-like): True values of the target variable.
            y_pred (array-like): Predicted values of the target variable.

        Returns:
            float: R-squared score.
        """
        return r2_score(y_true, y_pred)

    @staticmethod
    def calculate_mae(y_true, y_pred):
        """
        Calculate the Mean Absolute Error (MAE) regression loss.

        Args:
            y_true (array-like): True values of the target variable.
            y_pred (array-like): Predicted values of the target variable.

        Returns:
            float: Mean Absolute Error.
        """
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def calculate_mse(y_true, y_pred):
        """
        Calculate the Mean Squared Error (MSE) regression loss.

        Args:
            y_true (array-like): True values of the target variable.
            y_pred (array-like): Predicted values of the target variable.

        Returns:
            float: Mean Squared Error.
        """
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def calculate_rmse(y_true, y_pred):
        """
        Calculate the Root Mean Squared Error (RMSE) regression loss.

        Args:
            y_true (array-like): True values of the target variable.
            y_pred (array-like): Predicted values of the target variable.

        Returns:
            float: Root Mean Squared Error.
        """
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def calculate_correlation(y_true, y_pred):
        """
        Calculate the Pearson correlation coefficient between the true and predicted values.

        Args:
            y_true (array-like): True values of the target variable.
            y_pred (array-like): Predicted values of the target variable.

        Returns:
            float: Pearson correlation coefficient.
        """
        return np.corrcoef(y_true, y_pred)[0, 1]

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        """
        Calculate multiple regression metrics and return them in a dictionary.

        Args:
            y_true (array-like): True values of the target variable.
            y_pred (array-like): Predicted values of the target variable.

        Returns:
            dict: Dictionary containing MAE, MSE, RMSE, R-squared, and correlation coefficient.
        """
        return {
            "mae": Metrics.calculate_mae(y_true, y_pred),
            "mse": Metrics.calculate_mse(y_true, y_pred),
            "rmse": Metrics.calculate_rmse(y_true, y_pred),
            "r2": Metrics.calculate_r2(y_true, y_pred),
            "r": Metrics.calculate_correlation(y_true, y_pred),
        }
