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
    def calculate_rc(y_true_train, y_pred_train):
        """
        Calculates the Regression Coefficient (RC) on the training set.

        Args:
            y_true_train: Array of true values for the training set.
            y_pred_train: Array of predicted values for the training set.

        Returns:
            float: RC value calculated using the provided true and predicted values.
        """
        return Metrics.calculate_r2(y_true_train, y_pred_train)

    @staticmethod
    def calculate_rp(y_true, y_pred):
        """
        Calculates the Regression Weight (RP) on the test set.

        Args:
            y_true: Array of true values for the test set.
            y_pred: Array of predicted values for the test set.

        Returns:
            float: RP value calculated using the provided true and predicted values.
        """
        return Metrics.calculate_r2(y_true, y_pred)

    def calculate_rmsec(y_true_train, y_pred_train):
        """
        Calculates the Root Mean Squared Error of Calibration (RMSEC).

        Args:
            y_true_train: Array of true values for the training set.
            y_pred_train: Array of predicted values for the training set.

        Returns:
            float: RMSEC value calculated using the provided true and predicted values.
        """
        return Metrics.calculate_rmse(y_true_train, y_pred_train)

    def calculate_rmsep(y_true, y_pred):
        """
        Calculates the Root Mean Squared Error of Prediction (RMSEP).

        Args:
            y_true: Array of true values for the test set.
            y_pred: Array of predicted values for the test set.

        Returns:
            float: RMSEP value calculated using the provided true and predicted values.
        """
        return Metrics.calculate_rmse(y_true, y_pred)

    def calculate_rpd(y_true, y_pred, rmse_func):
        """
        Calculates the Relative Predictive Deviation (RPD).

        Args:
            y_true: Array of true values.
            y_pred: Array of predicted values.
            rmse_func: Function to calculate RMSEP.

        Returns:
            float: RPD value calculated as the ratio of the standard deviation of y_true to RMSEP.

        Raises:
            ValueError: If RMSEP is zero, making RPD undefined.
        """
        sd = np.std(y_true)
        rmsep = rmse_func(y_true, y_pred)  # 使用提供的RMSEP计算函数
        if rmsep == 0:
            raise ValueError("RMSEP cannot be zero, which would make RPD undefined.")
        return sd / rmsep

    @staticmethod
    def calculate_metrics(y_true_train, y_pred_train, y_true_test, y_pred_test):
        """
        Calculate multiple metrics and return them in a dictionary.

        Args:
            y_true_train: Array of true values for the training set.
            y_pred_train: Array of predicted values for the training set.
            y_true_test: Array of true values for the test set.
            y_pred_test: Array of predicted values for the test set.

        Returns:
            Dictionary containing the calculated metrics:
                'mae': Mean Absolute Error (MAE) on the test set.
                'mse': Mean Squared Error (MSE) on the test set.
                'rmse': Root Mean Squared Error (RMSE) on the test set.
                'r2': Coefficient of Determination (R^2) on the test set.
                'r': Correlation coefficient (R) between true and predicted values on the test set.
                'Rc': Regression Coefficient (RC) calculated using true and predicted values from training set.
                'Rp': Regression Weight (RP) calculated using true and predicted values from test set.
                'RMSEC': Root Mean Squared Error of Calibration (RMSEC) on the training set.
                'RMSEP': Root Mean Squared Error of Prediction (RMSEP) on the test set.
        """

        return {
            "mae": Metrics.calculate_mae(y_true_test, y_pred_test),
            "mse": Metrics.calculate_mse(y_true_test, y_pred_test),
            "rmse": Metrics.calculate_rmse(y_true_test, y_pred_test),
            "r2": Metrics.calculate_r2(y_true_test, y_pred_test),
            "r": Metrics.calculate_correlation(y_true_test, y_pred_test),
            "Rc": Metrics.calculate_rc(y_true_train, y_pred_train),
            "Rp": Metrics.calculate_rp(y_true_test, y_pred_test),
            "RMSEC": Metrics.calculate_rmsec(y_true_train, y_pred_train),
            "RMSEP": Metrics.calculate_rmsep(y_true_test, y_pred_test),
        }
