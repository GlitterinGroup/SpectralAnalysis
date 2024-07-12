import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)


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
    def calculate_r(y_true, y_pred):
        """
        Calculate the Pearson correlation coefficient (r) between the true and predicted values.

        Args:
            y_true (array-like): True values of the target variable.
            y_pred (array-like): Predicted values of the target variable.

        Returns:
            float: Pearson correlation coefficient (r).
        """
        return np.corrcoef(y_true, y_pred)[0, 1]

    @staticmethod
    def calculate_rc(y_true_train, y_pred_train):
        """
        Calculate the calibration correlation coefficient (Rc) on the training set.

        Args:
            y_true_train: Array of true values for the training set.
            y_pred_train: Array of predicted values for the training set.

        Returns:
            float: Rc value calculated using the provided true and predicted values.
        """
        return Metrics.calculate_r(y_true_train, y_pred_train)

    @staticmethod
    def calculate_rp(y_true, y_pred):
        """
        Calculate the prediction correlation coefficient (Rp) on the test set.

        Args:
            y_true: Array of true values for the test set.
            y_pred: Array of predicted values for the test set.

        Returns:
            float: Rp value calculated using the provided true and predicted values.
        """
        return Metrics.calculate_r(y_true, y_pred)


    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def calculate_rpd(y_true, y_pred):
        """
        Calculates the Relative Predictive Deviation (RPD).

        Args:
            y_true: Array of true values.
            y_pred: Array of predicted values.

        Returns:
            float: RPD value calculated as the ratio of the standard deviation of y_true to RMSEP.

        Raises:
            ValueError: If RMSEP is zero, making RPD undefined.
        """
        sd = np.std(y_true)
        rmsep = Metrics.calculate_rmsep(y_true, y_pred)
        if rmsep == 0:
            raise ValueError("RMSEP cannot be zero, which would make RPD undefined.")
        return sd / rmsep

    @staticmethod
    def calculate_regression_metrics(y_true_train, y_pred_train, y_true_test, y_pred_test):
        """
        Calculate multiple metrics and return them in a dictionary.

        Args:
            y_true_train: Array of true values for the training set.
            y_pred_train: Array of predicted values for the training set.
            y_true_test: Array of true values for the test set.
            y_pred_test: Array of predicted values for the test set.

        Returns:
            Dictionary containing the calculated metrics:
            - 'mae': Mean Absolute Error (MAE) on the test set.
            - 'mse': Mean Squared Error (MSE) on the test set.
            - 'rmse': Root Mean Squared Error (RMSE) on the test set.
            - 'r2': Coefficient of Determination (R^2) on the test set.
            - 'r': Correlation coefficient (R) between true and predicted values on the test set.
            - 'Rc': Regression Coefficient (RC) calculated using true and predicted values from training set.
            - 'Rp': Regression Weight (RP) calculated using true and predicted values from test set.
            - 'RMSEC': Root Mean Squared Error of Calibration (RMSEC) on the training set.
            - 'RMSEP': Root Mean Squared Error of Prediction (RMSEP) on the test set.
            - 'RPD': Relative Predictive Deviation (RPD) on the test set.
        """

        return {
            "mae": Metrics.calculate_mae(y_true_test, y_pred_test),
            "mse": Metrics.calculate_mse(y_true_test, y_pred_test),
            "rmse": Metrics.calculate_rmse(y_true_test, y_pred_test),
            "r2": Metrics.calculate_r2(y_true_test, y_pred_test),
            "r": Metrics.calculate_r(y_true_test, y_pred_test),
            "Rc": Metrics.calculate_rc(y_true_train, y_pred_train),
            "Rp": Metrics.calculate_rp(y_true_test, y_pred_test),
            "RMSEC": Metrics.calculate_rmsec(y_true_train, y_pred_train),
            "RMSEP": Metrics.calculate_rmsep(y_true_test, y_pred_test),
            "RPD": Metrics.calculate_rpd(y_true_test, y_pred_test),
        }

    @staticmethod
    def calculate_accuracy(y_true, y_pred):
        """
        Calculate accuracy score.

        Args:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.

        Returns:
            float: Accuracy score.
        """
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def calculate_precision(y_true, y_pred, average='binary'):
        """
        Calculate precision score.

        Args:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.
            average (str, optional): Type of averaging performed on the data. 
                Defaults to 'binary'. Options: ['binary', 'micro', 'macro', 'samples', 'weighted'].

        Returns:
            float: Precision score.
        """
        return precision_score(y_true, y_pred, average=average)

    @staticmethod
    def calculate_recall(y_true, y_pred, average='binary'):
        """
        Calculate recall score.

        Args:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.
            average (str, optional): Type of averaging performed on the data. 
                Defaults to 'binary'. Options: ['binary', 'micro', 'macro', 'samples', 'weighted'].

        Returns:
            float: Recall score.
        """
        return recall_score(y_true, y_pred, average=average)

    @staticmethod
    def calculate_f1(y_true, y_pred, average='binary'):
        """
        Calculate F1 score.

        Args:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.
            average (str, optional): Type of averaging performed on the data. 
                Defaults to 'binary'. Options: ['binary', 'micro', 'macro', 'samples', 'weighted'].

        Returns:
            float: F1 score.
        """
        return f1_score(y_true, y_pred, average=average)

    @staticmethod
    def calculate_auc_roc(y_true, y_pred_proba):
        """
        Calculate Area Under the Receiver Operating Characteristic Curve (ROC AUC).

        Args:
            y_true (array-like): True labels.
            y_pred_proba (array-like): Predicted probabilities.

        Returns:
            float: ROC AUC score.
        """
        return roc_auc_score(y_true, y_pred_proba)

    @staticmethod
    def calculate_classification_metrics(y_true, y_pred, y_pred_proba=None, average='binary'):
        """
        Calculate various classification metrics.

        Args:
            y_true (array-like): True labels.
            y_pred (array-like): Predicted labels.
            y_pred_proba (array-like, optional): Predicted probabilities. Defaults to None.
            average (str, optional): Type of averaging performed on the data. 
                Defaults to 'binary'. Options: ['binary', 'micro', 'macro', 'samples', 'weighted'].

        Returns:
            dict: Dictionary of classification metrics, including accuracy, precision, recall, F1 score,
                and optionally ROC AUC if `y_pred_proba` is provided.
        """
        metrics = {
            "accuracy": Metrics.calculate_accuracy(y_true, y_pred),
            "precision": Metrics.calculate_precision(y_true, y_pred, average=average),
            "recall": Metrics.calculate_recall(y_true, y_pred, average=average),
            "f1": Metrics.calculate_f1(y_true, y_pred, average=average)
        }
        if y_pred_proba is not None:
            metrics["auc_roc"] = Metrics.calculate_auc_roc(y_true, y_pred_proba)
        return metrics