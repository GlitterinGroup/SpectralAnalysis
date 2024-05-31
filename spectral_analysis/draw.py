import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams["font.family"] = "SimHei"
plt.rcParams["axes.unicode_minus"] = False


class Draw:
    """
    A class to encapsulate various plotting functions for spectral data visualization.
    """

    @staticmethod
    def plot_wavelength_line(wavelengths, data, save_path, save_name, title=None):
        """
        Plot line graphs for spectral data.

        Args:
            wavelengths (array-like): Array of wavelength values.
            data (array-like): Spectral data array, shape (n_samples, n_wavelengths).
            save_path (str): Directory path to save the plot.
            save_name (str): Filename to save the plot.
            title (str, optional): Title of the plot. Default is None.

        Returns:
            None
        """
        plt.figure(figsize=(10, 5))
        for d in data:
            plt.plot(wavelengths, d, linewidth=1)
        plt.xlabel("Wavelength(nm)")
        # plt.ylabel("Reflectance")
        plt.title(title)
        plt.savefig(
            f"{save_path}/wave_line_{save_name}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    @staticmethod
    def plot_wavelength_scatter(wavelengths, data, save_path, save_name, title=None):
        """
        Plot scatter graphs for spectral data.

        Args:
            wavelengths (array-like): Array of wavelength values.
            data (array-like): Spectral data array, shape (n_samples, n_wavelengths).
            save_path (str): Directory path to save the plot.
            save_name (str): Filename to save the plot.
            title (str, optional): Title of the plot. Default is None.

        Returns:
            None
        """
        plt.figure(figsize=(10, 5))
        for d in data:
            plt.scatter(wavelengths, d, s=1)
        plt.xlabel("Wavelength(nm)")
        # plt.ylabel("Reflectance")
        plt.title(title)
        plt.savefig(
            f"{save_path}/wave_scatter_{save_name}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    @staticmethod
    def plot_train_test_scatter(
        wavelengths, train_data, test_data, save_path, save_name, title=None
    ):
        """
        Plot scatter graphs for train and test spectral data.

        Args:
            wavelengths (array-like): Array of wavelength values.
            train_data (array-like): Training spectral data array, shape (n_train_samples, n_wavelengths).
            test_data (array-like): Testing spectral data array, shape (n_test_samples, n_wavelengths).
            save_path (str): Directory path to save the plot.
            save_name (str): Filename to save the plot.
            title (str, optional): Title of the plot. Default is None.

        Returns:
            None
        """
        plt.figure(figsize=(10, 5))

        for d in train_data:
            plt.scatter(wavelengths, d, s=0.1, c="b", alpha=0.1)

        for d in test_data:
            plt.scatter(wavelengths, d, s=0.1, c="r", alpha=0.1)

        plt.xlabel("Wavelength(nm)")
        # plt.ylabel("Reflectance")
        plt.title(title)

        # 创建图例
        legend_elements = [
            Patch(facecolor="blue", edgecolor="none", label="Train"),
            Patch(facecolor="red", edgecolor="none", label="Test"),
        ]
        plt.legend(handles=legend_elements)

        plt.savefig(
            f"{save_path}/wave_scatter_{save_name}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    @staticmethod
    def plot_heatmap(data, save_path, save_name, title=None):
        """
        Plot a heatmap for spectral data.

        Args:
            data (array-like): Spectral data array, shape (n_samples, n_wavelengths).
            save_path (str): Directory path to save the plot.
            save_name (str): Filename to save the plot.
            title (str, optional): Title of the plot. Default is None.

        Returns:
            None
        """
        plt.imshow(data, aspect="auto", cmap="viridis")
        # plt.colorbar(label="Reflectance")
        plt.colorbar()
        plt.xlabel("Wavelength Index")
        plt.ylabel("Sample Index")
        plt.title("Spectral Data Heatmap")
        plt.title(title)
        plt.savefig(
            f"{save_path}/heatmap_{save_name}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    @staticmethod
    def plot_true_pred_line(y_true, y_pred, save_path, save_name, title=None):
        """
        Plot line graphs for true vs. predicted values.

        Args:
            y_true (array-like): True values of the target variable.
            y_pred (array-like or list of arrays): Predicted values of the target variable. Can be a single array or a list of arrays.
            save_path (str): Directory path to save the plot.
            save_name (str): Filename to save the plot.
            title (str, optional): Title of the plot. Default is None.

        Returns:
            None
        """
        plt.figure(figsize=(10, 5))
        plt.plot(y_true, label="True")
        if isinstance(y_pred, list):
            for idx, y_p in enumerate(y_pred):
                plt.plot(y_p, label=f"Predicted_{idx+1}", alpha=0.8, linewidth=0.8)
        else:
            plt.plot(y_pred, label="Predicted", alpha=0.5)

        plt.legend()
        plt.xlabel("Sample Index")
        plt.ylabel("Value")
        plt.title(title)
        plt.savefig(f"{save_path}/{save_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_true_pred_scatter(
        y_true, y_pred, scores, save_path, save_name, title=None
    ):
        """
        Plot scatter graphs for true vs. predicted values with regression metrics.

        Args:
            y_true (array-like): True values of the target variable.
            y_pred (array-like): Predicted values of the target variable.
            scores (dict): Dictionary of regression metrics (RMSE, MAE, R2).
            save_path (str): Directory path to save the plot.
            save_name (str): Filename to save the plot.
            title (str, optional): Title of the plot. Default is None.

        Returns:
            None
        """
        x_min = y_true.min()
        x_max = y_true.max()

        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred)

        ideal = np.linspace(x_min, x_max, 2)
        ax.plot(ideal, ideal, color="r", label="Ideal Fit")

        z = np.polyfit(y_true, y_pred, 5)
        fit_line = np.polyval(z, [x_min, x_max])
        ax.plot([x_min, x_max], fit_line, color="blue", label="Model Fit")

        score_text = f"RMSE = {scores['rmse']:.4f}, MAE = {scores['mae']:.4f}, $R^2$ = {scores['r2']:.4f}"
        ax.text(0.05, 0.9, score_text, transform=ax.transAxes)

        ax.set_xlabel("Measured Value")
        ax.set_ylabel("Predicted Value")
        plt.title(title)
        ax.legend()

        plt.savefig(f"{save_path}/{save_name}.png", dpi=300, bbox_inches="tight")
        plt.close()
