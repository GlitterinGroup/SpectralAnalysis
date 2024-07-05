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
    def plot_wavelength_line(
        wavelengths, selected_idx, data, save_path, save_name, title=None
    ):
        """
        Plot line graphs for spectral data.

        Args:
            wavelengths (array-like): Array of wavelength values.
            selected_idx (array-like): Indices of selected wavelengths to highlight.
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

        if selected_idx is not None:
            for idx in selected_idx:
                plt.axvline(
                    x=wavelengths[idx],
                    color="tab:green",
                    linestyle="-",
                    alpha=0.3,
                )
        plt.xlabel("Wavelength(nm)")
        # plt.ylabel("Reflectance")
        plt.title(title)
        plt.savefig(
            f"{save_path}/wave_line_{save_name}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    @staticmethod
    def plot_wavelength_scatter(
        wavelengths, selected_idx, data, save_path, save_name, title=None
    ):
        """
        Plot scatter graphs for spectral data.

        Args:
            wavelengths (array-like): Array of wavelength values.
            selected_idx (array-like): Indices of selected wavelengths to highlight.
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

        if selected_idx is not None:
            for idx in selected_idx:
                plt.axvline(
                    x=wavelengths[idx],
                    color="tab:green",
                    linestyle="-",
                    alpha=0.3,
                )
        plt.xlabel("Wavelength(nm)")
        # plt.ylabel("Reflectance")
        plt.title(title)
        plt.savefig(
            f"{save_path}/wave_scatter_{save_name}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    @staticmethod
    def plot_train_test_scatter(
        wavelengths,
        train_data,
        test_data,
        save_path,
        save_name,
        title=None,
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
        y_true, y_pred, scores, save_path, save_name, title=None, tolerance=None
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
            tolerance (float, optional): Tolerance range for the prediction. Default is None.

        Returns:
            None
        """
        x_min = y_true.min()
        x_max = y_true.max()

        fig, ax = plt.subplots()
        ax.scatter(y_true, y_pred, alpha=0.6, label="Predicted vs True")

        ideal = np.linspace(x_min, x_max, 2)
        ax.plot(ideal, ideal, color="green", linestyle="--", label="Ideal Fit")

        # z = np.polyfit(y_true, y_pred, 1)
        # fit_line = np.polyval(z, [x_min, x_max])
        # ax.plot([x_min, x_max], fit_line, color="royalblue", label="Model Fit")

        if tolerance is not None:
            offset = tolerance * (x_max - x_min)

            upper_bound = ideal + offset
            lower_bound = ideal - offset
            ax.plot(
                ideal,
                upper_bound,
                color="green",
                linestyle=":",
                alpha=0.5,
                label=f"Upper Bound (+{tolerance*100}%)",
            )
            ax.plot(
                ideal,
                lower_bound,
                color="green",
                linestyle=":",
                alpha=0.5,
                label=f"Lower Bound (-{tolerance*100}%)",
            )
            ax.fill_between(ideal, lower_bound, upper_bound, color="green", alpha=0.1)

        score_text = f"RMSE = {scores['rmse']:.4f}, MAE = {scores['mae']:.4f}, $R^2$ = {scores['r2']:.4f}"
        ax.text(0.05, 0.9, score_text, transform=ax.transAxes)

        ax.set_xlabel("Measured Value")
        ax.set_ylabel("Predicted Value")
        if title:
            plt.title(title)
        ax.legend()

        plt.savefig(f"{save_path}/{save_name}.png", dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_true_pred_scatter_train_test(
        y_train_true,
        y_train_pred,
        y_test_true,
        y_test_pred,
        scores,
        save_path,
        save_name,
        title=None,
        tolerance=None,
    ):
        """
        Plot scatter graphs for true vs. predicted values for both training and test sets with regression metrics.

        Args:
            y_train_true (numpy.ndarray): True values of the training dataset.
            y_train_pred (numpy.ndarray): Predicted values of the training dataset.
            y_test_true (numpy.ndarray): True values of the test dataset.
            y_test_pred (numpy.ndarray): Predicted values of the test dataset.
            scores (dict): Dictionary containing evaluation scores, e.g., {'Rc': 0.85, 'RMSEC': 0.123, 'Rp': 0.78, 'RMSEP': 0.234}.
            save_path (str): Directory path where the plot will be saved.
            save_name (str): Name of the saved plot file (without extension).
            title (str, optional): Title of the plot. Default is None.
            tolerance (float, optional): Tolerance range for the prediction. Default is None.

        Returns:
            None
        """
        fig, ax = plt.subplots()

        ax.scatter(
            y_train_true,
            y_train_pred,
            alpha=0.4,
            label=f"Rc={scores['Rc']:.4f}, RMSEC={scores['RMSEC']:.4f}",
        )

        ax.scatter(
            y_test_true,
            y_test_pred,
            alpha=0.6,
            label=f"Rp={scores['Rp']:.4f}, RMSEP={scores['RMSEP']:.4f}",
        )

        x_min = min(y_train_true.min(), y_test_true.min())
        x_max = max(y_train_true.max(), y_test_true.max())
        ideal = np.linspace(x_min, x_max, 2)
        ax.plot(ideal, ideal, color="green", linestyle="--", label="Ideal Fit")

        if tolerance is not None:
            offset = tolerance * (x_max - x_min)

            upper_bound = ideal + offset
            lower_bound = ideal - offset
            ax.plot(
                ideal,
                upper_bound,
                color="green",
                linestyle=":",
                alpha=0.5,
                # label=f"Upper Bound (+{tolerance*100}%)",
            )
            ax.plot(
                ideal,
                lower_bound,
                color="green",
                linestyle=":",
                alpha=0.5,
                # label=f"Lower Bound (-{tolerance*100}%)",
            )
            ax.fill_between(ideal, lower_bound, upper_bound, color="green", alpha=0.1)

        z_train = np.polyfit(y_train_true.flatten(), y_train_pred.flatten(), 1)
        fit_line_train = np.polyval(z_train, [x_min, x_max])
        ax.plot([x_min, x_max], fit_line_train, color="royalblue", label="Train Fit")

        z_test = np.polyfit(y_test_true.flatten(), y_test_pred.flatten(), 1)
        fit_line_test = np.polyval(z_test, [x_min, x_max])
        ax.plot([x_min, x_max], fit_line_test, color="darkorange", label="Test Fit")

        ax.set_xlabel("Measured Value")
        ax.set_ylabel("Predicted Value")
        if title:
            plt.title(title)
        ax.legend()

        plt.savefig(f"{save_path}/{save_name}.png", dpi=300, bbox_inches="tight")
        plt.close()
