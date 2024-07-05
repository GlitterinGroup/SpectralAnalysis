import copy

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (SelectFromModel, SelectKBest, f_classif,
                                       f_regression, mutual_info_classif,
                                       mutual_info_regression, r_regression)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_score
from tqdm import tqdm

from spectral_analysis.model.pls import PLSRegression
from spectral_analysis.preprocess import Preprocess


class FeatureSelection:
    """
    A class containing various methods for feature selection on datasets.
    """

    @staticmethod
    def pca(data, n_features=100):
        """
        Perform Principal Component Analysis (PCA) on the data.

        Args:
            data (numpy.ndarray): Input data array, with shape (n_samples, n_features).
            n_features (int, optional): Number of principal features to select.
                                        Default is 100.

        Returns:
            tuple:
                - numpy.ndarray: Selected features matrix, shape (n_samples, selected_n_features).
                - numpy.ndarray: Indices of selected features in the original feature set.
        """
        pca_model = PCA(n_components=n_features)
        pca_model.fit(data)

        loadings = pca_model.components_
        importance = np.sum(np.abs(loadings), axis=0)
        i_select = np.argsort(importance)[-n_features:]
        return data[:, i_select], i_select

    @staticmethod
    def spa(data, n_features=100, method=0):
        """
        Perform feature selection using the Successive Projections Algorithm (SPA).

        Args:
            data (numpy.ndarray): Input data array, with shape (n_samples, n_features).
            n_features (int, optional): Number of features to select. Default is 100.
            method (int, optional): Method for SPA (0 or 1). Method 1 is slower due to matrix inversion. Default is 0.

        Returns:
            tuple:
                - numpy.ndarray: Selected features matrix, shape (n_samples, selected_n_features).
                - numpy.ndarray: Indices of selected features in the original feature set.

        References:
            https://www.sciencedirect.com/science/article/pii/S0169743901001198
            https://www.sciencedirect.com/science/article/pii/S0165993612002750
        """

        m, n = data.shape
        i_init = np.random.choice(n)

        if not method in [0, 1]:
            raise ValueError("Unexpected method value.")

        data_copy = np.array(data)
        data_orth = [data[:, i_init]]
        i_select = [i_init]
        i_remain = [i for i in range(n) if i != i_init]

        for _ in range(n_features - 1):
            n_max = -1
            i_max = -1
            data_trans = np.array(data_orth).T

            for j in i_remain:
                if method == 0:
                    xi = data_trans[:, -1]
                    xj = data_copy[:, j]
                    proj = xj - xi * np.dot(xi, xj) / np.dot(xi, xi)
                    norm = np.linalg.norm(proj)
                else:
                    data = data_copy[:, j]
                    proj = (
                        np.identity(m)
                        - data_trans
                        @ np.linalg.inv(data_trans.T @ data_trans)
                        @ data_trans.T
                    ) @ data
                    norm = data.T @ proj

                data_copy[:, j] = proj
                if norm > n_max:
                    n_max, i_max = norm, j

            data_orth.append(data_copy[:, i_max])

            i_select.append(i_max)
            i_remain.remove(i_max)

        i_select.sort()
        return data[:, i_select], i_select

    @staticmethod
    def select_from_model(data, target):
        """
        Perform feature selection using a model-based approach with RandomForestRegressor.

        Args:
            data (numpy.ndarray): Input data array, with shape (n_samples, n_features).
            target (numpy.ndarray): Target data array, with shape (n_samples, ).

        Returns:
            tuple:
                - numpy.ndarray: Data array with selected features.
                - numpy.ndarray: Indices of selected features in the original feature set.
        """
        selector = SelectFromModel(estimator=RandomForestRegressor()).fit(data, target)
        i_select = selector.get_support(indices=True)
        return data[:, i_select], i_select

    @staticmethod
    def mrmr(data, target, n_features=100, task_="regression"):
        """
        Perform Minimum Redundancy Maximum Relevance (mRMR) feature selection.

        Args:
            data (numpy.ndarray): Input data array, with shape (n_samples, n_features).
            target (numpy.ndarray): Target data array, with shape (n_samples, ).
            n_features (int, optional): Number of features to select. Default is 100.
            task_ (str, optional): Task type ('regression' or 'classification'). Default is 'regression'.

        Returns:
            tuple:
                - numpy.ndarray: Selected features matrix, shape (n_samples, selected_n_features).
                - numpy.ndarray: Indices of selected features in the original feature set.
        """
        if task_ == "regression":
            mi = mutual_info_regression(data, target)
        else:
            mi = mutual_info_classif(data, target)

        i_select = np.argsort(mi)[-n_features:]

        return data[:, i_select], i_select

    @staticmethod
    def uve(data, target, n_components=50, cv=10):
        """
        Perform Uninformative Variable Elimination (UVE).

        Args:
            data (numpy.ndarray): Input spectral data array, with shape (n_samples, n_features).
            target (numpy.ndarray): Target data array, with shape (n_samples, ).
            n_components (int, optional): Number of components to use in the PLS model. Default is 50.
            cv (int, optional): Number of cross-validation folds. Default is 10.

        Returns:
            tuple:
                - numpy.ndarray: Data array with selected features (half of the original features).
                - numpy.ndarray: Indices of selected features in the original feature set.
        """

        def optimize_pls_components(X, Y, max_components, cv):
            """
            Optimize the number of PLS components using cross-validation.

            Args:
                X (numpy.ndarray): Input data array.
                Y (numpy.ndarray): Target data array.
                max_components (int): Maximum number of components to test.
                cv (int): Number of cross-validation folds.

            Returns:
                int: Optimal number of PLS components.
            """
            best_r2 = -np.inf
            best_n_components = 0

            for n in range(2, min(max_components, X.shape[1]) + 1):
                pls = PLSRegression(n_components=n)
                r2 = cross_val_score(pls, X, Y, cv=cv, scoring="r2").mean()
                if r2 > best_r2:
                    best_r2 = r2
                    best_n_components = n

            return best_n_components

        N, D = data.shape
        optimal_components = optimize_pls_components(data, target, n_components, cv)

        pls = PLSRegression(n_components=optimal_components)
        pls.fit(data, target)
        original_coefs = np.abs(pls.coef_).reshape(-1)

        # Calculate stability for each variable through cross-validation
        stability_scores = np.zeros(D)
        for i in tqdm(range(D), desc="uve: "):
            # Exclude the current variable from the dataset
            data_reduced = np.delete(data, i, axis=1)
            pls_cv = PLSRegression(n_components=optimal_components)
            cv_scores = cross_val_score(
                pls_cv, data_reduced, target, cv=cv, scoring="neg_mean_squared_error"
            )
            stability_scores[i] = np.mean(np.abs(cv_scores))

        # Calculate the informativeness score for each variable
        informative_scores = stability_scores * original_coefs
        threshold = np.median(informative_scores)
        i_select = np.where(informative_scores >= threshold)[0]
        return data[:, i_select], i_select

    @staticmethod
    def cars(
        data, target, n_features=100, n_sample_runs=50, pls_components=20, n_cv_folds=5
    ):
        """
        Perform Competitive Adaptive Reweighted Sampling (CARS).

        Args:
            data (numpy.ndarray): Input spectral data array, with shape (n_samples, n_features).
            target (numpy.ndarray): Target data array, with shape (n_samples, ).
            n_features (int, optional): Number of features to select. Default is 100.
            n_sample_runs (int, optional): Number of sampling runs. Default is 50.
            pls_components (int, optional): Number of PLS components. Default is 20.
            n_cv_folds (int, optional): Number of cross-validation folds. Default is 5.

        Returns:
            tuple:
                - numpy.ndarray: Selected features matrix, shape (n_samples, selected_n_features).
                - numpy.ndarray: Indices of selected features in the original feature set.
        """

        def _pc_cross_validation(data, target, pc, cv):
            kf = KFold(n_splits=cv, shuffle=True)
            RMSECV = []
            for i in range(pc):
                RMSE = []
                for train_index, test_index in kf.split(data):
                    x_train, x_test = data[train_index], data[test_index]
                    y_train, y_test = target[train_index], target[test_index]
                    pls = PLSRegression(n_components=i + 1)
                    pls.fit(x_train, y_train)
                    y_predict = pls.predict(x_test)
                    RMSE.append(np.sqrt(mean_squared_error(y_test, y_predict)))
                RMSE_mean = np.mean(RMSE)
                RMSECV.append(RMSE_mean)
            rindex = np.argmin(RMSECV)
            RMSE_mean_min = RMSECV[rindex]
            return RMSECV, rindex, RMSE_mean_min

        samples_ratio = 0.8
        n_samples, n_wavelengths = data.shape

        # prepare for edf_schedule
        u = np.power((n_wavelengths / 2), (1 / (n_sample_runs - 1)))
        k = (1 / (n_sample_runs - 1)) * np.log(n_wavelengths / 2)

        n_fit_samples = np.round(n_samples * samples_ratio)
        b2 = np.arange(n_wavelengths)
        x_copy = copy.deepcopy(data)
        idx_with_x = np.vstack((np.array(b2).reshape(1, -1), data))

        wave_data = []
        wave_num_list = []
        RMSECV = []
        selection_ratios = []
        for i in range(1, n_sample_runs + 1):

            # edf schedule
            selection_ratios.append(u * np.exp(-1 * k * i))
            wave_num = int(np.round(selection_ratios[i - 1] * n_wavelengths))
            wave_num_list = np.hstack((wave_num_list, wave_num))

            fitting_samples_index = np.random.choice(
                np.arange(n_samples), size=int(n_fit_samples), replace=False
            )
            wavelength_index = b2[0:wave_num].reshape(1, -1)[0]

            x_pls_fit = x_copy[
                np.ix_(list(fitting_samples_index), list(wavelength_index))
            ]
            y_pls_fit = target[fitting_samples_index]
            x_copy = x_copy[:, wavelength_index]

            idx_with_x = idx_with_x[:, wavelength_index]
            d = idx_with_x[0, :].reshape(1, -1)

            if (n_wavelengths - wave_num) > 0:
                d = np.hstack((d, np.full((1, (n_wavelengths - wave_num)), -1)))

            if len(wave_data) == 0:
                wave_data = d
            else:
                wave_data = np.vstack((wave_data, d.reshape(1, -1)))

            if wave_num < pls_components:
                pls_components = wave_num

            pls = PLSRegression(n_components=pls_components)
            pls.fit(x_pls_fit, y_pls_fit)
            beta = pls.coef_
            b = np.abs(beta)
            b2 = np.argsort(-b).squeeze()

            _, rindex, rmse_min = _pc_cross_validation(
                x_pls_fit, y_pls_fit, pls_components, n_cv_folds
            )
            RMSECV.append(rmse_min)

        wavelengths_set = []
        for i in range(wave_data.shape[0]):
            wd = wave_data[i, :]
            wd_ones = np.ones((len(wd)))
            for j in range(len(wd)):
                ind = np.where(wd == j)
                if len(ind[0]) == 0:
                    wd_ones[j] = 0
                else:
                    wd_ones[j] = wd[ind[0]]
            if len(wavelengths_set) == 0:
                wavelengths_set = copy.deepcopy(wd_ones)
            else:
                wavelengths_set = np.vstack((wavelengths_set, wd_ones.reshape(1, -1)))

        min_idx = np.argmin(RMSECV)
        optimal = wavelengths_set[min_idx, :]
        i_select = np.argsort(-optimal)[:n_features]
        return data[:, i_select], i_select

    @staticmethod
    def corr_coefficient(data, target, threshold=0.5):
        """
        Perform feature selection based on correlation coefficient.

        Args:
            data (numpy.ndarray): Input data array, with shape (n_samples, n_features).
            target (numpy.ndarray): Target data array, with shape (n_samples, ).
            threshold (float, optional): Threshold for selecting features based on absolute correlation coefficient. Default is 0.5.

        Returns:
            tuple:
                - numpy.ndarray: Data array with selected features.
                - numpy.ndarray: Indices of selected features in the original feature set.
        """
        coef = r_regression(data, target)
        i_select = np.where(np.abs(coef) > threshold)[0]
        return data[:, i_select], i_select

    @staticmethod
    def anova(data, target, task_="regression", threshold=0.5):
        """
        Perform feature selection using ANOVA F-test.

        Args:
            data (numpy.ndarray): Input data array, with shape (n_samples, n_features).
            target (numpy.ndarray): Target data array, with shape (n_samples, ).
            task_ (str, optional): Task type ('regression' or 'classification'). Default is 'regression'.
            threshold (float, optional): Threshold for selecting features based on normalized ANOVA F-test scores. Default is 0.5.

        Returns:
            tuple:
                - numpy.ndarray: Data array with selected features.
                - numpy.ndarray: Indices of selected features in the original feature set.
        """
        if task_ == "regression":
            fs = SelectKBest(score_func=f_regression, k=data.shape[1])
            fit = fs.fit(data, target)
        elif task_ == "classification":
            fs = SelectKBest(score_func=f_classif, k=data.shape[1])
            fit = fs.fit(data, target)

        i_select = np.where(Preprocess.normalization(np.abs(fit.scores_)) > threshold)[
            0
        ]
        return data[:, i_select], i_select