import copy

import numpy as np
import pandas as pd
import pywt
from obspy.signal.detrend import polynomial
from pybaselines.whittaker import airpls, derpsalsa, iarpls
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from scipy.stats import f
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict


class Preprocess:
    """
    A collection of preprocessing methods for spectral data.
    """

    @staticmethod
    def mean_centering(data, axis=None):
        """
        Mean centers the given 2D array such that each element is
        subtracted by the mean of the specified axis.

        Args:
            data (numpy.ndarray): The input data array to be mean centered
                with shape (n_samples, n_features).
            axis (int, optional): The axis along which to compute the mean.
                If None, mean centering is performed element-wise.
                If 0, mean centering is column-wise.
                If 1, mean centering is row-wise. Defaults to None.

        Raises:
            ValueError: If the axis is not one of the following values: None, 0, 1.

        Returns:
            numpy.ndarray: The mean centered data array with the same shape
            as the input data.
        """
        if axis not in [None, 0, 1]:
            raise ValueError("Unexpected axis value.")

        data_mean = np.mean(data, axis=axis)

        if axis != None:
            data_mean = np.expand_dims(data_mean, axis=axis)

        return data - data_mean

    @staticmethod
    def normalization(data, axis=None):
        """
        Normalizes the given 2D array such that each element is scaled
        to the range [0, 1].

        Args:
            data (numpy.ndarray): The input data array to be normalized
                with shape (n_samples, n_features).
            axis (int, optional): The axis along which to compute the min
                and max values.
                If None, normalization is performed element-wise.
                If 0, normalization is column-wise.
                If 1, normalization is row-wise. Defaults to None.

        Raises:
            ValueError: If the axis is not one of the following values: None, 0, 1.

        Returns:
            numpy.ndarray: The normalized data array with the same shape
            as the input data.
        """
        if axis not in [None, 0, 1]:
            raise ValueError("Unexpected axis value.")

        data_min = np.min(data, axis=axis)
        data_max = np.max(data, axis=axis)

        if axis != None:
            data_min = np.expand_dims(data_min, axis=axis)
            data_max = np.expand_dims(data_max, axis=axis)

        return (data - data_min) / (data_max - data_min)

    @staticmethod
    def standardization(data, axis=None):
        """
        Standardizes the given 2D array such that each element is
        subtracted by the mean and divided by the standard deviation.

        Args:
            data (numpy.ndarray): The input data array to be standardized
                with shape (n_samples, n_features).
            axis (int, optional): The axis along which to compute the mean
                and standard deviation.
                If None, standardization is performed element-wise.
                If 0, standardization is column-wise.
                If 1, standardization is row-wise. Defaults to None.

        Raises:
            ValueError: If the axis is not one of the following values: None, 0, 1.

        Returns:
            numpy.ndarray: The standardized data array with the same shape
            as the input data.
        """

        if axis not in [None, 0, 1]:
            raise ValueError("Unexpected axis value.")

        data_mean = np.mean(data, axis=axis)
        data_std = np.std(data, axis=axis)

        if axis != None:
            data_mean = np.expand_dims(data_mean, axis=axis)
            data_std = np.expand_dims(data_std, axis=axis)

        return (data - data_mean) / data_std

    @staticmethod
    def snv(data):
        """
        Standard Normal Variate (SNV) transformation.

        Args:
            data (numpy.ndarray): The input data array with shape (n_samples, n_features).

        Returns:
            numpy.ndarray: The data array after SNV transformation, where
            each row is standardized by its own mean and standard deviation.
        """
        return Preprocess.standardization(data, axis=1)

    @staticmethod
    def sg(data, window_len=11, poly_order=1, deriv=1):
        """
        Applies the Savitzky-Golay filter to smooth and differentiate data.

        Args:
            data (numpy.ndarray): The input data array to be smoothed and differentiated, with
                shape (n_samples, n_features).
            window_len (int, optional): The length of the filter window (i.e., the number of coefficients).
                Must be a positive odd integer. Defaults to 11.
            poly_order (int, optional): The order of the polynomial used to fit the samples.
                Must be less than window_len. Defaults to 1.
            deriv (int, optional): The order of the derivative to compute.
                0 means only smoothing, 1 means the first derivative, etc. Defaults to 1.

        Returns:
            numpy.ndarray: The smoothed and/or differentiated data.
        """
        return savgol_filter(data, window_len, poly_order, deriv)

    @staticmethod
    def msc(data, mean_center=True, reference=None):
        """
        Applies Multiplicative Scatter Correction (MSC) to the given data.

        Args:
            data (numpy.ndarray): The input data array to be corrected, with shape (n_samples, n_features).
            mean_center (bool, optional): Whether to mean center the data before applying MSC.
                Defaults to True.
            reference (numpy.ndarray, optional): The reference spectrum to use for MSC.
                If None, the mean spectrum of the data is used. Defaults to None.

        Returns:
            numpy.ndarray: The corrected data array with the same shape as the input data.
        """
        if mean_center:
            data = Preprocess.mean_centering(data, 1)

        data_msc = np.empty(data.shape)
        data_ref = np.mean(data, 0) if reference is None else reference

        for i in range(data.shape[0]):
            a, b = np.polyfit(data_ref, data[i], 1)
            data_msc[i] = (data[i] - b) / a

        return data_msc

    @staticmethod
    def poly_detrend(data, poly_order=2):
        """
        Applies polynomial detrending to the given data.

        Args:
            data (numpy.ndarray): The input data array to be detrended,
                with shape (n_samples, n_features).
            poly_order (int, optional): The order of the polynomial used for detrending.
                Defaults to 2.

        Returns:
            numpy.ndarray: The detrended data array with the same shape as the input data.
        """
        data_det = np.zeros_like(data)
        for i in range(data.shape[0]):
            data_det[i] = polynomial(data[i].copy(), order=poly_order, plot=False)

        return data_det

    @staticmethod
    def rnv(data, percent=25):
        """
        Applies Robust Normal Variate (RNV) transformation to the given data.

        Args:
            data (numpy.ndarray or list): The input data array to be transformed,
                with shape (n_samples, n_features).
            percent (int): The percentile value for the transformation,
                typically between 0 and 100. Defaults to 25.

        Returns:
            numpy.ndarray: The RNV-transformed data array with the same shape as the input data.

        Raises:
            AssertionError: If the input data is not of type numpy.ndarray or list.
        """

        assert isinstance(data, np.ndarray) or isinstance(
            data, list
        ), "Variable X is of wrong type, must be ndarray or list"

        if isinstance(data, list):
            data = np.array(data)

        data_rnv = np.zeros_like(data)

        if data.ndim == 2:
            for i in range(data.shape[0]):
                percentile_value = np.percentile(
                    data[i], percent, method="median_unbiased"
                )
                data_rnv[i] = (data[i] - percentile_value) / np.std(
                    data[i][data[i] <= percentile_value]
                )
        elif data.ndim == 1:
            percentile_value = np.percentile(data, percent, method="median_unbiased")
            data_rnv = (data - percentile_value) / np.std(
                data[data <= percentile_value]
            )
        else:
            assert False, "Variable X dimension error"

        return data_rnv

    @staticmethod
    def d1(data):
        """
        Computes the first derivative of the given data.

        Args:
            data (numpy.ndarray): The input raw spectrum data,
                with shape (n_samples, n_features).

        Returns:
            numpy.ndarray: The first derivative of the data,
                with shape (n_samples, n_features-1).
        """
        n, p = data.shape
        spec_D1 = np.ones((n, p - 1))
        for i in range(n):
            spec_D1[i] = np.diff(data[i])
        return spec_D1

    @staticmethod
    def d2(data):
        """
        Computes the second derivative of the given data.

        Args:
            data (numpy.ndarray or pandas.DataFrame): The input raw spectrum data,
                with shape (n_samples, n_features).

        Returns:
            numpy.ndarray: The second derivative of the data,
                with shape (n_samples, n_features-2).
        """
        data = copy.deepcopy(data)
        if isinstance(data, pd.DataFrame):
            data = data.values
        temp2 = (pd.DataFrame(data)).diff(axis=1)
        temp3 = np.delete(temp2.values, 0, axis=1)
        temp4 = (pd.DataFrame(temp3)).diff(axis=1)
        spec_D2 = np.delete(temp4.values, 0, axis=1)
        return spec_D2

    @staticmethod
    def move_avg(data, window_size=11):
        """
        Applies moving average filtering to the given data.

        Args:
            data (numpy.ndarray): The input raw spectrum data,
                with shape (n_samples, n_features).
            window_size (int, optional): The size of the moving window,
                must be an odd number. Defaults to 11.

        Returns:
            numpy.ndarray: The data after applying moving average filtering.
        """
        data_ma = copy.deepcopy(data)
        for i in range(data.shape[0]):
            out0 = (
                np.convolve(data_ma[i], np.ones(window_size, dtype=int), "valid")
                / window_size
            )
            r = np.arange(1, window_size - 1, 2)
            start = np.cumsum(data_ma[i, : window_size - 1])[::2] / r
            stop = (np.cumsum(data_ma[i, :-window_size:-1])[::2] / r)[::-1]
            data_ma[i] = np.concatenate((start, out0, stop))

        return data_ma

    def gaussian_smoothing(data, sigma=1.0):
        """
        Smooth a signal using Gaussian filter.

        Args:
            signal (numpy.ndarray): The input signal to be smoothed.
            sigma (float, optional): The standard deviation of the Gaussian kernel. Default is 1.0.

        Returns:
            numpy.ndarray: The smoothed signal.
        """
        smoothed_signal = gaussian_filter1d(data, sigma)
        return smoothed_signal

    @staticmethod
    def wavelet_denoising(data, wavelet='db1', level=None):
        """
        Denoise a signal using wavelet transform.

        Args:
            data (numpy.ndarray): The input signal to be denoised.
            wavelet (str, optional): The type of wavelet to use. Default is 'db1'.
            level (int, optional): The level of decomposition to perform. If None, the level is calculated based on the length of the data.

        Returns:
            numpy.ndarray: The denoised signal.
        """
        # Perform wavelet decomposition
        coeff = pywt.wavedec(data, wavelet, mode='per', level=level)

        # Calculate the universal threshold
        sigma = np.median(np.abs(coeff[-1])) / 0.6745
        uthresh = sigma * np.sqrt(2 * np.log(len(data)))

        # Apply soft thresholding to detail coefficients
        coeff[1:] = [pywt.threshold(i, value=uthresh, mode='soft') for i in coeff[1:]]

        # Reconstruct the signal using the thresholded coefficients
        return pywt.waverec(coeff, wavelet, mode='per')
    
    # Baseline correction =====================================================

    @staticmethod
    def baseline_iarpls(data, lam=1000):
        """
        Applies the Iterative Asymmetric Reweighted Penalized Least Squares (IARPLS)
        baseline correction method to the given data.

        Args:
            data (numpy.ndarray): The input data array to be baseline corrected,
                with shape (n_samples, n_features).
            lam (int, optional): The smoothing parameter for the IARPLS method.
                Defaults to 1000.

        Returns:
            numpy.ndarray: The baseline-corrected data array with the same shape
            as the input data.
        """
        data_iarpls = copy.deepcopy(data)
        for k in range(data.shape[0]):
            baseline, _ = iarpls(data_iarpls[k, :], lam)
            data_iarpls[k, :] = data_iarpls[k, :] - baseline

        return data_iarpls

    @staticmethod
    def baseline_airpls(data, lam=1000):
        """
        Applies the Asymmetric Reweighted Penalized Least Squares (ARPLS)
        baseline correction method to the given data.

        Args:
            data (numpy.ndarray): The input data array to be baseline corrected,
                with shape (n_samples, n_features).
            lam (int, optional): The smoothing parameter for the ARPLS method.
                Defaults to 1000.

        Returns:
            numpy.ndarray: The baseline-corrected data array with the same shape
            as the input data.
        """
        data_airpls = copy.deepcopy(data)
        for k in range(data.shape[0]):
            baseline, _ = airpls(data_airpls[k, :], lam)
            data_airpls[k, :] = data_airpls[k, :] - baseline

        return data_airpls

    @staticmethod
    def baseline_derpsalsa(data, lam=1000):
        """
        Applies the Derivative Penalized Signal Adaptive Least Squares (DERPSALSA)
        baseline correction method to the given data.

        Args:
            data (numpy.ndarray): The input data array to be baseline corrected,
                with shape (n_samples, n_features).
            lam (int, optional): The smoothing parameter for the DERPSALSA method.
                Defaults to 1000.

        Returns:
            numpy.ndarray: The baseline-corrected data array with the same shape
            as the input data.
        """
        data_derpsalsa = copy.deepcopy(data)
        for k in range(data.shape[0]):
            baseline, _ = derpsalsa(data_derpsalsa[k, :], lam)
            data_derpsalsa[k, :] = data_derpsalsa[k, :] - baseline

        return data_derpsalsa

    # Outlier remove =====================================================

    @staticmethod
    def mahalanobis(mahal_X, y, threshold=95):
        """
        Applies the Mahalanobis distance method to identify and remove outliers from the input data.

        Args:
            mahal_X (numpy.ndarray): The input feature matrix.
            y (numpy.ndarray): The corresponding labels or targets.
            threshold (float): The percentile threshold to identify outliers. Default is 95.

        Returns:
            numpy.ndarray: The feature matrix with outliers removed.
            numpy.ndarray: The corresponding labels or targets with outliers removed.
        """
        mahal_X = np.asarray(mahal_X)
        x_mu = mahal_X - np.mean(mahal_X, axis=0)
        cov = np.cov(mahal_X.T)
        inv_covmat = np.linalg.inv(cov)
        left_term = np.dot(x_mu, inv_covmat)
        mahal = np.dot(left_term, x_mu.T)
        d = mahal.diagonal()
        threshold = np.percentile(d, threshold)
        mahal_x = mahal_X[d < threshold]

        return mahal_x, y[d < threshold]

    @staticmethod
    def remove_outliers_pls(x_, y_, max_outliers=10):
        """
        Removes outliers from the input data using Partial Least Squares (PLS) regression.

        Args:
            x_ (numpy.ndarray): The input feature matrix.
            y_ (numpy.ndarray): The corresponding labels or targets.
            max_outliers (int): The maximum number of outliers to remove. Default is 10.

        Returns:
            numpy.ndarray: The feature matrix with outliers removed.
            numpy.ndarray: The corresponding labels or targets with outliers removed.
        """

        def optimize_pls_components(x, y, cv):
            """
            Optimizes the number of PLS components using cross-validation.

            Args:
                x (numpy.ndarray): The input feature matrix.
                y (numpy.ndarray): The corresponding labels or targets.
                cv (KFold): The cross-validation strategy.

            Returns:
                tuple: A tuple containing arrays of MAE, MSE, RMSE, R2, and correlation coefficients for each number of components.
            """

            mae, mse, rmse, r2, r = [], [], [], [], []

            for n_components in range(1, 17):
                model = PLSRegression(n_components=n_components)
                y_pred = cross_val_predict(model, x, y, cv=cv)

                mae.append(mean_absolute_error(y, y_pred))
                mse.append(mean_squared_error(y, y_pred, squared=True))
                rmse.append(mean_squared_error(y, y_pred, squared=False))
                r2.append(r2_score(y, y_pred))
                r.append(np.corrcoef(y, y_pred)[0, 1])

            return (
                np.array(mae),
                np.array(mse),
                np.array(rmse),
                np.array(r2),
                np.array(r),
            )

        cv = KFold(3, shuffle=True, random_state=42)
        mae, mse, rmse, r2, r = optimize_pls_components(x_, y_, cv)
        ncomp = np.argmin(rmse) + 1
        pls = PLSRegression(n_components=ncomp)
        pls.fit(x_, y_)
        T = pls.x_scores_
        P = pls.x_loadings_
        Err = x_ - np.dot(T, P.T)
        Q = np.sum(Err**2, axis=1)
        Tsq = np.sum((pls.x_scores_ / np.std(pls.x_scores_, axis=0)) ** 2, axis=1)

        # set the confidence level
        conf = 0.95
        # Calculate confidence level for T-squared from the ppf of the F distribution
        Tsq_conf = (
            f.ppf(q=conf, dfn=ncomp, dfd=(x_.shape[0] - ncomp))
            * ncomp
            * (x_.shape[0] - 1)
            / (x_.shape[0] - ncomp)
        )

        # Estimate the confidence level for the Q-residuals
        i = np.max(Q) + 1
        while 1 - np.sum(Q > i) / np.sum(Q > 0) > conf:
            i -= 1
        Q_conf = i

        # Sort the RMS distance from the origin in descending order (largest first)
        rms_dist = np.flip(np.argsort(np.sqrt(Q**2 + Tsq**2)), axis=0)
        Xc = x_[rms_dist, :]
        Yc = y_[rms_dist]

        mse = np.zeros(max_outliers)
        for j in range(max_outliers):
            pls = PLSRegression(n_components=ncomp)
            pls.fit(Xc[j:, :], Yc[j:])
            y_cv = cross_val_predict(pls, Xc[j:, :], Yc[j:], cv=3)
            mse[j] = mean_squared_error(Yc[j:], y_cv)

        msemin = np.where(mse == np.min(mse[np.nonzero(mse)]))[0][0]
        print("Remove %d outliers" % (msemin + 1))

        outlier_i = rms_dist[0 : msemin + 1]

        spectrum_select, y_select = np.delete(x_, outlier_i, axis=0), np.delete(
            y_, outlier_i, axis=0
        )
        return spectrum_select, y_select
