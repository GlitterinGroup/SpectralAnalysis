import importlib
import itertools
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from main.data_adapter import DataAdapter
from spectral_analysis.data_split import DataSplit
from spectral_analysis.draw import Draw
from spectral_analysis.feature_selection import FeatureSelection
from spectral_analysis.metrics import Metrics
from spectral_analysis.preprocess import Preprocess


class MainController:
    """
    Main controller class responsible for orchestrating the entire data processing pipeline.
    """

    def __init__(self, config_name):
        """
        Initializes the MainController with the given configuration file.

        Args:
            config_file (str): Path to the YAML configuration file.
        """
        config_path = os.path.join(
            os.path.dirname(__file__), "config", f"{config_name}.yaml"
        )
        self.config = self._load_config(config_path)
        self.plot_data = self.config.get("plot_data", False)
        self.plot_results = self.config.get("plot_results", False)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.task_name = Path(self.config["data"]["file_name"]).stem
        self.task_type = self.config["data"]["task_type"].lower()
        self.save_dir = Path(
            os.path.join(
                os.path.dirname(__file__),
                f"./result/{config_name}/{self.task_name}/{timestamp}",
            )
        )
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_file):
        """
        Load configuration settings from the YAML file.

        Args:
            config_file (str): Path to the YAML configuration file.

        Returns:
            dict: Configuration settings loaded from the YAML file.
        """
        with open(config_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_data(self):
        """
        Load data from file.

        Returns:
            tuple: Tuple containing features (X) and target (y) data.
        """
        adapter = DataAdapter(self.config["data"]["file_name"])
        X, y = adapter.get_feature_and_target(
            data_sheet=self.config["data"]["data_sheet"],
            data_transpose=self.config["data"].get("data_transpose", False),
            data_start_row=self.config["data"].get("data_start_row", 0),
            target_sheet=self.config["data"]["target_sheet"],
            target_column=self.config["data"]["target_column"],
            target_start_row=self.config["data"].get("target_start_row", 0),
        )

        self.wave_num = X.shape[1]

        if self.plot_data:
            self.draw_data(X, "Raw")

        return X, y

    def draw_data(self, X, stage, selected_idx=None, selected_X=None):
        """
        Draw data visualizations.

        Args:
            X (numpy.ndarray or tuple): Input data. If `stage` indicates a split stage, this should be a tuple of (train_data, test_data).
            stage (str): Stage of data processing.
            selected_idx (list, optional): List of selected feature indices. Defaults to None.
            selected_X (numpy.ndarray, optional): Data after feature selection. Defaults to None.
        """

        def _read_wavelengths(file_path):
            return (
                pd.read_excel(file_path, header=None)
                .to_numpy()
                .astype(np.float64)
                .squeeze()
            )

        def _get_wavelengths(wavelengths_key):
            file_path_map = {
                884: os.path.join(
                    os.path.dirname(__file__), "data/wl_0884.xlsx"
                ),  # [1240, 1700] nm
                1899: os.path.join(
                    os.path.dirname(__file__), "data/wl_1899.xlsx"
                ),  # [866, 2530] nm
            }
            if wavelengths_key not in file_path_map:
                raise ValueError(
                    f"Invalid wavelengths: {wavelengths_key}. Valid wavelengths are 884 and 1899."
                )
            return _read_wavelengths(file_path_map[wavelengths_key])

        stage = stage.lower()
        wavelengths = _get_wavelengths(self.wave_num)
        plot_title = f"{self.task_name} {stage}"

        if stage.startswith("split") or stage.startswith("data"):
            # 数据划分后，以"data"开头的是数据经过多种处理方法后再split，直接split是前面处理方法为一种
            train_data, test_data = X
            if "feature_selection" not in self.config:
                Draw.plot_train_test_scatter(
                    wavelengths,
                    train_data,
                    test_data,
                    self.save_dir,
                    stage,
                    plot_title,
                )
            Draw.plot_heatmap(
                train_data,
                self.save_dir,
                stage + "_train",
                plot_title + "_train",
            )
            Draw.plot_heatmap(
                test_data,
                self.save_dir,
                stage + "_test",
                plot_title + "_test",
            )
        else:
            if selected_X is not None:
                # 进行了特征选择，即stage为feature_selection
                Draw.plot_heatmap(
                    selected_X,
                    self.save_dir,
                    stage,
                    plot_title,
                )
            else:
                Draw.plot_heatmap(
                    X,
                    self.save_dir,
                    stage,
                    plot_title,
                )
            # 暂时不画line
            # Draw.plot_wavelength_line(
            #     wavelengths,
            #     selected_idx,
            #     X,
            #     self.save_dir,
            #     stage,
            #     plot_title,
            # )
            Draw.plot_wavelength_scatter(
                wavelengths,
                selected_idx,
                X,
                self.save_dir,
                stage,
                plot_title,
            )

    def preprocess(self, X, y):
        """
        Preprocesses the input data according to the specified methods in the configuration.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The corresponding labels or targets.

        Returns:
            list: A list of preprocessed feature matrices.
            list: A list of preprocessed target arrays.
        """

        def process_single_method(processed_X, processed_y, method_config):
            method_name = method_config["method"].lower()
            method = getattr(Preprocess, method_name)
            params = method_config.get("params", {})
            param_combinations = self._get_param_combinations(params)

            p_x_list = []
            p_y_list = []
            for i, param in enumerate(param_combinations):
                if isinstance(processed_X, list):
                    processed_X_list, processed_y_list = [], []
                    for j, (p_x, p_y) in enumerate(zip(processed_X, processed_y)):
                        if (
                            method_name == "mahalanobis"
                            or method_name == "remove_outliers_pls"
                        ):
                            pr_x, pr_y = method(p_x, p_y, **param)
                        else:
                            pr_x = method(p_x, **param)
                            pr_y = p_y
                        processed_X_list.append(pr_x)
                        processed_y_list.append(pr_y)
                        if self.plot_data:
                            self.draw_data(
                                pr_x, f"Preprocessed ({method_name}_{i+1}_{j+1})"
                            )
                    p_x_list.extend(processed_X_list)
                    p_y_list.extend(processed_y_list)
                else:
                    if (
                        method_name == "mahalanobis"
                        or method_name == "remove_outliers_pls"
                    ):
                        processed_X, processed_y = method(
                            processed_X, processed_y, **param
                        )
                    else:
                        processed_X = method(processed_X, **param)
                    p_x_list.append(processed_X)
                    p_y_list.append(processed_y)
                    if self.plot_data:
                        self.draw_data(
                            processed_X, f"Preprocessed ({method_name}_{i+1})"
                        )
            return p_x_list, p_y_list

        if "preprocess" in self.config:
            processed_X = X.copy()
            processed_y = y.copy()

            preprocess_methods = self.config["preprocess"]

            for method_config in preprocess_methods:
                p_x, p_y = process_single_method(
                    processed_X, processed_y, method_config
                )
                processed_X = p_x
                processed_y = p_y
            return processed_X, processed_y

        return [X], [y]

    def feature_selection(self, X, y):
        """
        Performs feature selection on the input data according to the specified methods in the configuration.

        Args:
            X (list): The input data.
            y (list): The target data.

        Returns:
            list: A list of selected feature matrices.
            list: A list of corresponding target arrays.
        """
        if "feature_selection" in self.config:
            original_data = X
            selected_X = X
            selected_y = y
            all_selected_idx = []

            feature_selection_params = self.config["feature_selection"]

            for selection_config in feature_selection_params:
                method_name = selection_config["method"].lower()
                method = getattr(FeatureSelection, method_name)
                params = selection_config.get("params", {})
                param_combinations = self._get_param_combinations(params)

                new_selected_X = []
                new_selected_y = []
                new_selected_idx = []

                for param in param_combinations:
                    selected_data = []
                    selected_target = []
                    selected_idx = []

                    for data, target in zip(selected_X, selected_y):
                        if method_name in ["pca", "spa"]:
                            single_selected_data, single_selected_idx = method(
                                data, **param
                            )
                        else:
                            single_selected_data, single_selected_idx = method(
                                data, target, **param
                            )

                        selected_data.append(single_selected_data)
                        selected_target.append(target)

                        if len(all_selected_idx) < len(original_data):
                            selected_idx.append(single_selected_idx)
                        else:
                            previous_indices = all_selected_idx[-len(original_data)]
                            current_selected_indices = [
                                previous_indices[i] for i in single_selected_idx
                            ]
                            selected_idx.append(current_selected_indices)

                    new_selected_X.extend(selected_data)
                    new_selected_y.extend(selected_target)
                    new_selected_idx.extend(selected_idx)

                selected_X = new_selected_X
                selected_y = new_selected_y
                all_selected_idx.extend(new_selected_idx)

                if self.plot_data:
                    for idx, (
                        x,
                        single_selected_data,
                        single_selected_idx,
                    ) in enumerate(
                        zip(
                            itertools.cycle(original_data), selected_X, new_selected_idx
                        )
                    ):
                        self.draw_data(
                            x,
                            f"feature_selection_{method_name}_{idx+1}",
                            single_selected_idx,
                            single_selected_data,
                        )
            return selected_X, selected_y
        return X, y

    def _get_param_combinations(self, config):
        """
        Get combinations of parameters.

        Args:
            config (dict): Dictionary containing parameter configurations.

        Returns:
            list: List of dictionaries representing parameter combinations.
        """
        # 找到所有列表类型的参数
        list_params = {k: v for k, v in config.items() if isinstance(v, list)}
        non_list_params = {k: v for k, v in config.items() if not isinstance(v, list)}

        if list_params:
            # 找到所有参数的组合
            keys, values = zip(*list_params.items())
            combinations = [
                dict(zip(keys, combination))
                for combination in itertools.product(*values)
            ]

            # 将非列表类型的参数添加到每个组合中
            for combination in combinations:
                combination.update(non_list_params)
            return combinations
        else:
            return [config]

    def data_split(self, X, y):
        """
        Split data into train and test sets based on configurations.

        Args:
            X (numpy.ndarray or list): Input feature data.
            y (numpy.ndarray): Target data.

        Returns:
            tuple or tuple of lists: Depending on the number of splits, returns train and test data splits.
        """

        def _split_single_data(X, y, method_name, split_params):
            """
            Perform single data split using specified method.

            Args:
                X (numpy.ndarray): Input feature data.
                y (numpy.ndarray): Target data.
                method_name (str): Name of the splitting method.
                split_params (dict): Parameters for the splitting method.

            Returns:
                list of tuples: List containing tuples of train-test splits.
            """
            split_method = getattr(DataSplit, method_name, None)

            if split_method is None:
                raise ValueError(f"Unsupported data_split type: {method_name}")

            # 调用对应的划分方法并传递参数
            if method_name == "kfold_split":
                return split_method(X, y, **split_params)
            else:
                return [split_method(X, y, **split_params)]

        def _get_data_split_params():
            """
            Get data split method and its parameters from the configuration.

            Returns:
                tuple: Data split method name and its parameters.
            """
            if "data_split" in self.config:
                method_name = self.config["data_split"].lower() + "_split"
                params = self.config.get("split_params", {})
                return method_name, params
            return "random_split", {}

        def _process_splits(splits, idx, i=None):
            """
            Process splits and draw visualizations if specified.

            Args:
                splits (list of tuples): List containing tuples of train-test splits.
                idx (int): Index of the split.
                i (int, optional): Index of the data if X is a list. Defaults to None.
            """
            for split_idx, (X_train, X_test, y_train, y_test) in enumerate(splits):
                if self.plot_data:
                    stage = (
                        f"data_{i+1}_split_{idx+1}"
                        if i is not None
                        else f"split_{idx+1}"
                    )
                    if len(splits) > 1:
                        stage += f"_fold_{split_idx+1}"
                    self.draw_data((X_train, X_test), stage)
                X_train_list_all.append(X_train)
                X_test_list_all.append(X_test)
                y_train_list_all.append(y_train)
                y_test_list_all.append(y_test)

        split_method_name, split_params = _get_data_split_params()
        data_split_param_combinations = self._get_param_combinations(split_params)

        X_train_list_all, X_test_list_all, y_train_list_all, y_test_list_all = (
            [],
            [],
            [],
            [],
        )

        for idx, params in enumerate(data_split_param_combinations):
            for i, (single_X, single_y) in enumerate(zip(X, y)):
                splits = _split_single_data(
                    single_X, single_y, split_method_name, params
                )
                _process_splits(splits, idx, i)

        return X_train_list_all, X_test_list_all, y_train_list_all, y_test_list_all

    def model_training(self, X_train, y_train):
        """
        Train models based on the specified configurations.

        Args:
            X_train (numpy.ndarray or list): Input feature data for training.
            y_train (numpy.ndarray or list): Target data for training.

        Returns:
            object or list: Trained model(s).
        """

        def _get_model(model_name, params):
            """
            Get model class based on its name and parameters.

            Args:
                model_name (str): Name of the model.
                params (dict): Parameters for the model.

            Returns:
                object: Instantiated model object.
            """
            module_name = f"spectral_analysis.model.{model_name}"
            if self.task_type == "regression":
                model_class_name = f"{model_name.upper()}Regression"
            else:
                model_class_name = f"{model_name.upper()}Classifier"

            try:
                module = importlib.import_module(module_name)
                model_class = getattr(module, model_class_name)
                return model_class(**params)
            except (ImportError, AttributeError) as e:
                raise ImportError(f"Could not import module or class: {e}")

        if "model" in self.config:
            model_name = self.config.get("model").lower()
            model_params = self.config.get("model_params", {})
            all_param_combinations = self._get_param_combinations(model_params)

            trained_models = []
            for params in all_param_combinations:
                if isinstance(X_train, list):
                    for single_X_train, single_y_train in zip(X_train, y_train):
                        single_model = _get_model(model_name, params)
                        single_model.train(single_X_train, single_y_train)
                        trained_models.append(single_model)
                else:
                    model = _get_model(model_name, params)
                    model.train(X_train, y_train)
                    trained_models.append(model)
            return trained_models
        return None

    def model_predict(self, models, X_train, X_test):
        """
        Perform predictions using given models on training and test data.

        Args:
            models (list): List of trained models to use for predictions.
            X_train (array-like or list): Training data or list of training data batches.
            X_test (array-like or list): Test data or list of test data batches.

        Returns:
            Tuple[List, List]: Two lists of predictions:
                all_predictions_train (list): List of predictions on training data by each model.
                all_predictions (list): List of predictions on test data by each model.
        """
        if models:
            all_predictions_train = []
            all_predictions = []
            if isinstance(X_test, list):
                for model, x_train in zip(models, itertools.cycle(X_train)):
                    all_predictions_train.append(model.predict(x_train))
                for model, x_test in zip(models, itertools.cycle(X_test)):
                    all_predictions.append(model.predict(x_test))
            else:
                # 仅模型为list参数
                for model in models:
                    all_predictions_train.append(model.predict(X_train))
                    all_predictions.append(model.predict(X_test))
            return all_predictions_train, all_predictions
        return None, None

    def get_scores(self, y_true_train, y_pred_train, y_true, y_pred):
        """
        Calculate evaluation scores.

        Args:
            y_true_train (numpy.ndarray or list): True target values for training data.
            y_pred_train (numpy.ndarray or list): Predicted values for training data.
            y_true (numpy.ndarray or list): True target values for test data.
            y_pred (numpy.ndarray or list): Predicted values for test data.

        Returns:
            List[Dict]: List of dictionaries containing evaluation scores for each corresponding pair of `y_true` and `y_pred`.
        """
        all_scores = []
        if isinstance(y_true, list):
            for y_t_tr, y_p_tr, y_t, y_p in zip(
                itertools.cycle(y_true_train),
                y_pred_train,
                itertools.cycle(y_true),
                y_pred,
            ):
                scores = Metrics.calculate_metrics(y_t_tr, y_p_tr, y_t, y_p)
                all_scores.append(scores)
        else:
            for y_p_tr, y_p in zip(y_pred_train, y_pred):
                scores = Metrics.calculate_metrics(y_true_train, y_p_tr, y_true, y_p)
                all_scores.append(scores)
        return all_scores

    def save_results(self, y_true, y_pred, scores):
        """
        Save the prediction results along with evaluation scores to a CSV file.

        Args:
            y_true (numpy.ndarray or list): True target values.
            y_pred (numpy.ndarray or list): Predicted values.
            scores (dict or list of dicts): Evaluation scores.

        Returns:
            str: Path to the saved result file.
        """
        results_path = os.path.join(self.save_dir, "result.csv")

        if not all(isinstance(y_t, np.ndarray) for y_t in y_true):
            y_true = [y_true]

        all_results = []
        for idx, (single_y_pred, single_y_true) in enumerate(
            zip(y_pred, itertools.cycle(y_true))
        ):
            # 创建每个模型的 DataFrame
            single_result_df = pd.DataFrame(
                {
                    "Actual": single_y_true,
                    "Predicted": single_y_pred,
                    "Param": idx + 1,
                }
            )

            # 将评分指标作为单独的行保存
            for metric, value in scores[idx].items():
                score_df = pd.DataFrame(
                    {"Param": [idx + 1], "Metrics": [metric], "Value": [value]}
                )
                single_result_df = pd.concat(
                    [single_result_df, score_df], ignore_index=True
                )
            all_results.append(single_result_df)

        # 合并所有结果
        results_df = pd.concat(all_results, ignore_index=True)
        results_df["Param"] = results_df["Param"].astype(int)

        # 将 config 信息存储在结果文件的最后一行的一个单元格中
        config_str = str(self.config)
        last_row_index = results_df.index.max() + 1
        results_df.loc[last_row_index, "Config"] = config_str

        # 保存结果
        results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
        return results_path

    def show_results(self, result_path):
        """
        Load and display the saved results from a CSV file.

        Args:
            result_path (str): Path to the result CSV file.

        Returns:
            pandas.DataFrame: DataFrame containing the results.
        """
        df = pd.read_csv(result_path)
        grouped = df.groupby(["Param", "Metrics"])

        results_df = pd.DataFrame(columns=["Param", "Metrics", "Value"])
        for group_name, group_df in grouped:
            param, metric = group_name
            value = group_df["Value"].values[0]

            results_df = pd.concat(
                [
                    results_df,
                    pd.DataFrame(
                        {"Param": [param], "Metrics": [metric], "Value": [value]}
                    ),
                ],
                ignore_index=True,
            )

        # 使用Pandas的pivot函数重新排列数据，以便每个模型的指标在同一行上
        results_df = results_df.pivot(index="Param", columns="Metrics", values="Value")

        # 计算各列的平均值，并添加到DataFrame的最后一行
        mean_values = results_df.mean(axis=0)
        mean_values.name = "Average"
        results_df = pd.concat([results_df, pd.DataFrame(mean_values).T])

        results_df = results_df.reset_index()
        return results_df

    def draw_results(self, y_true_train, y_pred_train, y_true, y_pred, scores):
        """
        Draw various plots based on the results of model predictions and true values.

        Args:
            y_true_train (numpy.ndarray or list): True target values for training data.
            y_pred_train (numpy.ndarray or list): Predicted values for training data.
            y_true (numpy.ndarray or list): True target values.
            y_pred (numpy.ndarray or list): Predicted values.
            scores (list): List of evaluation scores corresponding to each prediction.

        Notes:
            - If `y_true` is a list, multiple plots are generated for each subset of true values and predictions.
              - Line plots (`plot_true_pred_line`) are generated for each subset of `y_true` and corresponding `y_pred`.
              - Scatter plots (`plot_true_pred_scatter` and `plot_train_test_scatter`) are generated for each subset
                of `y_true`, `y_pred`, and corresponding `y_true_train`, `y_pred_train`, and scores.
            - If `y_true` is not a list, a single line plot and multiple scatter plots are generated based on the provided data.
        """
        if self.plot_results:
            if isinstance(y_true, list):
                # 数据划分及之前多个参数
                for i, single_y_true in enumerate(y_true):
                    indices = range(
                        i, len(y_pred), len(y_true)
                    )  # 从 y_pred 中选择对应的索引
                    selected_y_pred = [y_pred[i] for i in indices]
                    Draw.plot_true_pred_line(
                        single_y_true,
                        selected_y_pred,
                        self.save_dir,
                        f"result_line_split_{i+1}",
                        self.task_name,
                    )

                for idx, (
                    single_y_true_tr,
                    single_y_pred_tr,
                    single_y_true,
                    single_y_pred,
                    single_scores,
                ) in enumerate(
                    zip(
                        itertools.cycle(y_true_train),
                        y_pred_train,
                        itertools.cycle(y_true),
                        y_pred,
                        scores,
                    )
                ):
                    Draw.plot_true_pred_scatter(
                        single_y_true,
                        single_y_pred,
                        single_scores,
                        self.save_dir,
                        f"result_scatter_{idx+1}",
                        f"{self.task_name}_{idx+1}",
                        0.1,
                    )
                    Draw.plot_true_pred_scatter_train_test(
                        single_y_true_tr,
                        single_y_pred_tr,
                        single_y_true,
                        single_y_pred,
                        single_scores,
                        self.save_dir,
                        f"train_test_scatter_{idx+1}",
                        f"{self.task_name}_{idx+1}",
                        0.1,
                    )
            else:
                Draw.plot_true_pred_line(
                    y_true, y_pred, self.save_dir, "result_line", self.task_name
                )

                for idx, (single_y_pred_tr, single_y_pred, single_scores) in enumerate(
                    zip(y_pred_train, y_pred, scores)
                ):
                    Draw.plot_true_pred_scatter(
                        y_true,
                        single_y_pred,
                        single_scores,
                        self.save_dir,
                        f"result_scatter_{idx+1}",
                        f"{self.task_name}_{idx+1}",
                        0.1,
                    )
                    Draw.plot_true_pred_scatter_train_test(
                        y_true_train,
                        single_y_pred_tr,
                        y_true,
                        single_y_pred,
                        single_scores,
                        self.save_dir,
                        f"train_test_scatter_{idx+1}",
                        f"{self.task_name}_{idx+1}",
                        0.1,
                    )
