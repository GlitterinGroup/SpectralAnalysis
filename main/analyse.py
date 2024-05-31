import os
import sys
from pathlib import Path

sys.path.insert(1, str(Path(__file__).resolve().parent.parent))

import warnings

from main.main_controller import MainController

warnings.filterwarnings("ignore", category=FutureWarning)


def main_flow(config_name):
    """
    Main workflow function for executing the entire data processing pipeline.

    Args:
        config_name (str): Name of the configuration file without extension.

    Returns:
        None
    """
    # 加载配置文件
    config_path = os.path.join(
        os.path.dirname(__file__), "config", f"{config_name}.yaml"
    )

    # 创建主控制器实例并传入配置文件
    controller = MainController(config_path)

    X, y = controller.load_data()

    # 数据预处理
    processed_data = controller.preprocess(X)

    # 特征选择
    features = controller.feature_selection(processed_data, y)

    # 数据划分
    X_train, X_test, y_train, y_test = controller.data_split(features, y)

    # 模型训练
    trained_model = controller.model_training(X_train, y_train)

    # 模型预测
    predictions = controller.model_predict(trained_model, X_test)

    # 计算指标
    scores = controller.get_scores(y_test, predictions)

    # 保存结果
    result_path = controller.save_results(y_test, predictions, scores)

    results_df = controller.show_results(result_path)
    markdown_table = results_df.to_markdown()
    print(markdown_table)

    # 绘制结果
    controller.draw_results(y_test, predictions, scores)


if __name__ == "__main__":
    config_name = "皮肤水分-样机"   # 修改此处为你的配置文件名
    main_flow(config_name)
