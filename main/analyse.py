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
    # 创建主控制器实例并传入配置文件
    controller = MainController(config_name)

    X, y = controller.load_data()

    # 数据预处理
    processed_data, processed_y = controller.preprocess(X, y)

    # 特征选择
    features, processed_y = controller.feature_selection(processed_data, processed_y)
    
    # 数据划分
    X_train, X_test, y_train, y_test = controller.data_split(features, processed_y)

    # 模型训练
    trained_model = controller.model_training(X_train, y_train)

    # 模型预测
    predictions_train, predictions = controller.model_predict(
        trained_model, X_train, X_test
    )

    # 计算指标
    scores = controller.get_scores(y_train, predictions_train, y_test, predictions)

    # 保存结果
    result_path = controller.save_results(y_test, predictions, scores)

    results_df = controller.show_results(result_path)
    markdown_table = results_df.to_markdown(index=False)
    print(markdown_table)

    # 绘制结果
    controller.draw_results(y_train, predictions_train, y_test, predictions, scores)


if __name__ == "__main__":
    config_name = "样机_芯片2"  # 修改此处为你的配置文件名
    # config_name = "葡萄糖-小浓度5"  
    main_flow(config_name)
