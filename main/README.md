# 光谱数据分析

---

## 使用方法

1. **数据导入**
   将数据文件放入 `main/data` 目录。

2. **设置配置文件**
   在 `main/config` 目录下创建相应的 `.yaml` 配置文件，建议与数据文件同名。具体参数设置，参见 [Config 设置](#Config设置)。

3. **修改配置名称**
   编辑 `main/analyse.py` 文件末尾的 `config_name` 变量，将其设置为新建的配置文件名。

4. **运行分析**
   运行 `analyse.py` 文件。

5. **查看结果**
   在 `main/result` 目录下查看生成的结果。

---

## 配置说明

### Config设置

- **数据设置 (`data` 部分)**

  - `file_name`: 数据文件的完整名称，包括后缀
  - `data_sheet`: 光谱和光源光谱的表名，作为一个包含两个元素的数组
  - `target_sheet`: 标签所在表的名称
  - `target_column`: 标签所在列的名称
  - `task_type`: 任务类型，可选 `regression` 或 `classification`

- **可视化设置 (`plot` 部分)**

  - `plot_data`: 是否可视化数据
  - `plot_results`: 是否可视化结果

- **预处理设置 (`preprocess` 部分)**

  - `preprocess`: 预处理方法，对应 `preprocess.py` 内的函数名，大小写不限，详见文档
  - `preprocess_params`: 预处理参数
    - 可以缺失，使用默认参数，例如：
      ```yaml
      preprocess: move_avg
      ```
    - 支持多个参数，例如：
      ```yaml
      preprocess: move_avg
      preprocess_params:
        window_size: [2, 3, 4]
      ```
    - 支持多组参数，例如：
      ```yaml
      preprocess: sg
      preprocess_params:
        deriv: 1
        poly_order: [1, 2]
        window_len: [11, 12, 13]
      ```

- **特征选择设置 (`feature_selection` 部分)**

  - `feature_selection`: 特征选择方法，对应 `feature_selection.py` 内的函数名，大小写不限，详见文档
  - `feature_params`: 特征选择参数
    - 可以缺失，使用默认参数
    - 支持多个参数和多组参数，与预处理参数类似

- **数据集划分设置 (`data_split` 部分)**

  - `data_split`: 训练集测试集划分方法，对应 `data_split.py` 内的函数名（去掉 `_split`），大小写不限，详见文档
  - `split_params`: 训练集测试集划分参数
    - 可以缺失，使用默认参数
    - 支持多个参数和多组参数，与预处理参数类似

- **模型设置 (`model` 部分)**
  - `model`: 模型名称，即 `/model` 目录下的文件名（不包含后缀），大小写不限，详见文档
  - `model_params`: 模型参数
    - 可以缺失，使用默认参数
    - 支持多个参数和多组参数，与预处理参数类似

**注：** 整个 `config` 文件支持多部分参数的组合。例如多组特征选择参数 + 多组模型参数。最后保存时以 `Param` 标识不同的参数，序号是递归的。例如 pca 列表参数 + pls 模型参数列表会以模型参数在最外层的顺序保存。

### `result.csv` 文件内容

该 CSV 文件用于记录模型预测的详细结果及评估指标。其格式如下：

```csv
Actual,Predicted,Param,Metrics,Value,Config
<实际值>,<预测值>,<参数序号>,<度量标准>,<度量值>,<配置>
```

#### 字段说明：

1. **Actual**: 实际观测值，表示真实的数据值。
2. **Predicted**: 预测值，表示模型预测的数据值。
3. **Param**: 参数序号，用于标识不同的实验设置。
4. **Metrics**: 评估指标的名称，如 MAE（平均绝对误差）、MSE（均方误差）、RMSE（均方根误差）、R²（判定系数）等。
5. **Value**: 评估指标的数值。
6. **Config**: 当前使用的完整配置文件。

#### 其他细节：

- 每组 `Param` 结果之后，包含该组的 `Metrics` 记录。
- 最后一行的 `Config` 列记录使用的配置文件内容。

---

### 其他

- 如果数据存储格式不同，可以修改 `data_adapter.py` 和相应的 `.yaml` 文件，以获取所需的 X 和 y。
- `spectral_analysis` 文件夹内包含基础的处理方法，通常不需要改动。
- 主要逻辑位于 `main_controller.py`，可根据需求进行修改。
- Docstring 格式: Google
