import os

import numpy as np
import pandas as pd


class DataAdapter:
    """
    A class to handle data file format inconsistencies and obtain unified X and y data.

    Attributes:
        filename (str): Name of the data file.
        directory (str): Directory path where the data file is located.
        data: Loaded data from the file.
        file_path (str): Full path to the data file.
        ext (str): File extension of the data file.

    Methods:
        _get_file_path(): Get the full path to the data file.
        _load_data(): Load the data from the file based on its extension.
        get_feature_and_target(data_sheet, target_sheet, target_column): Extract feature and target data from specified sheets.
    """

    def __init__(
        self, filename, directory=os.path.join(os.path.dirname(__file__), "data")
    ):
        """
        Initialize the DataAdapter with the filename and directory.

        Args:
            filename (str): Name of the data file.
            directory (str, optional): Directory path where the data file is located. Default is "./data".
        """
        self.filename = filename
        self.directory = directory

        self._load_data()

    def _get_file_path(self):
        """
        Get the full path to the data file.
        """
        self.file_path = os.path.join(self.directory, self.filename)
        _, self.ext = os.path.splitext(self.file_path)

    def _load_data(self):
        """
        Load the data from the file based on its extension.
        """
        self._get_file_path()

        if self.ext == "csv":
            self.data = pd.read_csv(self.file_path)
        elif self.ext in [".xls", ".xlsx"]:
            self.data = pd.read_excel(self.file_path, sheet_name=None, header=None)
        else:
            raise ValueError(f"Unsupported file extension: {self.ext}")

    def get_feature_and_target(
        self,
        data_sheet,
        data_transpose,
        data_start_row,
        target_sheet,
        target_column,
        target_start_row,
    ):
        """
        Extract feature and target data from specified sheets.

        Args:
            data_sheet (list): List of sheet names to extract feature data from.
                           It should contain one or two sheet names.
            data_transpose (bool): Whether to transpose the feature data matrices.
            data_start_row (int): The starting row index for the feature data extraction.
            target_sheet (str): The sheet name to extract target data from.
            target_column (int): The column index for the target data extraction.
            target_start_row (int): The starting row index for the target data extraction.

        Returns:
            tuple: A tuple containing X (features) and y (target) data arrays.

        Raises:
            ValueError: If `data_sheet` contains more than two sheet names or if the two data sheets have different shapes when element-wise division is required.
        """
        if len(data_sheet) == 2:
            sheet1_data = self.data[data_sheet[0]][data_start_row:].to_numpy(
                dtype=float
            )
            sheet2_data = self.data[data_sheet[1]][data_start_row:].to_numpy(
                dtype=float
            )

            if data_transpose:
                sheet1_data = sheet1_data.T
                sheet2_data = sheet2_data.T

            if sheet1_data.shape != sheet2_data.shape:
                raise ValueError(
                    "The two data sheets must have the same shape for element-wise division."
                )

            # Perform element-wise division to get feature matrix X
            X = sheet1_data / sheet2_data
        elif len(data_sheet) == 1:
            sheet_data = self.data[data_sheet[0]][data_start_row:].to_numpy(dtype=float)

            if data_transpose:
                sheet_data = sheet_data.T

            X = sheet_data
        else:
            raise ValueError(
                "data_sheets should be a list containing exactly one or two sheet names."
            )

        tar = pd.read_excel(self.file_path, sheet_name=target_sheet, header=None)
        y = tar.iloc[target_start_row:, target_column].to_numpy()
        y = y.astype(np.float64)

        return X, y
