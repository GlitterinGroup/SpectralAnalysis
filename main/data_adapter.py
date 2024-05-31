import os

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

    def __init__(self, filename, directory=os.path.join(os.path.dirname(__file__), "data")):
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

    def get_feature_and_target(self, data_sheet, target_sheet, target_column):
        """
        Extract feature and target data from specified sheets.

        Args:
            data_sheet (list): List containing the names of two data sheets.
            target_sheet (str): Name of the target sheet.
            target_column (str): Name of the target column in the target sheet.

        Returns:
            tuple: A tuple containing X (features) and y (target) data arrays.
        """
        if len(data_sheet) != 2:
            raise ValueError(
                "data_sheets should be a list containing exactly two sheet names."
            )

        # Read data from two sheets and perform element-wise division
        sheet1_data = self.data[data_sheet[0]].to_numpy()
        sheet2_data = self.data[data_sheet[1]].to_numpy()

        # Ensure both sheets have the same shape for element-wise division
        if sheet1_data.shape != sheet2_data.shape:
            raise ValueError(
                "The two data sheets must have the same shape for element-wise division."
            )

        # Perform element-wise division
        X = sheet1_data / sheet2_data

        tar = pd.read_excel(self.file_path, sheet_name=target_sheet)
        y = tar[target_column].to_numpy()
        return X, y
