from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class DTRegression:
    """
    A regression model using a decision tree.

    Methods:
        train(X, y): Train the regression model with input data X and target values y.
        predict(X): Predict the target values for the input data X.
    """

    def __init__(self):
        """
        Initialize the DTRegression model.
        """
        self.model = DecisionTreeRegressor()
    
    def train(self, X, y):
        """
        Train the regression model with input data and target values.

        Args:
            X (array-like): Training data.
            y (array-like): Target values.
        """
        self.model.fit(X, y)
    
    def predict(self, X):
        """
        Predict the target values for the input data.

        Args:
            X (array-like): Input data for which to predict target values.

        Returns:
            array: Predicted target values for the input data.
        """
        return self.model.predict(X)

class DTClassifier:
    """
    A classification model using a decision tree.

    Methods:
        train(X, y): Train the classification model with input data X and target labels y.
        predict(X): Predict the labels for the input data X.
    """

    def __init__(self):
        """
        Initialize the DTClassifier model.
        """
        self.model = DecisionTreeClassifier()
    
    def train(self, X, y):
        """
        Train the classification model with input data and target labels.

        Args:
            X (array-like): Training data.
            y (array-like): Target labels.
        """
        self.model.fit(X, y)
    
    def predict(self, X):
        """
        Predict the labels for the input data.

        Args:
            X (array-like): Input data for which to predict labels.

        Returns:
            array: Predicted labels for the input data.
        """
        return self.model.predict(X)
