import numpy as np
from src.decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees=5, subsample_size=1, feature_proportion=1):
        self.trees = []
        self.n_trees = n_trees
        self.subsample_size = subsample_size
        self.feature_proportion = feature_proportion


    def fit(self, X_train_array, y_train_array):
        """
        Fits the random forest model 

        Takes in:
            2 Numpy arrays: X_train_array, y_train_array
        """

        for i in range(self.n_trees):
            boot_X_array, boot_y_array = self.bootstrap(X_train_array, y_train_array)
            dt = DecisionTree()
            dt.fit(boot_X_array, boot_y_array)
            self.trees.append(dt)

        return 


    def bootstrap(self, X_train_array, y_train_array):
        """
        Bootstrap function for bootstrapping dataset
        Randomly shuffles rows and returns arrays with number of rows determined by subsample_size
        Returns arrays with number of cols determined by feature_proportion

        Takes in:
            2 Numpy arrays: X_train_array, y_train_array
        
        Returns:
            2 Numpy arrays: boot_X_array, boot_y_array
        """

        # Calculate number of rows for subsamples based on subsample_size ratio
        n_rows = round(len(X_train_array) * self.subsample_size)

        # index of rows to bootstrap
        idx_rows = np.random.choice(len(X_train_array), size=n_rows, replace=True)

        # For columns, because of the way decision_tree is built,
        # We can't shuffle the order of the columns or remove columns
        # What we can do is zero out the columns that won't be used for training
        # Calculate number of cols to be zeroed out based on feature_proportion ratio
        n_cols = len(X_train_array[1]) - round(len(X_train_array[1]) * self.feature_proportion)

        # Index of columns to be zeroed out
        idx_cols = (np.random.choice(len(X_train_array[1]), n_cols, replace=False)).tolist()
        
        # Reshape y from 1D array to 2D array
        y_train_array = np.expand_dims(y_train_array, axis=1)
        
        boot_X_array = X_train_array[idx_rows, :]
        # zero out the columns that won't be used for training
        boot_X_array[:, idx_cols] = 0

        boot_y_array = y_train_array[idx_rows, :]
     
        return boot_X_array, boot_y_array


    def predict(self, X_test_array):
        """
        Makes predictions on new data from fitted model

        Takes in:
            Numpy arrays: X_test_array

        Returns:
            List: Predicted labels
        """

        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X_test_array))

        predictions_mean = np.mean(predictions, axis=0).tolist()
        labels = [round(label) for label in predictions_mean]

        return labels

