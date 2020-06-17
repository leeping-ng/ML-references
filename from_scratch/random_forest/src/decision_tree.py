import pandas as pd
import numpy as np

class DecisionTree:
    def __init__(self):
        pass

    def partition(self, rows, question):
        """
        Partitions a dataset

        For each row in the dataset, check if it matches the question. 
        If so, add it to 'true rows', otherwise, add it to 'false rows'.

        Takes in:
            list of lists: rows, dataset
            Question object: to call the "match" method to partition the dataset

        Returns:
            2 list of lists: true_rows, false_rows
        """
        true_rows, false_rows = [], []
        for row in rows:
            if question.match(row):
                true_rows.append(row)
            else:
                false_rows.append(row)
        return true_rows, false_rows


    def gini_impurity(self, rows):
        """
        Calculate the Gini Impurity for a node

        Takes in:
            list of lists: dataset at the node

        Returns:
            float: Gini Impurity
        """
        counts = {}  
        for row in rows:
            # in our dataset format, the label is always the last column
            label = row[-1]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        
        impurity = 1
        # iterate through num of unique labels
        for lbl in counts:
            prob_of_lbl = counts[lbl] / float(len(rows))
            impurity -= prob_of_lbl**2
        return impurity


    def find_best_split(self, rows):
        """
        Find the best question to ask by iterating over every feature / value
        and calculating the information gain.
        
        Takes in:
            list of lists: rows, dataset with labels

        Returns:
            float: best_gain
            Question object: best_question
        """
        best_gain = 0  
        best_question = None
        current_uncertainty = self.gini_impurity(rows)
        
        # Subtract 1 to exclude the label column
        n_features = len(rows[0]) - 1  

        for col in range(n_features):  
            # unique values for each column
            values = set([row[col] for row in rows])  

            for val in values:
                question = self.Question(col, val)

                # try splitting the dataset
                true_rows, false_rows = self.partition(rows, question)

                # Skip this split if it doesn't divide the dataset
                if len(true_rows) == 0 or len(false_rows) == 0:
                    continue

                # Calculate the information gain from this split using weighted average
                true_uncertainty = float(len(true_rows)) / (len(true_rows)+len(false_rows)) \
                    * self.gini_impurity(true_rows)
                false_uncertainty = float(len(false_rows)) / (len(true_rows)+len(false_rows)) \
                    * self.gini_impurity(false_rows)
                gain = current_uncertainty - true_uncertainty - false_uncertainty

                if gain >= best_gain:
                    best_gain, best_question = gain, question

        return best_gain, best_question


    def recur_build_tree(self, rows):
        """
        Recursive function that builds the tree
        It will keep storing Leafs and Decision_Nodes in memory
        The final returned Decision_Node is the root node of the tree

        Takes in:
            list of lists: rows, dataset with label
        
        Returns:
            Decision_Node object: records the best value to ask at this point, and the
            corresponding branches (True or False)
        """
        # Find the best question to ask with the best information gain for the dataset
        gain, question = self.find_best_split(rows)

        # Base case: no further info gain
        # Since we can ask no further questions, we'll return a leaf
        if gain == 0:
            return self.Leaf(rows)

        # If we reach here, we have found a useful value to partition on.
        true_rows, false_rows = self.partition(rows, question)

        # Recursively build the true branch
        true_branch = self.recur_build_tree(true_rows)

        # Recursively build the false branch
        false_branch = self.recur_build_tree(false_rows)

        return self.Decision_Node(question, true_branch, false_branch)

    def fit(self, X_train_array, y_train_array):
        '''
        Fits a decision tree model from the training data.
        
        Takes in:
            Numpy array: X_train_array containing features
            Numpy array: y_train_array containing labels
            
        Returns:
            Tree object
        '''
        # y_train has to be joined as the last column
        rows = np.column_stack((X_train_array, y_train_array))

        # convert Numpy array to list of lists
        rows = rows.tolist()

        self.Tree = self.recur_build_tree(rows)
        return self.Tree


    def recur_classify(self, row, node):
        """
        Recursive function to label the data, row by row

        Takes in:
            list: row of dataset without label
            Leaf or Decision_Node object
        """
        # Base case: we've reached a leaf
        if isinstance(node, self.Leaf):
            return node.predictions

        # Decide whether to follow the true_branch or the false_branch
        # Compare the feature/value stored in the node, to the example we're considering
        if node.question.match(row):
            return self.recur_classify(row, node.true_branch)
        else:
            return self.recur_classify(row, node.false_branch)

    def predict(self, X_test_array):
        '''
        Returns a list of predicted labels for an array with features
        Calls the recursive recur_classify function
        
        Takes in:
            Numpy array: X_test_array containing features
            
        Returns:
            List: Labels of integer 1 or 0
        '''
        # Convert from numpy array to list of lists
        rows = X_test_array.tolist()

        labels = []
        # Loop through to add label to list for each row of data
        for i in range(len(rows)):
            labels.append(self.recur_classify(rows[i], self.Tree))

        return labels

    ##################################################################################
    # This section contains all the classes
    # Classes: Question, Leaf, Decision_Node, Tree
    ##################################################################################

    class Question:
        """
        A Question is used to partition a dataset.

        Attributes:
            column: Records a column number
            value: Records a column value

        Methods:
            match: Checks if value in example >= value in Question, and returns a boolean
            __repr__: Prints the question in a readable format

        """
        def __init__(self, column, value):
            self.column = column
            self.value = value

        def match(self, example):
            # Compare the feature value in an example to the
            # feature value in this question.
            val = example[self.column]
            return val >= self.value

        def __repr__(self):
            return "Is feature %s >= %s?" % (self.column, str(self.value))


    class Leaf:
        """
        A Leaf node is used to store the label of a row of data in memory.

        Attributes:
            predictions: Records the label of the leaf node
        """
        def __init__(self, rows):
            self.predictions = rows[0][-1]
            

    class Decision_Node:
        """
        A Decision Node is used to store the question asked at a node, and the "True" and "False" data pertaining to the question

        Attributes:
            question: Records the question asked
            true_branch: Records the lists of data that answer "True" to the question
            false_branch: Records the lists of data that answer "False" to the question
        """

        def __init__(self, question, true_branch, false_branch):
            self.question = question
            self.true_branch = true_branch
            self.false_branch = false_branch


    class Tree:
        """
        A Tree is used to store the resulting decision tree returned from fitting a dataset
        """
        def __init__(self):
            pass

    ##################################################################################
    # This section contains the helper functions for printing
    # Functions: recur_print_tree, print
    ##################################################################################

    def recur_print_tree(self, node, spacing=""):
        """
        A recursive helper function used to print the tree, row by row

        Takes in:
            node: Tree object from print() function
        """
        # Base case: we've reached a leaf
        if isinstance(node, self.Leaf):
            print (spacing + "Predict", node.predictions)
            return

        # Print the question at this node
        print (spacing + str(node.question))

        # Call this function recursively on the true branch
        print (spacing + '--> True:')
        self.recur_print_tree(node.true_branch, spacing + "  ")

        # Call this function recursively on the false branch
        print (spacing + '--> False:')
        self.recur_print_tree(node.false_branch, spacing + "  ")


    def print(self):
        """
        A helper function used to print the entire decision tree
        Calls the recursive recur_print_tree function with a Tree object
        """
        self.recur_print_tree(self.Tree)
        return