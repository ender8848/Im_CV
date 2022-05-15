#############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the fit() and predict() methods of DecisionTreeClassifier.
# You are free to add any other methods as needed. 
##############################################################################

import numpy as np
from util import Util
from numpy.random import default_rng

class Node:
    """Basic decision tree classifier
    Attributes:
    attribute(int): indicate feature number, range from 0 to 15
    value(int):     indicate solit value for a feature, range from feature.min to feature.max
    left(pointer):  pointer to left Node
    right(pointer): pointer to right Node
    label(char):    the label of current node, leaf node has this value, other node is set as None
    
    Methods:
    add_child(node):Add a child node to the current node, by default add to left first, then right, no return.
    """
    def __init__(self, attribute=-1, value=-1, train_set=None, train_label=None):
        self.attribute = attribute  # 16 feature number, in range(0,15)
        self.value = value  # split value for a feature, in range(min, max)
        self.left = None  # pointer to left Node
        self.right = None # pointer to right Node
        self.label = None # leaf node has label indicating the classification result
        self.leaf = False
        self.x = train_set 
        self.y = train_label 

    def __str__(self):
        """
        add child node, by default add to left first and then right
        Args: 
        node(Node): child node to be added
        """
        return "attribute:" + str(self.attribute) + "\nvalue:" + str(self.value) + "\nlabel:" + str(self.label)

    def add_child(self, node):
        if self.left == None:
            self.left = node
            return
        elif self.right == None:
            self.right = node
            return
        print("left & right node are full! Something's wrong!")
        return


class DecisionTreeClassifier(object):
    """ Basic decision tree classifier

    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained

    Methods:
    is_leaf(features, labels): Check if the datdaset can be divided further, if not then it is leaf, return True. 
    entropy(labels): calculate entropy given labels, return a float number
    IG(labels_all, labels_left, labels_right): calculate infotmation gain given labels and split labels, return a float number
    find_best_node(features, labels): get the best split attribute number and value, store as Node and return
    split_dataset(features, labels, node): split dataset according to the found best node, 
                                           return a list of two tuples, each containing feature and label data
    induce_decision_tree(features, labels): generate node recursively according to algorithm in specW
    fit(x, y): Constructs a decision tree from data x and label y
    predict_one_data(node, x): predict the label of one data
    predict(x): Predicts the class label of samples x
    prune(x_val, y_val): Post-prunes the decision tree
    """
    depth_limit = 17 # 17 is actually the best hyper parameter
    def __init__(self, node = None):
        # if a node is passed in, then mark the decision as trained
        if node != None:
            self.is_trained = True
            self.root_node = node
            return
        self.is_trained = False
        self.root_node = None

    def is_leaf(features, labels):
        """
        check if the remaining feature is leaf
        Args:
        features(array_like): 2D feature array
        labels(array_like): 1D label array
        Return:
        Boolean value
        """
        if np.unique(labels).size == 1 or features.shape[1] == 1:
            return True
        return False
    
    def entropy(labels):
        """
        compute entropy according to label
        Args: 
        labels(array_like): 1D label array
        Return:
        float: the entropy
        """
        unique, counts = np.unique(labels, return_counts=True)
        distribution = counts/sum(counts)
        return -sum(distribution * np.log2(distribution))
    
    def IG(labels_all, labels_left, labels_right):
        """
        compute information gain given full label and split labels
        Args:
        labels_all(array_like): full label array
        labels_left(array_like): left label array
        labels_right(array_like): right label array
        Return:
        float: information gain
        """
        new_entropy = ((DecisionTreeClassifier.entropy(labels_left) * labels_left.size) + 
                       (DecisionTreeClassifier.entropy(labels_right) * labels_right.size)) / labels_all.size
        return DecisionTreeClassifier.entropy(labels_all) - new_entropy

    def find_best_node(features, labels):
        """
        find best split arrtibute and value
        Args: 
        features(array_like): 2D feature array
        labels(array_like): 1D label array
        Return:
        Node: a node containing best split attribute and value
        """
        result = {}
        feature_num = features.shape[1]
        for attribute_num in range(feature_num):
            for split_value in range(features[:, attribute_num].min(), features[:, attribute_num].max()):
                labels_left = labels[features[:, attribute_num] <= split_value]
                labels_right = labels[features[:, attribute_num] > split_value]
                result[(attribute_num, split_value)] = DecisionTreeClassifier.IG(labels, labels_left, labels_right)
        max_key = max(result, key=result.get)
        return Node(*max_key)

    def split_dataset(features, labels, node):
        """
        split dataset according to given node
        Args:
        features(array_like): 2D feature array
        labels(array_like): 1D label array
        node(Node): the node indicating split attribute and value
        Return:
        split features and labels
        """
        features_left = features[features[:, node.attribute] <= node.value]
        labels_left = labels[features[:, node.attribute] <= node.value]
        features_right = features[features[:, node.attribute] > node.value]
        labels_right = labels[features[:, node.attribute] > node.value]
        return [(features_left, labels_left), (features_right, labels_right)]

    def induce_decision_tree(features, labels, depth = 0):
        """
        induce decision tree recursively
        Args:
        features(array_like): 2D feature array
        labels(array_like): 1D label array
        Return:
        Node: a tree structure node
        """
        if DecisionTreeClassifier.is_leaf(features, labels) or depth > DecisionTreeClassifier.depth_limit:
            unique, counts = np.unique(labels, return_counts=True)
            node = Node()
            node.label = unique[counts.argmin()] # set most frequent label for leaf node
            node.leaf = True
            return node
        
        node = DecisionTreeClassifier.find_best_node(features, labels)
        children_datasets = DecisionTreeClassifier.split_dataset(features, labels, node)
        for child_dataset in children_datasets:
            child_node = DecisionTreeClassifier.induce_decision_tree(*child_dataset, depth + 1)
            child_node.x = child_dataset[0]
            child_node.y = child_dataset[1]
            node.add_child(child_node)
        return node
   
    def fit(self, x, y):
        """ 
        Constructs a decision tree classifier from data
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (N, K)
                               N is the number of instances
                               K is the number of attributes
        y (numpy.ndarray): Class labels, numpy array of shape (N, )
                               Each element in y is a str
        """
        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."

        self.root_node = DecisionTreeClassifier.induce_decision_tree(x, y)
        # set a flag so that we know that the classifier has been trained
        self.is_trained = True
        return self.root_node

    def predict_one_data(node, x):
        """
        predict a single label using a set of data recursively
        Args: 
        node(Node): the decision tree node
        """
        if x[node.attribute] <= node.value:
            if type(node.left) is Node:
                return DecisionTreeClassifier.predict_one_data(node.left, x)
            else:
                return node.label
        else:
            if type(node.right) is Node:
                return DecisionTreeClassifier.predict_one_data(node.right, x)
            else:
                return node.label
    
    def predict(self, x):
        """ 
        Predicts a set of samples using the trained DecisionTreeClassifier.
        Assumes that the DecisionTreeClassifier has already been trained.
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (M, K)
                           M is the number of test instances
                           K is the number of attributes
        Returns:
        numpy.ndarray: A numpy array of shape (M, ) containing the predicted
                       class label for each instance in x
        """
        # make sure that the classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")

        # set up an empty (M, ) numpy array to store the predicted labels
        predictions = np.zeros((x.shape[0],), dtype=np.object)

        for i in range(x.shape[0]):
            predictions[i] = DecisionTreeClassifier.predict_one_data(self.root_node, x[i, :])

        return predictions
    
    
class Evaluation:
    
    def accuracy(y, y_prediction):
        """
        Compute the accuracy given the ground truth and predictions
        Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels
        Returns:
        float : the accuracy
        """
        assert len(y) == len(y_prediction)

        try:
            return np.sum(y == y_prediction) / len(y)
        except ZeroDivisionError:
            return 0.

    def confusion_matrix(y_gold, y_prediction, class_labels = None):
        """
        compute confusion matrix given ground truth label and prediction label
        """
        # if no class_labels are given, obtain the set of unique class labels from
        # the union of the ground truth annotation and the prediction
        if not class_labels:
            class_labels = np.unique(np.concatenate((y_gold, y_prediction)))
        sorted(class_labels)
        print(class_labels)
        confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int)
        # for each correct class (row),
        # compute how many instances are predicted for each class (columns)
        for (i, label) in enumerate(class_labels):
            # get predictions where the ground truth is the current class label
            indices = (y_gold == label)
            gold = y_gold[indices]
            predictions = y_prediction[indices]
            # quick way to get the counts per label
            (unique_labels, counts) = np.unique(predictions, return_counts=True)
            # convert the counts to a dictionary
            frequency_dict = dict(zip(unique_labels, counts))
            # fill up the confusion matrix for the current row
            for (j, class_label) in enumerate(class_labels):
                confusion[i, j] = frequency_dict.get(class_label, 0)
        return confusion


    def recall(y_gold, y_prediction):
        """ 
        Compute the recall score per class given the ground truth and predictions
        Also return the macro-averaged recall across classes.
        Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels
        Returns:
        tuple: returns a tuple (recalls, macro_recall) where
                - recalls is a np.ndarray of shape (C,), where each element 
                  is the recall for class c
                - macro-recall is macro-averaged recall (a float)
        """

        confusion = Evaluation.confusion_matrix(y_gold, y_prediction)
        r = np.zeros((len(confusion),))
        for c in range(confusion.shape[0]):
            if np.sum(confusion[c, :]) > 0:
                r[c] = confusion[c, c] / np.sum(confusion[c, :])
        macro_r = 0.
        if len(r) > 0:
            macro_r = np.mean(r)

        return (r, macro_r)

    def precision(y_gold, y_prediction):
        """
        compute precision for each class and macro precision
        """

        confusion = Evaluation.confusion_matrix(y_gold, y_prediction)
        p = np.zeros((len(confusion),))
        for c in range(confusion.shape[0]):
            if np.sum(confusion[:, c]) > 0:
                p[c] = confusion[c, c] / np.sum(confusion[:, c])
        macro_p = 0.
        if len(p) > 0:
            macro_p = np.mean(p)

        return (p, macro_p)

    def f1_score(y_gold, y_prediction):
        """
        compute f1 for each class and macro f1
        """

        (precisions, macro_p) = Evaluation.precision(y_gold, y_prediction)
        (recalls, macro_r) = Evaluation.recall(y_gold, y_prediction)
        # make sure same length
        assert len(precisions) == len(recalls)
        f = np.zeros((len(precisions),))
        for c, (p, r) in enumerate(zip(precisions, recalls)):
            if p + r > 0:
                f[c] = 2 * p * r / (p + r)

        # Compute the macro-averaged F1
        macro_f = 0.
        if len(f) > 0:
            macro_f = np.mean(f)

        return (f, macro_f)
    
class CrossValidation:
    
    def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
        """ Split n_instances into n mutually exclusive splits at random.
        
        Args:
        n_splits (int): Number of splits
        n_instances (int): Number of instances to split
        random_generator (np.random.Generator): A random generator
        
        Returns:
        list: a list (length n_splits). Each element in the list should contain a 
        numpy array giving the indices of the instances in that split.
        """
        # generate a random permutation of indices from 0 to n_instances
        shuffled_indices = random_generator.permutation(n_instances)
        # split shuffled indices into almost equal sized splits
        split_indices = np.array_split(shuffled_indices, n_splits)
        return split_indices
    
    def train_test_k_fold(n_folds, n_instances, random_generator=default_rng()):
        """ Generate train and test indices at each fold.
        
        Args:
        n_folds (int): Number of folds
        n_instances (int): Total number of instances
        random_generator (np.random.Generator): A random generator

        Returns:
        list: a list of length n_folds. Each element in the list is a list (or tuple) 
        with two elements: a numpy array containing the train indices, and another 
        numpy array containing the test indices.
        """

        # split the dataset into k splits
        split_indices = CrossValidation.k_fold_split(n_folds, n_instances, random_generator)

        folds = []
        for k in range(n_folds):
            # TODO: Complete this
            # take the splits from split_indices and keep the k-th split as testing
            # and concatenate the remaining splits for training
            test_indices = split_indices[k]
            train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])
            folds.append([train_indices, test_indices])
        return folds
    
    def cross_validation(features, labels, n_folds):
        node_list = []
        accuracies = np.zeros((n_folds, ))
        recalls = np.zeros((n_folds, ))
        precisions = np.zeros((n_folds, ))
        f1s = np.zeros((n_folds, ))
        for i, (train_indices, test_indices) in enumerate(CrossValidation.train_test_k_fold(n_folds, len(labels))):
            x_train = features[train_indices, :]
            y_train = labels[train_indices]
            x_test = features[test_indices, :]
            y_test = labels[test_indices]
            classifier = DecisionTreeClassifier()
            node = classifier.fit(x_train, y_train)
            node_list.append(node)
            predictions = classifier.predict(x_test)
            accuracies[i] = Evaluation.accuracy(y_test, predictions)
            recalls[i] = Evaluation.recall(y_test, predictions)[1]
            precisions[i] = Evaluation.precision(y_test, predictions)[1]
            f1s[i] = Evaluation.f1_score(y_test, predictions)[1]
        print(f"all accuracies in {n_folds} cross validation:")
        print(accuracies)
        print(f"mean: {accuracies.mean():.4f}")
        print(f"standard deviation: {accuracies.std():.4f}")
        
        print(f"all recalls in {n_folds} cross validation:")
        print(recalls)
        print(f"mean: {recalls.mean():.4f}")
        print(f"standard deviation: {recalls.std():.4f}")
        
        print(f"all precisions in {n_folds} cross validation:")
        print(precisions)
        print(f"mean: {precisions.mean():.4f}")
        print(f"standard deviation: {precisions.std():.4f}")
        
        print(f"all f1_score in {n_folds} cross validation:")
        print(f1s)
        print(f"mean: {f1s.mean():.4f}")
        print(f"standard deviation: {f1s.std():.4f}")
        return node_list, node_list[f1s.argmax()]
