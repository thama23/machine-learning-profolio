'''
Implementation option.
house-votes-84.data, predicting political party
which is listed as first column.

Compare with implemented decision tree
and sklearn decision tree.
'''

from math import log
import operator
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np

'''
Implement a new algorithm, the ID3 Decision Tree.
'''
class ID3Tree:
    def __init__(self):
        # Constructor
        self.tree = {}
        self.feature_names = []

    def fit(self, training_data, feature_names):
        '''
        Fit given data and build the decision tree
        :param training_data: input training data
        :param feature_names: names for each attribute
        :return:
        '''
        self.tree = self.build_tree(training_data.copy(), feature_names.copy())
        self.feature_names = feature_names

    def predict(self, testing_data):
        '''
        Predict the labels of input testing data
        :param testing_data: input testing data
        :return: the predicted labels
        '''
        pred = []
        for data in testing_data:
            # Input testing data
            pred.append(self.classify(self.tree, self.feature_names, data[:-1]))
        return pred

    def accuracy(self, testing_data):
        '''
        Calculate the accuracy score for input testing data
        :param testing_data:  input testing data
        :return: prediction accuracy
        '''
        pred = self.predict(testing_data)
        count = 0
        for p, t in zip(pred, testing_data):
            if p == t[-1]:
                count += 1
        return 1.0 * count / len(pred)

    @staticmethod
    def shannon_entropy(current_data):
        '''
        Calculate the shannon entropy of current data
        :param current_data:
        :return:
        '''
        instance_number = len(current_data)
        label_frequency = defaultdict(int)
        for feature in current_data:
            label_frequency[feature[-1]] += 1
        entropy = 0.0
        for key in label_frequency:
            prob = float(label_frequency[key]) / instance_number
            entropy -= prob * log(prob, 2)  # log base 2
        return entropy

    @staticmethod
    def split_data(current_data, split_position, value):
        '''
        Split current data by specified splitting position
        :param current_data: current dataset
        :param split_position: specified splitting position
        :param value: splitting feature value
        :return: the split dataset
        '''
        split_data = []
        for feature in current_data:
            if feature[split_position] == value:
                reduced_feature = feature[:split_position]
                reduced_feature.extend(feature[split_position + 1:])
                split_data.append(reduced_feature)
        return split_data

    @staticmethod
    def find_split_feature(current_data):
        '''
        Find the best splitting position of current dataset
        :param current_data: current dataset
        :return: the best splitting position
        '''
        features_number = len(current_data[0]) - 1
        entropy_before_split = ID3Tree.shannon_entropy(current_data)
        best_info_gain = 0.0
        best_feature = -1
        for i in range(features_number):
            features = [example[i] for example in current_data]
            new_entropy = 0.0
            for value in set(features):
                split_data = ID3Tree.split_data(current_data, i, value)
                prob = len(split_data) / float(len(current_data))
                new_entropy += prob * ID3Tree.shannon_entropy(split_data)
            info_gain = entropy_before_split - new_entropy
            if info_gain > best_info_gain:
                best_info_gain = info_gain
                best_feature = i
        return best_feature

    @staticmethod
    def count_majority(labels):
        '''
        Count the most frequent label in the label list
        :param labels: input label list
        :return: the most frequent label
        '''
        label_count = {}
        for vote in labels:
            if vote not in label_count.keys(): label_count[vote] = 0
            label_count[vote] += 1
        return sorted(label_count.items(), key=operator.itemgetter(1), reverse=True)[0][0]

    @staticmethod
    def build_tree(current_data, feature_names):
        '''
        Build the decision tree recursively
        :param current_data: current dataset
        :param feature_names: name of the features
        :return: the decision tree node for current dataset
        '''
        labels = [example[-1] for example in current_data]
        if labels.count(labels[0]) == len(labels):
            return labels[0]
        if len(current_data[0]) == 1:
            return ID3Tree.count_majority(labels)
        split_feature = ID3Tree.find_split_feature(current_data)
        split_feature_name = feature_names[split_feature]
        node = {split_feature_name: {}}
        del (feature_names[split_feature])
        feature_value = [example[split_feature] for example in current_data]
        for value in set(feature_value):
            node[split_feature_name][value] = ID3Tree.build_tree(ID3Tree.split_data(current_data, split_feature, value), feature_names.copy())
        return node

    @staticmethod
    def classify(current_tree, feature_names, test_feature):
        """
        classify the label of a input testing feature under current tree
        :param current_tree: current tree
        :param feature_names: name of the features
        :param test_feature: input testing feature
        :return: the classified label
        """
        root_feature = next(iter(current_tree))
        children = current_tree[root_feature]
        root_feature_index = feature_names.index(root_feature)
        key = test_feature[root_feature_index]
        if key not in children:
            value_of_feat = children[next(iter(children))]
        else:
            value_of_feat = children[key]
        if isinstance(value_of_feat, dict):
            test_label = ID3Tree.classify(value_of_feat, feature_names, test_feature)
        else:
            test_label = value_of_feat
        return test_label

# Driver for the program.
if __name__ == '__main__':
    # load the dataset and initlize the data set.
    data = []
    with open("house-votes-84.data") as infile:
        for line in infile.read().splitlines():
            tokens = line.split(",")
            data.append(tokens[1:] + [tokens[0]])
    feature_names = [
        "handicapped-infants",
        "water-project-cost-sharing",
        "adoption-of-the-budget-resolution",
        "physician-fee-freeze",
        "el-salvador-aid",
        "religious-groups-in-schools",
        "anti-satellite-test-ban",
        "aid-to-nicaraguan-contras",
        "mx-missile",
        "immigration",
        "synfuels-corporation-cutback",
        "education-spending",
        "superfund-right-to-sue",
        "crime",
        "duty-free-exports",
        "export-administration-act-south-africa"
    ]

    # spilt the data to training and testing parts, 75% train and 25% test data.
    training_data, testing_data = train_test_split(data, test_size=0.25)

    # using the ID3Tree to classify the data
    print("Using our implemented Decision Tree:")
    id3_tree = ID3Tree()
    id3_tree.fit(training_data, feature_names)
    print("Training accuracy={}".format(id3_tree.accuracy(training_data)))
    print("Testing accuracy={}".format(id3_tree.accuracy(testing_data)))

    # using the sklearn tree
    print("\nUsing the sklearn Decision Tree:")

    training_features = [d[:-1] for d in training_data]
    training_features = np.asarray(training_features)
    training_features[training_features == 'y'] = 1
    training_features[training_features == 'n'] = 2
    training_features[training_features == '?'] = 0
    training_lables = [d[-1] for d in training_data]
    testing_features = [d[:-1] for d in testing_data]
    testing_features = np.asarray(testing_features)
    testing_features[testing_features == 'y'] = 1
    testing_features[testing_features == 'n'] = 2
    testing_features[testing_features == '?'] = 0
    testing_lables = [d[-1] for d in testing_data]

    clf = tree.DecisionTreeClassifier()
    clf.fit(training_features, training_lables)
    print("Training accuracy={}".format(clf.score(training_features, training_lables)))
    print("Testing accuracy={}".format(clf.score(testing_features, testing_lables)))
