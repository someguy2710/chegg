#!/usr/bin/env python
# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree


def load_data_file(filename):
    with open(filename) as f:
        data = f.readlines()
    data = np.array([row.split(' ')[:-1] for row in data])
    X = data[:, 1:]
    y = data[:, 0]
    assert X.shape[0] == y.shape[0]
    return X, y


def load_data(dataset):
    """loads both train and test data of a dataset from files
    e.g., X_train, y_train, X_test, y_test = load_data("monks-1")
    """
    train_file_name = dataset + ".train.csv"
    test_file_name = dataset + ".test.csv"
    X_train, y_train = load_data_file(train_file_name)
    print("Training set contains %d examples with %d attributes" % X_train.shape)
    X_test, y_test = load_data_file(test_file_name)
    print("Test set contains %d examples" % X_test.shape[0])
    assert X_train.shape[1] == X_test.shape[1]
    return X_train, y_train, X_test, y_test


def sample(size, X, y):
    """returns a random sample of the given size from the dataset
    """
    indices = np.random.choice(range(len(X)), size=size, replace=False)
    return X[indices], y[indices]


def train_tree(X, y):
    dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
    print("Training tree on %d examples ..." % X.shape[0])
    dt.fit(X, y)
    return dt


def main():
    # Loading the dataset
    X_train, y_train, X_test, y_test = load_data("monks-1")
    # Training a Decision Tree
    X, y = sample(100, X_train, y_train)
    dt = train_tree(X, y)
    print("Accuracy on the training data: %.2f%%" % (100 * dt.score(X, y)))
    print("Accuracy on the test data: %.2f%%" % (100 * dt.score(X_test, y_test)))

    fig, ax = plt.subplots()
    fig.set_size_inches(20, 10)
    plot_tree(dt, ax=ax, fontsize=8)
    plt.show()
    return


if __name__ == "__main__":
    main()
