import os
from abc import ABC

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

import graphviz

from sklearn import tree

class TreeTop(ABC):

    def __init__(
        self,
        tree_input,
        tree_output,
    ) -> None:
        super().__init__()

        self.tree_output = tree_output
        self.tree_input = tree_input

        len_1 = len(tree_output)
        len_2 = len(tree_input)
        y = tree_output
        x = tree_input

        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
        clf = DecisionTreeClassifier(max_depth=3, random_state=0)
        clf.fit(x_train, y_train)

        accuracy = clf.score(x_test, y_test)
        print("决策树精度:", accuracy)
        fn = ["position", "angle", "cart", "rate"]
        cn = ["left", "right"]

        dot_data = tree.export_graphviz(clf
                            ,out_file = None
                            ,feature_names= fn
                            ,class_names=cn
                            ,filled=True
                            ,rounded=True
                            ,impurity=False # Remove Gini index
                            ,proportion=False
                            )
        graph = graphviz.Source(dot_data)
        #remember to change for your own address
        graph.save('resource/tree/tree.dot')
        graph.render('resource/tree/tree', format='png')
