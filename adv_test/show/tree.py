import os
from abc import ABC

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

import graphviz

from sklearn import tree

def write_to_file(text):
    # 检查文件是否存在
    if not os.path.exists("view/display.txt"):
        # 如果文件不存在，创建文件
        with open("view/display.txt", 'w', encoding='utf-8') as f:
            f.write(text)
    else:
        # 如果文件已存在，在末尾追加内容
        with open("view/display.txt", 'a', encoding='utf-8') as f:
            f.write(text)
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
        # write_to_file("决策树精度:"+str(round(accuracy),2))
        fn = ["position", "angle", "cart", "rate"]
        cn = ["left", "right"]

        dot_data = tree.export_graphviz(
            clf,
            out_file=None,
            feature_names=fn,
            class_names=cn,
            filled=True,
            rounded=True,
            impurity=False,  # Remove Gini index
            proportion=False,
        )

        graph = graphviz.Source(dot_data)

        # remember to change for your own address
        graph.save("resource/tree/tree.dot")

        graph.render("resource/tree/tree", format="png")
