import os
from abc import ABC

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

from sklearn import tree


class TreeTop(ABC):

    def __init__(
        self,
        tree_input,
        tree_output,
        path,
        env_name,
    ) -> None:
        super().__init__()

        self.tree_output = tree_output
        self.tree_input = tree_input

        len_1 = len(tree_output)
        len_2 = len(tree_input)
        y = tree_output
        x = tree_input

        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
        if env_name == "CartPole-v1":
            fn = ["position", "angle", "cart", "rate"]
            cn = ["left", "right"]
            te = DecisionTreeClassifier(max_depth=len(fn), random_state=0)

        elif env_name == "highway-v0":
            print("未实现")
            return
        elif env_name == "Pendulum-v1":
            fn = ["cos(theta)", "sin(theta)", "thetadot"]
            cn = ["torque"]
            te = DecisionTreeRegressor(max_depth=len(fn), random_state=0)
        else:
            print("未实现")
            return
        te.fit(x_train, y_train)

        accuracy = te.score(x_test, y_test)
        print("决策树精度:", accuracy)

        plt.figure(figsize=(16, 8))
        # fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
        tree.plot_tree(te, feature_names=fn, class_names=cn, filled=True)

        self.ensure_dir(path)

        plt.savefig(path, dpi=600)

    def ensure_dir(self, file_path):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
