import os
from abc import ABC
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

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

        # Create directory if it doesn't exist
        os.makedirs("resource/tree", exist_ok=True)

        # Plot the decision tree
        plt.figure(figsize=(12, 6))
        plot_tree(clf, filled=True, feature_names=["position", "angle", "cart", "rate"], class_names=["left", "right"])
        plt.savefig("resource/tree/tree.png")  # Save the tree plot as an image
