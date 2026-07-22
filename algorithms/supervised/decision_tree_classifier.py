import numpy as np
from collections import Counter
from dataclasses import dataclass


@dataclass
class Node:
    feature: int = None
    threshold: float = None
    left: "Node" = None
    right: "Node" = None
    value: int = None

    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeClassifier:
    """Decision Tree Classifier using the Gini Impurity criterion."""

    def __init__(
        self,
        max_depth: int = 100,
        min_samples_split: int = 2,
        n_features: int = None,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        if self.n_features is None:
            self.n_features = X.shape[1]
        else:
            self.n_features = min(self.n_features, X.shape[1])

        self.root = self._grow_tree(X, y)
        return self

    def _grow_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int = 0,
    ) -> Node:

        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feature_indices = np.random.choice(
            n_features,
            self.n_features,
            replace=False,
        )

        best_feature, best_threshold = self._best_split(
            X,
            y,
            feature_indices,
        )

        if best_feature is None:
            return Node(value=self._most_common_label(y))

        left_indices, right_indices = self._split(
            X[:, best_feature],
            best_threshold,
        )

        left = self._grow_tree(
            X[left_indices],
            y[left_indices],
            depth + 1,
        )

        right = self._grow_tree(
            X[right_indices],
            y[right_indices],
            depth + 1,
        )

        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left,
            right=right,
        )

    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: np.ndarray,
    ):

        best_gain = -1
        split_feature = None
        split_threshold = None

        for feature in feature_indices:

            X_column = X[:, feature]
            thresholds = np.unique(X_column)

            for threshold in thresholds:

                gain = self._information_gain(
                    y,
                    X_column,
                    threshold,
                )

                if gain > best_gain:
                    best_gain = gain
                    split_feature = feature
                    split_threshold = threshold

        return split_feature, split_threshold
        
    def _information_gain(
        self,
        y: np.ndarray,
        X_column: np.ndarray,
        threshold: float,
    ) -> float:

        parent_gini = self._gini(y)

        left_indices, right_indices = self._split(
            X_column,
            threshold,
        )

        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        n = len(y)
        n_left = len(left_indices)
        n_right = len(right_indices)

        left_gini = self._gini(y[left_indices])
        right_gini = self._gini(y[right_indices])

        child_gini = (
            (n_left / n) * left_gini
            + (n_right / n) * right_gini
        )

        return parent_gini - child_gini

    def _gini(self, y: np.ndarray) -> float:
        counts = np.bincount(y.astype(int))
        probabilities = counts / len(y)

        return 1 - np.sum(probabilities ** 2)

    def _split(
        self,
        X_column: np.ndarray,
        threshold: float,
    ):

        left_indices = np.argwhere(
            X_column <= threshold
        ).flatten()

        right_indices = np.argwhere(
            X_column > threshold
        ).flatten()

        return left_indices, right_indices

    def _most_common_label(self, y: np.ndarray):
        return Counter(y).most_common(1)[0][0]

    def _traverse_tree(
        self,
        x: np.ndarray,
        node: Node,
    ):

        if node.is_leaf():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)

        return self._traverse_tree(x, node.right)

    def predict(
        self,
        X_test: np.ndarray,
    ) -> np.ndarray:

        if self.root is None:
            raise ValueError("Model has not been fitted yet.")

        X_test = np.asarray(X_test, dtype=float)

        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        predictions = [
            self._traverse_tree(x, self.root)
            for x in X_test
        ]

        return np.asarray(predictions)