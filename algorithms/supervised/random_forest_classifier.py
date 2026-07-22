import numpy as np
from collections import Counter

from decision_tree_classifier import DecisionTreeClassifier


class RandomForestClassifier:
    """Random Forest Classifier."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 100,
        min_samples_split: int = 2,
        n_features: int = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def _bootstrap_sample(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ):

        n_samples = X.shape[0]

        indices = np.random.choice(
            n_samples,
            size=n_samples,
            replace=True,
        )

        return X[indices], y[indices]

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "RandomForestClassifier":

        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                "X and y must contain the same number of samples."
            )

        self.trees = []

        for _ in range(self.n_estimators):

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features,
            )

            X_sample, y_sample = self._bootstrap_sample(X, y)

            tree.fit(X_sample, y_sample)

            self.trees.append(tree)

        return self
    
    def predict(
        self,
        X_test: np.ndarray,
    ) -> np.ndarray:

            if len(self.trees) == 0:
                raise ValueError("Model has not been fitted yet.")

            X_test = np.asarray(X_test, dtype=float)

            if X_test.ndim == 1:
                X_test = X_test.reshape(1, -1)

            tree_predictions = np.array(
                [tree.predict(X_test) for tree in self.trees]
            )

            # Shape: (n_estimators, n_samples)
            tree_predictions = tree_predictions.T

            predictions = []

            for sample_predictions in tree_predictions:
                prediction = Counter(sample_predictions).most_common(1)[0][0]
                predictions.append(prediction)

            return np.asarray(predictions)