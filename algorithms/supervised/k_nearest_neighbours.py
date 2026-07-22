import numpy as np
from collections import Counter

class KNearestNeighbours:
    """K-Nearest Neighbours Classifier"""
    def __init__(self, k: int=3):
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        self.k = k
        self.X = None
        self.y = None

    def _euclidean_distance(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Calculates the euclidean distance"""
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)

        return np.sqrt( np.sum( (p - q)**2 ) )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "KNearestNeighbours":
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)

        if self.k > X_train.shape[0]:
            raise ValueError(
                "k cannot be greater than the number of training samples."
                )

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                "X_train and y_train must have same number of samples"
                )
        
        self.X = X_train
        self.y = y_train

        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.X is None or self.y is None:
            raise ValueError("Model has not been fitted yet.")
        
        X_test = np.asarray(X_test, dtype=float)

        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        predictions = []
        for x in X_test:
            distances = []
            for train_x, label in zip(self.X, self.y):
                distance = self._euclidean_distance(train_x, x)
                distances.append((distance, label))

            labels = [
                label for _, label in sorted(distances)[:self.k]
                ]
            result = Counter(labels).most_common(1)[0][0]
            predictions.append(result)

        predictions = np.array(predictions)
        return predictions[0] if len(predictions) == 1 else predictions
    
            
