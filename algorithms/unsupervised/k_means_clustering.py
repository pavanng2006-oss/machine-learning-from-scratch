import numpy as np

class KMeansClustering:
    """K-Means Clustering model for clustering"""
    def __init__(self, k: int = 3, max_iters: int = 300):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    def _euclidean_distance(self, points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        centroids = np.asarray(centroids, dtype=float)

        return np.sqrt(np.sum((centroids - points)**2, axis=1))

    def fit(self, X: np.ndarray) -> "KMeansClustering":
        if self.k > X.shape[0]:
            raise ValueError("k cannot be greater than the number of samples")
        
        X = np.asarray(X, dtype=float)
        indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[indices]
        for _ in range(self.max_iters):

            clusters = [[] for _ in range(self.k)]

            for idx, point in enumerate(X):
                distances = self._euclidean_distance(point, self.centroids)
                cluster_idx = np.argmin(distances)
                clusters[cluster_idx].append(idx)


            new_centroids = np.array([X[cluster].mean(axis=0) if cluster 
                                      else self.centroids[i] for i, cluster 
                                      in enumerate(clusters)])

            if np.allclose(new_centroids, self.centroids):
                break

            self.centroids = new_centroids
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet")

        X = np.asarray(X, dtype=float)
        labels = []

        for point in X:
            distances = self._euclidean_distance(point, self.centroids)
            labels.append(np.argmin(distances))

        return np.array(labels)

    