import numpy as np


class DBSCAN:
    """Density-Based Spatial Clustering of Applications with Noise"""

    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    @staticmethod
    def _euclidean_distance(p: np.ndarray, q: np.ndarray) -> float:
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)
        return np.sqrt(np.sum((p - q) ** 2))

    def _region_query(self, X: np.ndarray, point_idx: int) -> list:
        neighbours = []

        for i in range(len(X)):
            if self._euclidean_distance(X[point_idx], X[i]) <= self.eps:
                neighbours.append(i)

        return neighbours

    def _expand_cluster(
        self,
        X: np.ndarray,
        point_idx: int,
        neighbours: list,
        cluster_id: int,
        visited: np.ndarray,
    ):

        self.labels[point_idx] = cluster_id

        i = 0
        while i < len(neighbours):
            neighbour_idx = neighbours[i]

            if not visited[neighbour_idx]:
                visited[neighbour_idx] = True

                neighbour_neighbours = self._region_query(X, neighbour_idx)

                if len(neighbour_neighbours) >= self.min_samples:
                    for point in neighbour_neighbours:
                        if point not in neighbours:
                            neighbours.append(point)

            if self.labels[neighbour_idx] == -1:
                self.labels[neighbour_idx] = cluster_id

            i += 1

    def fit(self, X: np.ndarray) -> "DBSCAN":
        X = np.asarray(X, dtype=float)

        n_samples = len(X)

        visited = np.zeros(n_samples, dtype=bool)
        self.labels = np.full(n_samples, -1)

        cluster_id = 0

        for point_idx in range(n_samples):

            if visited[point_idx]:
                continue

            visited[point_idx] = True

            neighbours = self._region_query(X, point_idx)

            if len(neighbours) < self.min_samples:
                self.labels[point_idx] = -1
            else:
                self._expand_cluster(
                    X,
                    point_idx,
                    neighbours,
                    cluster_id,
                    visited,
                )
                cluster_id += 1

        return self

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.labels