import numpy as np

class PrincipalComponentAnalysis:
    """PCA method for dimensionality reduction"""
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def fit(self, X: np.ndarray) -> "PrincipalComponentAnalysis":
        
        X = np.asarray(X, dtype=float)
        if self.n_components > X.shape[1]:
            raise ValueError(
                "n_components cannot exceed the number of features"
                )

        self.mean = np.mean(X, axis=0)

        X_centered = X - self.mean

        covariance = np.cov(X_centered, rowvar=False) #Covariance matrix

        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        indices = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[indices]

        eigenvectors = eigenvectors[:, indices]

        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        self.explained_variance_ratio = (
            self.explained_variance / np.sum(eigenvalues)
        )
        

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components is None or self.mean is None:
            raise ValueError("Model has not been fitted yet")
        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean

        return X_centered @ self.components

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)