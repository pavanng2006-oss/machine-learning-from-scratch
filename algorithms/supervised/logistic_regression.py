import numpy as np

class LogisticRegression:
    """Logistic Regression for binary classification"""
    def __init__(self, alpha: float=0.01, n_iters: int=10000, threshold: float=0.5):
        self.alpha = alpha
        self.n_iters = n_iters
        self.threshold = threshold
        self.beta = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        return 1 / (1 + np.exp(-z))
    
    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n = X.shape[0] # Number of training samples(data points)
        ones = np.ones((n, 1)) # Bias column for the intercept term
        return np.hstack([ones, X])
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LogisticRegression":
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                "X_train and y_train must contain the same number of samples."
            )

        X_prime = self._add_bias(X_train)
        n, p_plus_1 = X_prime.shape
        self.beta = np.zeros(p_plus_1)
        for _ in range(self.n_iters):
            z = X_prime @ self.beta
            y_hat = self._sigmoid(z)
            gradient = X_prime.T @ (y_hat - y_train) / n
            self.beta -= self.alpha * gradient

        return self
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_test = np.asarray(X_test, dtype=float)
        X_prime = self._add_bias(X_test)

        if self.beta is None:
            raise ValueError("Model has not been fitted yet.")

        z = X_prime @ self.beta
        probabilities =  self._sigmoid(z)
        return (probabilities >= self.threshold).astype(int)
    
    @property
    def coef_(self) -> np.ndarray:
        """Returns coefficients learned during training"""
        if self.beta is None:
            raise ValueError("fit() must be called before accessing model parameters.")
        return self.beta[1:]
    
    @property
    def intercept_(self) -> float:
        """Returns intercept learned during training"""
        if self.beta is None:
            raise ValueError("fit() must be called before accessing model parameters.")
        return self.beta[0]
    


