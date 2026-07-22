import numpy as np

class LinearRegressionClosed:
    """Linear Regression using the Normal Equation."""
    def __init__(self):
        self.beta = None
        # Regression coefficients learned during training.

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n = X.shape[0] # Number of training samples(data points)
        ones = np.ones((n, 1)) # Bias column for the intercept term
        return np.hstack([ones, X])
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LinearRegressionClosed":
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                "X_train and y_train must contain the same number of samples."
            )
        X_prime = self._add_bias(X_train) 
        self.beta = np.linalg.solve(X_prime.T @ X_prime, X_prime.T @ y_train)
        # Solve (X'ᵀX')β = X'ᵀy using the Normal Equation,
        # where X' = [1 | X] (the feature matrix with the bias column added).
        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        X_test = np.asarray(X_test, dtype=float)
        if self.beta is None:
            raise ValueError("Model has not been fitted yet.")
        X_test_prime = self._add_bias(X_test)
        return X_test_prime @ self.beta
    
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
