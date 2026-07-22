import numpy as np

class LinearRegressionGD:
    """Linear Regression using the Gradient Descent Method"""
    def __init__(self, alpha: float=0.01, n_iters: int=10000):
        self.alpha = alpha
        self.n_iters = n_iters
        self.beta = None

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n = X.shape[0] # Number of training samples(data points)
        ones = np.ones((n, 1)) # Bias column for the intercept term
        return np.hstack([ones, X])
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "LinearRegressionGD":
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float)

        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                "X_train and y_train must contain the same number of samples."
            )

        X_prime = self._add_bias(X_train)
        # n = number of training samples
        # p_plus_1 = number of parameters (features + bias)
        n, p_plus_1 = X_prime.shape

        self.beta = np.zeros(p_plus_1) # Initializing the coef matrix to zero matrix

        for _ in range(self.n_iters):
            # Predicted values: ŷ = X'β
            y_hat = X_prime @ self.beta

            # Gradient of the Mean Squared Error (MSE):
            # ∇J(β) = (X'ᵀ(ŷ - y)) / n
            gradient = (X_prime.T @ (y_hat - y_train)) / n 

            # Gradient Descent update:
            # β ← β − α∇J(β)
            self.beta -= self.alpha * gradient

        return self

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Returns the predicted values"""
        X_test = np.asarray(X_test, dtype=float)
        X_test_prime = self._add_bias(X_test)
        if self.beta is None:
            raise ValueError("Model has not been fitted yet.")
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
    



