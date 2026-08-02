import numpy as np
def step_function(x):
    return np.where(x > 0, 1, 0)

class Perceptron:
    """
    Binary Classifier using Rosenblatt's Perceptron Learning Algorithm
    """
    def __init__(self, n_iters: int = 1000, alpha: float = 0.001):
        self.n_iters = n_iters
        self.alpha = alpha 
        self.weights = None
        self.bias = None

        self.activation_function = step_function

    def fit(self, X:np.ndarray, y:np.ndarray) -> "Perceptron":
        _, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        y_ = np.where(y > 0, 1, 0)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)

                #update rule

                update = self.alpha * (y_[idx] - y_predicted)

                self.weights += update * x_i
                self.bias += update

        return self
    

    def predict(self, X:np.ndarray) -> np.ndarray:
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)

        return y_predicted




