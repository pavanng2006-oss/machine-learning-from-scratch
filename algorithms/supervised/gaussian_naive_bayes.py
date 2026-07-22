import numpy as np

class GaussianNaiveBayes:
    """Gaussian Naive Bayes classifier."""

    def __init__(self):
        self.classes_ = None
        self.priors_ = None
        self.means_ = None
        self.vars_ = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GaussianNaiveBayes":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples.")

        self.classes_ = np.unique(y)

        n_classes = len(self.classes_)
        n_features = X.shape[1]

        self.priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))
        self.vars_ = np.zeros((n_classes, n_features))

        for idx, cls in enumerate(self.classes_):
            X_cls = X[y == cls]

            self.priors_[idx] = X_cls.shape[0] / X.shape[0]
            self.means_[idx] = np.mean(X_cls, axis=0)
            self.vars_[idx] = np.var(X_cls, axis=0)

        return self

    def _gaussian_pdf(self, class_idx: int, x: np.ndarray) -> np.ndarray:
        mean = self.means_[class_idx]
        var = self.vars_[class_idx] + 1e-9  #to avoid division by 0

        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise ValueError("Model has not been fitted yet.")

        X_test = np.asarray(X_test, dtype=float)

        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        predictions = []

        for x in X_test:
            posteriors = []

            for idx, cls in enumerate(self.classes_):
                prior = np.log(self.priors_[idx])
                likelihood = np.sum(np.log(self._gaussian_pdf(idx, x)))

                posterior = prior + likelihood
                posteriors.append(posterior)

            predictions.append(self.classes_[np.argmax(posteriors)])

        return np.asarray(predictions)
     