import numpy as np
from itertools import combinations_with_replacement

class LogisticModel:
    def __init__(self, data, labels, degree=0, normalized=True):
        self.data = LogisticModel.prepare_for_training(data, degree, normalized)
        self.degree = degree
        self.normalized = normalized
        self.labels = labels.flatten()
        self.unique_labels = np.unique(self.labels)
        self.num_features = self.data.shape[1]
        self.classifiers = {} 

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y_hat, y, eps=1e-15):
        y_hat = np.clip(y_hat, eps, 1 - eps)
        return -1 * (y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def compute_cost(self, y_hat, y):
        m = y.shape[0]
        return (1/m) * np.sum(self.compute_loss(y_hat, y))

    def compute_gradients(self, labels, labels_pred):
        m = labels.shape[0]
        delta = labels_pred - labels
        dw = (1/m) * np.dot(self.data.T, delta)
        return dw

    def train(self, alpha=0.01, epochs=1000):
        for label in self.unique_labels:
            y_binary = (self.labels == label).astype(int).reshape(-1, 1)
            w = np.zeros((self.num_features, 1))

            for _ in range(epochs):
                z = np.dot(self.data, w)
                y_hat = self.sigmoid(z)
                dw = self.compute_gradients(y_binary, y_hat)
                w -= alpha * dw

            self.classifiers[label] = w

    def predict_proba(self, X):
        X_prepared = LogisticModel.prepare_for_training(X, self.degree, self.normalized)
        m = X_prepared.shape[0]
        probs = np.zeros((m, len(self.unique_labels)))

        for i, label in enumerate(self.unique_labels):
            w = self.classifiers[label]
            z = np.dot(X_prepared, w)
            probs[:, i] = self.sigmoid(z).flatten()

        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        predictions = np.argmax(probs, axis=1)
        return self.unique_labels[predictions]

    @staticmethod
    def normalize(features):
        features_normalized = np.copy(features).astype(float)
        features_mean = np.mean(features, 0)
        features_deviation = np.std(features, 0)

        if features.shape[0] > 1:
            features_normalized -= features_mean

        features_normalized /= (features_deviation + 1e-8)
        return features_normalized

    @staticmethod
    def generate_polynomials(dataset, degree, normalized=False):
        n = dataset.shape[1]
        poly_features = []

        for degree in range(1, degree + 1):
            for i in combinations_with_replacement(range(n), degree):
                new_feature = np.prod(dataset[:, i], axis=1).reshape(-1, 1)
                poly_features.append(new_feature)

        polynomials = np.hstack(poly_features)

        if normalized:
            polynomials = LogisticModel.normalize(polynomials)

        return polynomials

    @staticmethod
    def prepare_for_training(data, degree=0, normalized=True):
        num_examples = data.shape[0]
        data_copy = np.copy(data)

        if normalized:
            data_copy = LogisticModel.normalize(data_copy)

        if degree > 0:
            polynomials = LogisticModel.generate_polynomials(data_copy, degree, normalized)
            data_copy = np.hstack((data_copy, polynomials))

        data_copy = np.hstack((np.ones((num_examples, 1)), data_copy))
        return data_copy
    
