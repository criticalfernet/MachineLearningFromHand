import numpy as np
from itertools import combinations_with_replacement

class RegressionModel:
    def __init__(self, data, labels, degree=0, normalized=True):
        self.data = RegressionModel.prepare_for_training(data, degree, normalized)
        self.labels = labels

        num_features = self.data.shape[1]
        self.weights = np.zeros((num_features, 1))

    def train(self, alpha, num_iterations=500):
        for i in range(num_iterations):
            grad = self.gradient_step()
            self.weights -= alpha*grad

            if i%100 == 0:
                print(f"Epoch {i} , Cost: {self.cost_function()}")

        return self.weights

    def gradient_step(self):
        num_examples = self.data.shape[0]

        predictions = RegressionModel.predict(self.data, self.weights)
        errors = predictions - self.labels

        grad = (1 / num_examples) * np.dot(self.data.T,errors)
        return grad

    def cost_function(self):
        num_examples = self.data.shape[0]

        errors = RegressionModel.predict(self.data, self.weights) - self.labels
        cost = (1 / 2 * num_examples) * np.dot(errors.T , errors)
        return cost[0][0]

    @staticmethod
    def predict(data, theta):
        predictions = np.dot(data,theta)
        return predictions

    @staticmethod
    def normalize(features):
        features_normalized = np.copy(features).astype(float)
        features_mean = np.mean(features, 0)
        features_deviation = np.std(features, 0)

        if features.shape[0] > 1:
            features_normalized -= features_mean

        features_normalized /= features_deviation
        return features_normalized

    @staticmethod
    def generate_polynomials(dataset, degree, normalized=False):
        n = dataset.shape[1]

        poly_features = []

        for deg in range(1, degree + 1):
            for i in combinations_with_replacement(range(n), deg):
                new_feature = np.prod(dataset[:, i], axis=1).reshape(-1, 1)
                poly_features.append(new_feature)


        polynomials = np.hstack(poly_features)

        if normalized:
            polynomials = RegressionModel.normalize(polynomials)

        return polynomials


    @staticmethod
    def prepare_for_training(data, degree=0,normalized=True):
        num_examples = data.shape[0]

        data_copy = np.copy(data)

        if normalized:
            data_copy = RegressionModel.normalize(data_copy)

        if degree > 0:
            polynomials = RegressionModel.generate_polynomials(data_copy, degree, normalized)
            data_copy = np.hstack((data_copy, polynomials))

        data_copy = np.hstack((np.ones((num_examples, 1)), data_copy))

        return data_copy
