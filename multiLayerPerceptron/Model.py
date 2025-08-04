import numpy as np


class MultiLayerPerceptron:
    def __init__(self, data, labels, layer_sizes):
        self.data = data
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.weights = []
        self.biases = []
        self.labels = MultiLayerPerceptron.one_hot_encode(labels,layer_sizes[-1])
        self.initialize_weights()

    def relu(self, x):
        return np.maximum(0, x)

    def d_relu(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exps / np.sum(exps, axis=1, keepdims=True)

    def cross_entropy(self ,y_pred):
        eps = 1e-8
        return -np.mean(np.sum(self.labels * np.log(y_pred + eps), axis=1))

    def initialize_weights(self):
        for i in range(self.num_layers):
            in_size = self.layer_sizes[i]
            out_size = self.layer_sizes[i + 1]
            W = np.random.randn(in_size, out_size) * np.sqrt(2. / in_size)
            b = np.zeros((1, out_size))
            self.weights.append(W)
            self.biases.append(b)

    def forward(self,data):
        activations = [data]
        pre_activations = []

        for i in range(self.num_layers):
            Z = activations[-1] @ self.weights[i] + self.biases[i]
            pre_activations.append(Z)

            if i == self.num_layers - 1:
                A = self.softmax(Z)
            else:
                A = self.relu(Z)

            activations.append(A)

        return activations, pre_activations

    def backward(self, alpha, activations, pre_activations):
        m = self.data.shape[0]
        grads_W = [None] * self.num_layers
        grads_b = [None] * self.num_layers

        dZ = activations[-1] - self.labels

        for i in reversed(range(self.num_layers)):
            A_prev = activations[i]
            grads_W[i] = (A_prev.T @ dZ) / m
            grads_b[i] = np.sum(dZ, axis=0, keepdims=True) / m

            if i != 0:
                dA_prev = dZ @ self.weights[i].T
                dZ = dA_prev * self.d_relu(pre_activations[i-1])

        for i in range(self.num_layers):
            self.weights[i] -= alpha * grads_W[i]
            self.biases[i] -= alpha * grads_b[i]

    def train(self,alpha, epochs):
        for epoch in range(epochs):
            activations, pre_activations = self.forward(self.data)
            loss = self.cross_entropy(activations[-1])
            self.backward(alpha, activations, pre_activations)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        activations, _ = self.forward(X)
        return np.argmax(activations[-1], axis=1)

    @staticmethod
    def one_hot_encode(y, num_classes):
        y = y.astype(int)
        one_hot = np.zeros((y.size, num_classes))
        one_hot[np.arange(y.size), y] = 1
        return one_hot

