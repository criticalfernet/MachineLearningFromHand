import numpy as np
import pandas as pd

data = pd.read_csv('fashion.csv').head(5000)

train_y = data["label"].to_numpy()
train_x = data.drop(columns=["label"]).to_numpy()

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return (x > 0).astype(float)

def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    eps = 1e-8
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

def initialize(input_size=784, hidden_size=25, output_size=10):
    W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2. / input_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2. / hidden_size)
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def compute_loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def backward(X, y, Z1, A1, Z2, A2, W2):
    m = X.shape[0]

    dZ2 = A2 - y  # softmax + cross-entropy
    dW2 = (A1.T @ dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * d_relu(Z1)
    dW1 = (X.T @ dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    return dW1, db1, dW2, db2


def train(X, y, hidden_size=25, lr=0.1, epochs=1000):
    input_size = X.shape[1]
    output_size = y.shape[1]

    W1, b1, W2, b2 = initialize(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        Z1, A1, Z2, A2 = forward(X, W1, b1, W2, b2)
        loss = cross_entropy(y, A2)

        dW1, db1, dW2, db2 = backward(X, y, Z1, A1, Z2, A2, W2)

        W1 -= lr * dW1
        b1 -= lr * db1
        W2 -= lr * dW2
        b2 -= lr * db2

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    return W1, b1, W2, b2


def predict(X, W1, b1, W2, b2):
    _, _, _, A2 = forward(X, W1, b1, W2, b2)
    return np.argmax(A2, axis=1)


def one_hot_encode(y, num_classes=10):
    y = y.astype(int)
    one_hot = np.zeros((y.size, num_classes))
    one_hot[np.arange(y.size), y] = 1
    return one_hot



X = train_x /255
y = train_y

y_onehot = one_hot_encode(y)

W1, b1, W2, b2 = train(X, y_onehot, hidden_size=25, lr=0.1, epochs=5000)

# Predict
preds = predict(X, W1, b1, W2, b2)
y_labels = np.argmax(y_onehot, axis=1)
accuracy = np.mean(preds == y_labels)
print("Training accuracy:", accuracy*100,"%")




