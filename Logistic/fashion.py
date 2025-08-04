import numpy as np
import pandas as pd
from Logistic import LogisticModel

data = pd.read_csv('data/fashion.csv').head(1000)

x_train = data.drop(columns=["label"]).to_numpy()
y_train = data["label"].to_numpy().reshape((-1, 1))
x_train = x_train / 255.0


model = LogisticModel(x_train, y_train,1, False)
model.train(0.05,1000)

# Predict
y_pred = model.predict(x_train)
accuracy = np.mean(y_pred == y_train.flatten())
print(f"Accuracy: {accuracy * 100:.2f}%")
