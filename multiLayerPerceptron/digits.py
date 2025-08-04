import numpy as np
import pandas as pd
from Model import MultiLayerPerceptron

data = pd.read_csv('data/mnist-demo.csv').head(5000)

train_y = data["label"].to_numpy()
train_x = data.drop(columns=["label"]).to_numpy()
train_x = train_x/255

mlp = MultiLayerPerceptron(train_x,train_y,layer_sizes=[784, 25, 10])
mlp.train(0.1,5000)

predictions = mlp.predict(train_x)

y_labels = np.argmax(mlp.labels, axis=1)
accuracy = np.mean(predictions == y_labels)
print("Training accuracy:{:.2f}%".format(accuracy*100))

