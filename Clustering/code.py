import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Kmeans import KmeansModel

data = pd.read_csv('data/iris.csv')
num_examples = data.shape[0]
x_axis = 'petal_length'
y_axis = 'petal_width'

x_train = data[[x_axis, y_axis]].to_numpy().reshape((num_examples, 2))


model = KmeansModel(x_train,3)
centroids, labels = model.train(1000)

k = centroids.shape[0]
for cluster_id in range(k):
        cluster_points = data[labels == cluster_id]
        plt.scatter(
            cluster_points[x_axis],
            cluster_points[y_axis],
            label='Cluster #' + str(cluster_id)
        )

for centroid in centroids:
    plt.scatter(centroid[0], centroid[1], c='black', marker='x')


plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.legend()
plt.show()

