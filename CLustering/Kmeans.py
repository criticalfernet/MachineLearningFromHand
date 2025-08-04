import numpy as np

class KmeansModel:
    def __init__(self,data,groups):
        self.data = data
        self.groups = groups
        self.centroids = []

    def initialize_centroids(self):
        i = np.random.choice(self.data.shape[0],size=self.groups)
        self.centroids = self.data[i]

    def assign_clusters(self,centroids):
        diff = self.data[:,np.newaxis,:] - centroids[np.newaxis,:,:]
        distances = np.linalg.norm(diff,axis=2)
        return np.argmin(distances,axis=1)

    def update_centroids(self,labels):
        new_c = np.zeros((self.groups,self.data.shape[1]))
        for i in range(self.groups):
            points = self.data[labels == i]
            if len(points) > 0:
                new_c[i] = points.mean(axis=0)
            else:
                new_c[i] = self.data[np.random.choice(self.data.shape[0])]
        self.centroids = new_c


    def train(self,epochs):
        self.initialize_centroids()

        for _ in range(epochs):
            labels = self.assign_clusters(self.centroids)
            self.update_centroids(labels)

        labels = self.assign_clusters(self.centroids)
        return self.centroids, labels


