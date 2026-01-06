import numpy as np
import random

class KMeans:
    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit_predict(self, X):
        # Randomly initialize centroids
        random_indices = random.sample(range(X.shape[0]), self.n_clusters)
        self.centroids = X[random_indices]

        for _ in range(self.max_iter):
            # Assign clusters
            cluster_group = self.assign_clusters(X)

            # Store old centroids
            old_centroids = self.centroids.copy()

            # Move centroids
            self.centroids = self.move_centroids(X, cluster_group)

            # Check convergence
            if np.allclose(old_centroids, self.centroids):
                break

        return cluster_group

    def assign_clusters(self, X):
        cluster_group = []

        for row in X:
            distances = []
            for centroid in self.centroids:
                distance = np.sqrt(np.dot(row - centroid, row - centroid))
                distances.append(distance)

            cluster_group.append(np.argmin(distances))

        return np.array(cluster_group)

    def move_centroids(self, X, cluster_group):
        new_centroids = []

        for i in range(self.n_clusters):
            points = X[cluster_group == i]
            new_centroids.append(points.mean(axis=0))

        return np.array(new_centroids)
