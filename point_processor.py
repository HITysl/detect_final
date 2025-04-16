import numpy as np
from sklearn.cluster import DBSCAN
from config import CLUSTER_PARAMS, GRID_PARAMS

class PointProcessor:
    def __init__(self):
        self.eps = CLUSTER_PARAMS['eps']
        self.min_samples = CLUSTER_PARAMS['min_samples']

    def cluster_and_filter(self, points):
        clustering = DBSCAN(eps=0.15, min_samples=2).fit(points)
        labels = clustering.labels_
        centers = [points[labels == label].mean(axis=0) if label != -1 else points[labels == label]
                   for label in set(labels)]
        centers = np.vstack([c if c.ndim > 1 else c.reshape(1, -1) for c in centers])
        if len(centers) == 0:
            raise ValueError("No cluster centers found")
        return centers

    def cluster_points(self, points_2d):
        clustering_x = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points_2d[:, [0]])
        clustering_z = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points_2d[:, [1]])
        unique_x = np.unique(clustering_x.labels_)
        unique_z = np.unique(clustering_z.labels_)
        GRID_PARAMS['row_count'] = len(unique_z) - (-1 in unique_z)
        GRID_PARAMS['col_count'] = len(unique_x) - (-1 in unique_x)
        print(f"Detected rows: {GRID_PARAMS['row_count']}, columns: {GRID_PARAMS['col_count']}")
        return clustering_x.labels_, clustering_z.labels_

    def sort_labels(self, points_2d, x_labels, z_labels):
        x_centroids = sorted([(l, points_2d[x_labels == l, 0].mean())
                              for l in np.unique(x_labels) if l != -1], key=lambda x: x[1])
        z_centroids = sorted([(l, points_2d[z_labels == l, 1].mean())
                              for l in np.unique(z_labels) if l != -1], key=lambda x: -x[1])
        x_map = {old: new for new, (old, _) in enumerate(x_centroids)}
        z_map = {old: new for new, (old, _) in enumerate(z_centroids)}
        return (np.array([x_map.get(l, -1) for l in x_labels]),
                np.array([z_map.get(l, -1) for l in z_labels]))
