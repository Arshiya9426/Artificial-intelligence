# Implementation of K-Means Clustering and Cluster Validation
import numpy as np

def euclidean_distance(a, b):
    """Calculates the Euclidean distance between two vectors."""
    return np.sqrt(np.sum((a - b) ** 2))

class KMeans:
    def __init__(self, k=2, max_iters=100):
        """
        Initializes the K-Means clustering algorithm.
        
        Arguments:
        k         -- The number of clusters to form.
        max_iters -- Maximum number of iterations to run the algorithm.
        """
        self.k = k
        self.max_iters = max_iters
        self.centroids = None
        self.labels = None

    def fit(self, X):
        """
        Computes K-Means clustering on the dataset X.
        
        Arguments:
        X -- A 2D NumPy array representing data instances.
        """
        # Pick random initial centroids from the dataset without replacement
        idx = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[idx]
        
        for _ in range(self.max_iters):
            # --- Cluster Assignment Phase ---
            # Assign each point to the closest centroid
            self.labels = np.array([
                np.argmin([euclidean_distance(x, c) for c in self.centroids]) 
                for x in X
            ])
            
            # --- Update Phase ---
            # Recompute centroids as the mean of assigned data instances
            new_centroids = np.array([
                X[self.labels == i].mean(axis=0) if len(X[self.labels == i]) > 0 else self.centroids[i] 
                for i in range(self.k)
            ])
            
            # Check for convergence (centroids do not change)
            if np.all(self.centroids == new_centroids):
                break
                
            self.centroids = new_centroids
            
        return self.labels


def silhouette_score_simple(X, labels):
    """
    Computes a simplified silhouette score for validation evaluation.
    
    Arguments:
    X      -- The input dataset matrix.
    labels -- Cluster assignments for each point in X.
    """
    scores = []
    unique_labels = set(labels)
    
    for i, x in enumerate(X):
        c_idx = labels[i]
        same_cluster = X[labels == c_idx]
        other_cluster = X[labels != c_idx]
        
        # 1. Intra-cluster distance (a): Mean distance to other points in the same cluster
        if len(same_cluster) > 1:
            a = np.mean([
                euclidean_distance(x, original) 
                for original in same_cluster 
                if not np.array_equal(x, original)
            ])
        else:
            a = 0
            
        # 2. Nearest-cluster distance (b): Mean distance to points in the closest neighboring cluster
        if len(other_cluster) > 0:
            b = np.min([
                np.mean([euclidean_distance(x, o) for o in X[labels == c]]) 
                for c in unique_labels if c != c_idx
            ])
        else:
            b = 0
            
        # 3. Silhouette evaluation coefficient computation
        if max(a, b) == 0:
            scores.append(0)
        else:
            scores.append((b - a) / max(a, b))
            
    return np.mean(scores)


# =====================================================================
# --- Example Execution ---
# =====================================================================
if __name__ == "__main__":
    # Highly separable mock dataset with two clear groups
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    
    km = KMeans(k=2)
    labels = km.fit(X)
    
    score = silhouette_score_simple(X, labels)
    
    print("Computed Cluster Centroids:")
    print(km.centroids)
    print("\nAssigned Labels per Point:")
    print(labels)
    print(f"\nCalculated Silhouette Validation Score: {score:.4f}")
