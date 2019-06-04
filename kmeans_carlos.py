import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import pandas as pd

def kMeans(b, k_clusters=2, tolerance=.00001, graph=True):
    # Check if data is a np.array, if not transform it:
    if type(b).__module__ != np.__name__:
        b = np.array(b)
    # Define random centroids:
    centroids = b[np.random.randint(b.shape[0], size=k_clusters), :]
    # Label data based on closest centroid:
    c = distance_matrix(b, centroids)
    labels = np.argmin(c, axis=1)
    # Define new centroids:
    df = pd.DataFrame({'x':b[:,0], 'y':b[:,1], 'label':labels})
    new_c = df.groupby('label').agg('mean')
    while (np.sum((new_c[['x','y']].values - centroids)/centroids * 100.0) > tolerance):
        centroids = new_c[['x','y']].values
        c = distance_matrix(b, centroids)
        labels = np.argmin(c, axis=1)
        df = pd.DataFrame({'x':b[:,0], 'y':b[:,1], 'label':labels})
        new_c = df.groupby('label').agg('mean')
        print(np.sum((new_c[['x','y']].values - centroids)/centroids * 100.0))
    if graph == True:
        plt.scatter(x=b[:,0], y=b[:,1], c=labels)
        plt.show()
    return labels, new_c

# Test of random points:
b = np.random.rand(250,2)
labels, centroids = kMeans(b, k_clusters=8, graph=False)
labels, centroids = kMeans(b, k_clusters=8)

# Test from sklearn blobs:
from sklearn.datasets.samples_generator import make_blobs
X, labels_true = make_blobs(n_samples=3000, centers=8, cluster_std=0.7)
labels, centroids = kMeans(X, k_clusters=7)
