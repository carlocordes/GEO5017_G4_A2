import numpy as np

from sklearn import svm
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import Bunch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as model_selection
from sklearn.neighbors import KDTree

from scipy.spatial import ConvexHull
import pandas as pd
import os
import matplotlib.pyplot as plt



def tsne_graph(bunch):
    df = pd.DataFrame(bunch['data'], columns = bunch['feature_names'])

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(X_scaled)

    # Plot the t-SNE results
    plt.figure(figsize=(8, 6))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.7, c = bunch['targets'])
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("t-SNE Visualization")
    plt.show()

def visualize_features():
    # Plot
    lidar_df = pd.DataFrame(lidar['data'], columns=lidar['feature_names'])
    pd.plotting.scatter_matrix(lidar_df, c=lidar['targets'], figsize=(15, 15),
                               marker='o', hist_kwds={'bins': 20}, s=60,
                               alpha=.8)
    plt.show()

def read_xyz(filenm):
    """
    :param filenm: name of the file
    :return: points
    """
    points = []
    with open(filenm, 'r') as f_input:
        for line in f_input:
            p = line.split()
            p = [float(i) for i in p]
            points.append(p)
    points = np.array(points).astype(np.float32)
    return points

# Possible features
# Scatter (std) in x,y -> poles
# Scatter (std) in x or y -> fences
# Root density

def root_count(points):
    # Lowest point
    root = points[[np.argmin(points[:, 2])]]
    kd_tree_2d = KDTree(points[:, :2], leaf_size=5)

    radius_root = 1

    count = kd_tree_2d.query_radius(root[:, :2], r = radius_root, count_only = True)
    return count

def hull_area(points):
    hull_2d = ConvexHull(points[:, :2])
    hull_area = hull_2d.volume
    hull_perimeter = hull_2d.area
    shape_index = 1.0 * hull_area / hull_perimeter
    return shape_index

def planarity(l1, l2, l3):
    return (l2 - l3)/l1

def linearity(l1, l2, l3):
    return (l1 - l2)/l1

def spherecity(l1, l2, l3):
    return l3/l1

def find_eigens2(points):
    k_top = max(int(len(points) * 0.005), 100)
    top = points[[np.argmax(points[:, 2])]]

    kd_tree_3d = KDTree(points, leaf_size = 5)
    idx = kd_tree_3d.query(top, k = k_top, return_distance = False)
    idx = np.squeeze(idx, axis = 0)
    neighbours = points[idx, :]
    cov = np.cov(neighbours.T)
    w, _ = np.linalg.eig(cov)

    return np.sort(w)[::-1]

def find_eigens(points):
    """
    Determine eigenvalues from file
    :param points: array as Nx3
    :return: eigenvalues [l1, l2, l3]
    """

    # Center the points for mean to be at the origin
    centered_points = points - np.mean(points, axis=0)

    # Compute covariance matrix
    covariance_matrix = np.cov(centered_points, rowvar=False)

    # Eigenvalues
    eigenvalues, _ = np.linalg.eigh(covariance_matrix)

    return np.sort(eigenvalues)[::-1]

def height(points):
    max_z = np.amax(points[:, 2])
    min_z = np.amin(points[:, 2])
    return max_z - min_z

def root_density(points):
    # Lowest point
    root = points[[np.argmin(points[:, 2])]]
    kd_tree_2d = KDTree(points[:, :2], leaf_size=5)

    radius_root = 1

    count = kd_tree_2d.query_radius(root[:, :2], r = radius_root, count_only = True)
    root_density = 1.0*count[0] / len(points)
    return root_density

if __name__ == '__main__':

    # Define paths
    data_path = os.getcwd() + '/pointclouds-500/pointclouds-500'
    files = sorted([f for f in os.listdir(data_path) if f.endswith('.xyz')])

    # Define metadata
    target_names = np.array(['building', 'car', 'fence', 'pole', 'tree'])
    feature_names = np.array(['Linearity', 'Spherecity', 'Height diff', 'Root count', 'Hull area'])
    targets = np.repeat(np.arange(1, 6), 100)

    feature_count = 5

    # Initialize Bunch
    lidar = Bunch(data=np.zeros((500, feature_count), dtype=float),
                  target_names=target_names,
                  feature_names=feature_names,
                  targets=targets)

    # Process each file
    for i, file in enumerate(files):
        points = read_xyz(os.path.join(data_path, file))
        l1, l2, l3 = find_eigens2(points)

        # Compute & push features
        entry = [linearity(l1, l2, l3), spherecity(l1, l2, l3), height(points), root_density(points), hull_area(points)]
        lidar['data'][i] = entry

    # Split in training & testing data
    ratio = 0.6
    X_train, X_test, y_train, y_test = model_selection.train_test_split(lidar['data'], lidar['targets'],
                                                                        train_size = ratio, test_size = 1.0 - ratio,
                                                                        random_state = 101
                                                                        )
    # Multi-class SVM Classification
    rbf = svm.SVC(kernel='rbf', gamma=1, C=0.1).fit(X_train, y_train)
    poly = svm.SVC(kernel='poly', degree=5, C=1).fit(X_train, y_train)

    poly_pred = poly.predict(X_test)
    rbf_pred = rbf.predict(X_test)

    poly_f1 = f1_score(y_test, poly_pred, average='weighted')
    rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')

    print("Test set score (RBF): {:.5f}".format(np.mean(rbf_pred == y_test)))
    #print("F1 score (RBF): {:.5f}".format(rbf_f1))
    print("Test set score (POLY): {:.5f}".format(np.mean(poly_pred == y_test)))
    #print("F1 score (POLY): {:.5f}".format(poly_f1))



    #visualize_features()
    print(confusion_matrix(y_true = y_test, y_pred = rbf_pred))
    #tsne_graph(lidar)

    #TODO:
    # Random Forest Classification
    # Learning Curve

