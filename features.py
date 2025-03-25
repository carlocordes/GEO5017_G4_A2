import math
import pandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.utils import Bunch
from sklearn.neighbors import KDTree
from sklearn import svm

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

import sklearn.model_selection as model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from scipy.spatial import ConvexHull
from tqdm import tqdm

import os
from os.path import exists, join
from os import listdir

def eigenvalue_derivatives(points, point, kdtree, k):  
    _, idx = kdtree.query(point, k=k)
    neighbours = points[np.unique(idx)] 
    cov_matrix = np.cov(neighbours.T)
    eigen_values, _ = np.linalg.eig(cov_matrix)
    eigen_values = np.sort(eigen_values)[::-1]

    l1 = eigen_values[0]
    l2 = eigen_values[1]
    l3 = eigen_values[2]

    if l1 == 0:
       return 0,0,0
    
    linearity = (l1 - l2) / l1
    planarity = (l2 - l3) / l1
    sphericity= l3 / l1
    return linearity, planarity, sphericity

def height(points):
    z = points[:,2]
    height_mean = np.mean(z)
    height_var = np.var(z)
    height_range = np.max(z) - np.min(z)
    return height_mean, height_var, height_range

def root_density(points):
    # Lowest point
    root = points[[np.argmin(points[:, 2])]]
    kd_tree_2d = KDTree(points[:, :2], leaf_size=5)

    radius_root = 1

    count = kd_tree_2d.query_radius(root[:, :2], r = radius_root, count_only = True)
    root_density = 1.0*count[0] / len(points)
    return root_density

def hull_area(points):
    hull_2d = ConvexHull(points[:, :2])
    hull_area = hull_2d.volume
    hull_perimeter = hull_2d.area
    shape_index = 1.0 * hull_area / hull_perimeter
    return shape_index

def tsne_visualisation(data_bunch):
    df = pd.DataFrame(data_bunch['data'], columns = data_bunch['feature_names'])

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Prepare the subplots for pairwise feature combinations
    feature_names = data_bunch['feature_names']
    feature_count = len(feature_names)
    n_plots = (feature_count * (feature_count - 1)) // 2
    n_cols = int(np.ceil(np.sqrt(n_plots)))
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 5))
    axes = axes.ravel()

    plot_idx = 0
    for i in range(feature_count):
        for j in range(i + 1, feature_count):
            X_pair = X_scaled[:, [i, j]]
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            X_embedded = tsne.fit_transform(X_pair)

            ax = axes[plot_idx]
            scatter = ax.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=0.7, c=data_bunch['targets'])
            ax.set_xlabel(f"{feature_names[i]}")
            ax.set_ylabel(f"{feature_names[j]}")
            ax.set_aspect('equal')
            plot_idx += 1

    for i in range(plot_idx, len(axes)):
        fig.delaxes(axes[i])
    
    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.tight_layout()
    plt.show()

def read_xyz(filenm):
    points = []
    with open(filenm, 'r') as f_input:
        for line in f_input:
            p = line.split()
            p = [float(i) for i in p]
            points.append(p)
    points = np.array(points).astype(np.float32)
    return points

def SVM(data_bunch, ratio, kernel='rbf', print_cf=True):
    X_train, X_test, y_train, y_test = model_selection.train_test_split(data_bunch['data'], data_bunch['targets'],
                                                                        train_size = ratio,
                                                                        )
    rbf = svm.SVC(kernel = kernel, gamma=1, C=0.1).fit(X_train, y_train)
    rbf_pred = rbf.predict(X_test)
    pass

def J_SwSb(data_bunch):
    data = data_bunch['data']
    labels = data_bunch['targets']
    feature_count = len(data_bunch['feature_names'])

    overall_mean = np.mean(data, axis=0)

    Sw = np.zeros((feature_count, feature_count))
    Sb = np.zeros((feature_count, feature_count))

    for c in range(1,6):
        class_data = data[labels == c]
        class_mean = np.mean(class_data, axis=0)

        Sw += np.dot((class_data - class_mean).T, (class_data - class_mean))

        num_samples = class_data.shape[0]
        mean_diff = (class_mean - overall_mean).reshape(feature_count, 1)
        Sb += num_samples * np.dot(mean_diff, mean_diff.T)
     
    J = np.trace(Sb) / np.trace(Sw) if np.trace(Sw) != 0 else np.inf
    return J

def J_feature(data_bunch):
    data = data_bunch['data']
    labels = data_bunch['targets']
    feature_names = data_bunch['feature_names']

    J_values = {}

    for i, feature in enumerate(feature_names):
        feature_data = data[:, i].reshape(-1, 1)  # Select only one feature

        # Create a temporary Bunch with a single feature
        temp_bunch = {'data': feature_data, 'targets': labels, 'feature_names': [feature]}

        J = J_SwSb(temp_bunch)
        J_values[feature] = J

    return J_values

def classify(pathname):

    # Define paths
    data_path = os.getcwd() + pathname
    files = sorted([f for f in os.listdir(data_path) if f.endswith('.xyz')])

    # Define metadata
    target_names = np.array(['building', 'car', 'fence', 'pole', 'tree'])
    feature_names = np.array(['Top Linearity', 'Top Planarity', 'Top Linearity', 
                              'Btm Linearity', 'Btm Planarity', 'Btm Linearity', 
                              'Mean Height', 'Height Variance', 'Height Range',
                              'Hull Area', 'Root Density'])
    targets = np.repeat(np.arange(1, 6), 100)

    feature_count = feature_names.size

    # Initialize Bunch
    data_bunch = Bunch(data=np.zeros((500, feature_count), dtype=float),
                  target_names=target_names,
                  feature_names=feature_names,
                  targets=targets)

    # Gather feature values
    for i, file in enumerate(tqdm(files, desc='Processing .xyz files', unit='file')):
        points = read_xyz(os.path.join(data_path, file))
        kdtree = KDTree(points)
        k = max(int(len(points) * 0.005), 100)

        top_most = points[[np.argmax(points[:, 2])]]
        btm_most = points[[np.argmin(points[:, 2])]]
        top_linearity, top_planarity, top_sphericity = eigenvalue_derivatives(points, top_most, kdtree, k)
        btm_linearity, btm_planarity, btm_sphericity = eigenvalue_derivatives(points, btm_most, kdtree, k)
        hull = hull_area(points)
        density = root_density(points)

        height_mean, height_var, height_range = height(points)

        # Compute & push features
        entry = [top_linearity, top_planarity, top_sphericity, 
                 btm_linearity, btm_planarity, btm_sphericity, 
                 height_mean, height_var, height_range, 
                 hull, density]
        data_bunch['data'][i] = np.array(entry).flatten()

    return data_bunch

if __name__ == '__main__':
    lidar = classify('/pointclouds-500/pointclouds-500')
    J = J_SwSb(lidar)
    print(J_feature(lidar))

    # tsne_visualisation(lidar)

