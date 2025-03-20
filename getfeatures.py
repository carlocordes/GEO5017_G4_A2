import numpy as np
from sklearn.metrics import f1_score

from sklearn.utils import Bunch
from sklearn import svm
import sklearn.model_selection as model_selection

import pandas as pd
import os
import matplotlib.pyplot as plt

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

def planarity(l1, l2, l3):
    return (l2 - l3)/l1

def linearity(l1, l2, l3):
    return (l1 - l2)/l1

def spherecity(l1, l2, l3):
    return l3/l1

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

def get_max_z(points):
    max_z = 0
    for pt in points:
        if pt[2] > max_z:
            max_z = pt[2]

    return max_z

def visualize_features():
        # Plot
        lidar_df = pd.DataFrame(lidar['data'], columns=lidar['feature_names'])
        pd.plotting.scatter_matrix(lidar_df, c=lidar['targets'], figsize=(15, 15),
                                   marker='o', hist_kwds={'bins': 20}, s=60,
                                   alpha=.8)
        plt.show()

# Define paths
data_path = os.getcwd() + '/pointclouds-500/pointclouds-500'
files = sorted([f for f in os.listdir(data_path) if f.endswith('.xyz')])

# Define metadata
target_names = np.array(['building', 'car', 'fence', 'pole', 'tree'])
feature_names = np.array(['Linearity', 'Planarity', 'Spherecity', 'Maximum z'])
targets = np.repeat(np.arange(1, 6), 100)

feature_count = 4

# Initialize Bunch
lidar = Bunch(data=np.zeros((500, feature_count), dtype=float),
              target_names=target_names,
              feature_names=feature_names,
              targets=targets)

# Process each file
for i, file in enumerate(files):
    points = read_xyz(os.path.join(data_path, file))
    l1, l2, l3 = find_eigens(points)

    # Compute & push features
    entry = [linearity(l1, l2, l3), planarity(l1, l2, l3), spherecity(l1, l2, l3), get_max_z(points)]
    lidar['data'][i] = entry
#visualize_features()


# Multi-class SVM Classification
ratio = 0.7
X_train, X_test, y_train, y_test = model_selection.train_test_split(lidar['data'], lidar['targets'],
                                                                    train_size = ratio, test_size=0.3,
                                                                    random_state= 0
                                                                    )

rbf = svm.SVC(kernel='rbf', gamma=0.8, C=0.1).fit(X_train, y_train)
poly = svm.SVC(kernel='poly', degree=5, C=1).fit(X_train, y_train)

poly_pred = poly.predict(X_test)
rbf_pred = rbf.predict(X_test)

poly_f1 = f1_score(y_test, poly_pred, average='weighted')
rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')

print("Test set score (RBF): {:.5f}".format(np.mean(rbf_pred == y_test)))
print("F1 score (RBF): {:.5f}".format(rbf_f1))
print("Test set score (POLY): {:.5f}".format(np.mean(poly_pred == y_test)))
print("F1 score (POLY): {:.5f}".format(poly_f1))

# Random Forest Classification
