"""
This demo shows how to visualize the designed features. Currently, only 2D feature space visualization is supported.
I use the same data for A2 as my input.
Each .xyz file is initialized as one urban object, from where a feature vector is computed.
6 features are defined to describe an urban object.
Required libraries: numpy, scipy, scikit learn, matplotlib, tqdm 
"""

import math
import matplotlib.pyplot as plt
import numpy as np
from pandas.core.common import random_state
from polyscope_bindings import custom

from sklearn.neighbors import KDTree
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
# from sklearn.model_selection import train_test_split
import sklearn.model_selection as model_selection

from scipy.spatial import ConvexHull
from tqdm import tqdm
from os.path import exists, join
from os import listdir
import os
import pandas as pd
import seaborn as sns


class urban_object:
    """
    Define an urban object
    """
    def __init__(self, filenm):
        """
        Initialize the object
        """
        # obtain the cloud name
        self.cloud_name = filenm.split('/\\')[-1][-7:-4]

        # obtain the cloud ID
        self.cloud_ID = int(self.cloud_name)

        # obtain the label
        self.label = math.floor(1.0*self.cloud_ID/100)

        # obtain the points
        self.points = read_xyz(filenm)

        # initialize the feature vector
        self.feature = []

    def compute_features(self):
        """
        Compute the features, here we provide two example features. You're encouraged to design your own features
        """
        ### Feature 1: height of the object
        # calculate the height
        height = np.amax(self.points[:, 2])
        self.feature.append(height)


        ### Feature 2: root density
        # get the root point and top point
        root = self.points[[np.argmin(self.points[:, 2])]]
        top = self.points[[np.argmax(self.points[:, 2])]]

        # construct the 2D and 3D kd tree
        kd_tree_2d = KDTree(self.points[:, :2], leaf_size=5)
        kd_tree_3d = KDTree(self.points, leaf_size=5)

        # compute the root point planar density
        radius_root = 0.2
        count = kd_tree_2d.query_radius(root[:, :2], r=radius_root, count_only=True)
        root_density = 1.0*count[0] / len(self.points)
        self.feature.append(root_density)


        ### Feature 3: area
        # compute the 2D footprint and calculate its area
        hull_2d = ConvexHull(self.points[:, :2])
        hull_area = hull_2d.volume
        self.feature.append(hull_area)

        ### Feature 4: shape index
        # get the hull shape index
        hull_perimeter = hull_2d.area
        shape_index = 1.0 * hull_area / hull_perimeter
        self.feature.append(shape_index)


        ### Feature 5 and 6: linearity and sphericity
        # obtain the point cluster near the top area
        k_top = max(int(len(self.points) * 0.005), 100)
        idx = kd_tree_3d.query(top, k=k_top, return_distance=False)
        idx = np.squeeze(idx, axis=0)
        neighbours = self.points[idx, :]

        # obtain the covariance matrix of the top points
        cov = np.cov(neighbours.T)
        w, _ = np.linalg.eig(cov)   #Eigenvalues of the covariance matrix
        w.sort()

        # calculate the linearity and sphericity and planarity
        linearity = (w[2]-w[1]) / (w[2] + 1e-5)
        sphericity = w[0] / (w[2] + 1e-5)
        planarity = (w[1]-w[0]) / (w[2] + 1e-5)     # add planarity
        self.feature += [linearity, sphericity, planarity]

        # TODO
        ### Feature 8: Vertical Distribution Ratio (VDR)
        # Compute vertical distribution ratio (VDR)
        z = self.points[:, 2]  # extract z values
        z_min = np.min(z)
        z_max = np.max(z)
        z_mid = (z_min + z_max) / 2

        # Count points above the mid-height
        points_above = np.sum(z > z_mid)
        vdr = points_above / len(z)

        # Append to feature vector
        self.feature.append(vdr)


def read_xyz(filenm):
    """
    Reading points
        filenm: the file name
    """
    points = []
    with open(filenm, 'r') as f_input:
        for line in f_input:
            p = line.split()
            p = [float(i) for i in p]
            points.append(p)
    points = np.array(points).astype(np.float32)
    return points


def feature_preparation(data_path):
    """
    Prepare features of the input point cloud objects
        data_path: the path to read data
    """
    # check if the current data file exist
    data_file = 'data.txt'
    if exists(data_file):
        return

    # obtain the files in the folder
    files = sorted(listdir(data_path))

    # initialize the data
    input_data = []

    # retrieve each data object and obtain the feature vector
    for file_i in tqdm(files, total=len(files)):
        # obtain the file name
        file_name = join(data_path, file_i)

        # read data
        i_object = urban_object(filenm=file_name)

        # calculate features
        i_object.compute_features()

        # add the data to the list
        i_data = [i_object.cloud_ID, i_object.label] + i_object.feature
        input_data += [i_data]

    ##### Add the following line to check the number of files #####
    print(f"len(files): {len(files)}")

    # transform the output data
    outputs = np.array(input_data).astype(np.float32)

    #TODO
    # write the output to a local file
    data_header = 'ID,label,height,root_density,area,shape_index,linearity,sphericity,planarity,vdr'
    np.savetxt(data_file, outputs, fmt='%10.5f', delimiter=',', newline='\n', header=data_header)


def data_loading(data_file='data.txt'):
    """
    Read the data with features from the data file
        data_file: the local file to read data with features and labels
    """
    # load data
    data = np.loadtxt(data_file, dtype=np.float32, delimiter=',', comments='#')

    # extract object ID, feature X and label Y
    ID = data[:, 0].astype(np.int32)
    y = data[:, 1].astype(np.int32)
    X = data[:, 2:].astype(np.float32)

    return ID, X, y


def feature_visualization(X):
    """
    Visualize the features
        X: input features. This assumes classes are stored in a sequential manner
    """
    # initialize a plot
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.title("feature subset visualization of 5 classes", fontsize="small")

    # define the labels and corresponding colors
    colors = ['firebrick', 'grey', 'darkorange', 'dodgerblue', 'olivedrab']
    labels = ['building', 'car', 'fence', 'pole', 'tree']

    #TODO
    # plot the data with first two features
    for i in range(5):
        ax.scatter(X[100*i:100*(i+1), 3], X[100*i:100*(i+1), 4], marker="o", c=colors[i], edgecolor="k", label=labels[i])

    # show the figure with labels
    """
    Replace the axis labels with your own feature names
    """
    ax.set_xlabel('x1:root density')
    ax.set_ylabel('x2:area')
    ax.legend()
    plt.show()

def feature_vis_scatter_matrix(X, y):
    """
    Create a scatter matrix for all features.
    X: features (numpy array)
    y: labels (numpy array)
    """

    # Define column names (adjust if needed)
    feature_names = ['height', 'root_density', 'area', 'shape_index',
                     'linearity', 'sphericity', 'planarity','vdr']

    # Create dataframe
    df = pd.DataFrame(X, columns=feature_names)
    df['label'] = y

    # Replace class IDs with names
    label_map = {0: 'building', 1: 'car', 2: 'fence', 3: 'pole', 4: 'tree'}
    df['label'] = df['label'].map(label_map)

    # Define custom color palette
    custom_palette = {'building': 'firebrick', 'car': 'grey', 'fence': 'darkorange', 'pole': 'dodgerblue', 'tree': 'olivedrab'}

    # Use seaborn to plot pairwise relationships
    sns.pairplot(df, hue='label', palette=custom_palette, corner=True, plot_kws={'alpha': 0.5, 's': 20})
    plt.suptitle('Scatter Matrix of Features by Class', y=1.02)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    labels = ['building', 'car', 'fence', 'pole', 'tree']

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels,
                cmap='Blues', cbar=False)
    plt.title(f"Confusion Matrix: {title}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def SVM_classification(X, y):
    """
    Conduct SVM classification
        X: features
        y: labels
    """
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.6, test_size=0.4, random_state=101)


    # SVM with Polynomial Kernel and RBF Kernel
    print("=====Running SVM with Polynomial Kernel and RBF Kernel=====")

    poly = svm.SVC(kernel='poly', degree=5, C=10).fit(X_train, y_train)
    rbf = svm.SVC(kernel='rbf', gamma=0.5, C=2).fit(X_train, y_train)

    poly_pred = poly.predict(X_test)
    rbf_pred = rbf.predict(X_test)

    poly_accuracy = accuracy_score(y_test, poly_pred)
    poly_f1 = f1_score(y_test, poly_pred, average='weighted')
    print(f"Hyperparameters (Polynomial Kernel): degree={poly.degree}, C={poly.C}")
    print("Accuracy (Polynomial Kernel): ", "%.2f" % (poly_accuracy * 100))
    print("F1 (Polynomial Kernel): ", "%.2f" % (poly_f1 * 100))

    rbf_accuracy = accuracy_score(y_test, rbf_pred)
    rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')
    print(f"Hyperparameters (RBF Kernel): gamma={rbf.gamma}, C={rbf.C}")
    print("Accuracy (RBF Kernel): ", "%.2f" % (rbf_accuracy * 100))
    print(f"F1 (RBF Kernel): {rbf_f1 * 100:.2f}")

    plot_confusion_matrix(y_test, poly_pred, "Polynomial Kernel")  # plot confusion matrix
    plot_confusion_matrix(y_test, rbf_pred, "RBF Kernel")       # plot confusion matrix

def SVM_rbf_grid_search(X, y):
    """
    Grid search for best C and gamma in RBF SVM
    """
    C_values = [0.01, 0.1, 1, 10]
    gamma_values = [0.01, 0.1, 1]

    best_f1 = 0     # initialize best F1 score
    best_params = {}

    print("Starting Grid Search for RBF SVM...\n")
    for C in C_values:
        for gamma in gamma_values:
            # 1. Train-test split
            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                X, y, train_size=0.6, random_state=42)

            # 2. Create model with current parameters
            model = svm.SVC(kernel='rbf', C=C, gamma=gamma)

            # 3. Train & predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # 4. Evaluate
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            print(f"C={C}, gamma={gamma} -> Accuracy: {acc:.2f}, F1 Score: {f1:.2f}")

            if f1 > best_f1:
                best_f1 = f1
                best_params = {'C': C, 'gamma': gamma}

    print("\nBest Hyperparameters (RBF SVM):")
    print(f"Best F1 Score: {best_f1:.2f}")
    print(f"Best Params: {best_params}")

def SVM_poly_grid_search(X, y):
    """
    Grid search for best C and gamma in Poly SVM
    """
    C_values = [0.01, 0.1, 1, 10]
    degree_values = [2, 3, 4]

    best_f1 = 0     # initialize best F1 score
    best_params = {}

    print("Starting Grid Search for Poly SVM...\n")
    for C in C_values:
        for degree in degree_values:
            # 1. Train-test split
            X_train, X_test, y_train, y_test = model_selection.train_test_split(
                X, y, train_size=0.6, random_state=42)

            # 2. Create model with current parameters
            model = svm.SVC(kernel='poly', C=C, degree=degree)

            # 3. Train & predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # 4. Evaluate
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            print(f"C={C}, degree={degree} -> Accuracy: {acc:.2f}, F1 Score: {f1:.2f}")

            if f1 > best_f1:
                best_f1 = f1
                best_params = {'C': C, 'degree': degree}

    print("\nBest Hyperparameters (Poly SVM):")
    print(f"Best F1 Score: {best_f1:.2f}")
    print(f"Best Params: {best_params}")


def RF_classification(X, y):
    """
    Conduct RF classification
        X: features
        y: labels
    """
    pass


if __name__=='__main__':
    # specify the data folder
    """"Here you need to specify your own path"""
    path = os.getcwd() + "/pointclouds-500/pointclouds-500"

    # conduct feature preparation
    print('Start preparing features')
    feature_preparation(data_path=path)

    # load the data
    print('Start loading data from the local file')
    ID, X, y = data_loading()

    # # visualize features using 2D plot for the first two features
    # print('Visualize the features')
    # feature_visualization(X=X)

    # Visualize features using scatter matrix
    # print('Visualize the features using scatter matrix')
    # feature_vis_scatter_matrix(X, y)

    # SVM classification
    print('Start SVM classification')
    SVM_classification(X, y)

    # Hyperparameter Tuning using Grid search for SVM (Polynomial Kernel)
    print('\nStart Grid Search for SVM (Polynomial Kernel)')
    SVM_poly_grid_search(X, y)

    # Hyperparameter Tuning using Grid search for SVM (RBF Kernel)
    print('\nStart Grid Search for SVM (RBF Kernel)')
    SVM_rbf_grid_search(X, y)


    # RF classification
    print('Start RF classification')
    RF_classification(X, y)