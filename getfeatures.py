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


# Possible features
# Scatter (std) in x,y -> poles
# Scatter (std) in x or y -> fences
# Root density

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
    return (l1 - l2)/(l1+ 1e-5)

def spherecity(l1, l2, l3):
    return l3/(l1+ 1e-5)

def find_eigens(points):
    k_top = max(int(len(points) * 0.005), 100)
    top = points[[np.argmax(points[:, 2])]]

    kd_tree_3d = KDTree(points, leaf_size = 5)
    idx = kd_tree_3d.query(top, k = k_top, return_distance = False)
    idx = np.squeeze(idx, axis = 0)
    neighbours = points[idx, :]
    cov = np.cov(neighbours.T)
    w, _ = np.linalg.eig(cov)

    return np.sort(w)[::-1]

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

def perform_svm(bunch, ratio, kernel = 'rbf', print_cf = True):
    """
    Perform SVM
    :param bunch:
    :param ratio:
    :param print_cf: bool, printing confusion matrix
    :return:
    """
    # Split in training & testing data
    X_train, X_test, y_train, y_test = model_selection.train_test_split(bunch['data'], bunch['targets'],
                                                                        train_size = ratio,
                                                                        )
    # Multi-class SVM Classification
    rbf = svm.SVC(kernel = kernel, gamma=1, C=0.1).fit(X_train, y_train)
    rbf_pred = rbf.predict(X_test)

    #rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')

    score = np.mean(rbf_pred == y_test)

    if print_cf:
        print(confusion_matrix(y_true=y_test, y_pred=rbf_pred))

    return score

def out_learning_rate(bunch, method, kernel = False):
    ratios = np.linspace(0.1, 0.95, 18)

    set_scores = []
    for ratio in ratios:
        set_value = 0
        iterations = 20
        for i in range(iterations):
            set_value += method(bunch, ratio, kernel, False)
        set_value /= iterations
        set_scores.append(set_value)

    plt.plot(ratios, set_scores)
    plt.title('Learning Curve')
    plt.xlabel('Training Set Ratio')
    plt.ylabel('Training Set Score')
    plt.grid()
    plt.show()

def classify(pathname):

    # Define paths
    data_path = os.getcwd() + pathname
    files = sorted([f for f in os.listdir(data_path) if f.endswith('.xyz')])

    # Define metadata
    target_names = np.array(['building', 'car', 'fence', 'pole', 'tree'])
    feature_names = np.array(['Linearity', 'Spherecity', 'Height diff', 'Root count', 'Hull area'])
    targets = np.repeat(np.arange(1, 6), 100)

    feature_count = 5

    # Initialize Bunch
    data_bunch = Bunch(data=np.zeros((500, feature_count), dtype=float),
                  target_names=target_names,
                  feature_names=feature_names,
                  targets=targets)

    # Gather feature values
    for i, file in enumerate(files):
        points = read_xyz(os.path.join(data_path, file))
        l1, l2, l3 = find_eigens(points)

        # Compute & push features
        entry = [linearity(l1, l2, l3), spherecity(l1, l2, l3), height(points), root_density(points), hull_area(points)]
        data_bunch['data'][i] = entry

    return data_bunch

if __name__ == '__main__':

    lidar = classify('/pointclouds-500/pointclouds-500')
    # Single Run
    score = perform_svm(lidar, 0.7, 'poly', True)
    print("Training Set Score: {:.2f}".format(score))

    # Multiple runs
    #out_learning_rate(lidar, perform_svm, 'poly')

    #visualize_features()
    #tsne_graph(lidar)

    #TODO:
    # Random Forest Classification
    # Learning Curve
    # Hyperparamter Tuning (A2 Intro slides)