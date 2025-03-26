import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.spatial import ConvexHull

from sklearn import svm
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import Bunch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import sklearn.model_selection as model_selection
from sklearn.neighbors import KDTree
from sklearn.ensemble import RandomForestClassifier

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

def height_diff(points):
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
                                                                        train_size = ratio)
    # Multi-class SVM Classification
    model = svm.SVC(kernel = kernel, gamma=1, C=0.1).fit(X_train, y_train)
    svm_pred = model.predict(X_test)

    #rbf_f1 = f1_score(y_test, rbf_pred, average='weighted')

    score = np.mean(svm_pred == y_test)

    if print_cf:
        # Compute the confusion matrix
        cm = confusion_matrix(y_test, svm_pred)
        #print(cm)

        # Plot the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=lidar['target_names'])
        disp.plot(cmap=plt.cm.Blues)
        plt.title('Confusion Matrix SVM ' + kernel)
        plt.show()

    return score

def perform_rf(bunch, ratio, kernel, print_cf = True):
    """
    Performs Random Forest Classification at given training ratio
    :param bunch: data to perform RF on
    :param ratio: training ratio of total data set
    :param print_cf: Whether to print confusion matrix
    :return:
    """
    X_train, X_test, y_train, y_test = model_selection.train_test_split(bunch['data'], bunch['targets'],
                                                                        train_size = ratio)

    clf = RandomForestClassifier(n_estimators=100, criterion= kernel)
    clf.fit(X_train, y_train)
    rf_pred = clf.predict(X_test)
    score = np.mean(rf_pred == y_test)

    if print_cf:
        print(confusion_matrix(y_true = y_test, y_pred = rf_pred))

    return score

def out_learning_rate(bunch, method, kernel = False):

    ratios = np.linspace(0.1, 0.95, 18)
    set_scores = []
    for ratio in ratios:
        set_value = 0
        smoothing = 10
        for i in range(smoothing):
            set_value += method(bunch, ratio, kernel, False)
        set_value /= smoothing
        set_scores.append(set_value)

    # Send to plot
    plt.plot(ratios, set_scores, label = kernel)

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
        entry = [linearity(l1, l2, l3), spherecity(l1, l2, l3), height_diff(points),
                 root_density(points), hull_area(points)]
        data_bunch['data'][i] = entry

    return data_bunch

def J_SwSb(data_bunch):
    data = data_bunch['data']
    labels = data_bunch['targets']
    feature_count = len(data_bunch['feature_names'])

    overall_mean = np.mean(data, axis=0)

    Sw = np.zeros((feature_count, feature_count))
    Sb = np.zeros((feature_count, feature_count))

    for c in range(1, 6):
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
        return 0, 0, 0

    linearity = (l1 - l2) / l1
    planarity = (l2 - l3) / l1
    sphericity = l3 / l1
    return linearity, planarity, sphericity

def height(points):
    z = points[:,2]
    height_mean = np.mean(z)
    height_var = np.var(z)
    height_range = np.max(z) - np.min(z)
    return height_mean, height_var, height_range

def classify_mj(pathname):

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
    for i, file in enumerate(files):
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

    J_values = J_feature(data_bunch)
    features_desc = sorted(J_values.items(), key=lambda x: x[1], reverse=True)

    # Select the top 4 features based on the J-values
    top_4_features = [features_desc[i][0] for i in range(4)]
    top_4_feature_indices = [data_bunch['feature_names'].tolist().index(f) for f in top_4_features]

    final_bunch = Bunch(
        data=data_bunch['data'][:, top_4_feature_indices],
        target_names=data_bunch['target_names'],
        feature_names=top_4_features,
        targets=data_bunch['targets']
    )
    #print('Final features selected based on J value:')
    #print(features_desc[0:3])
    return final_bunch


if __name__ == '__main__':

    # Classify
    lidar = classify_mj('/pointclouds-500/pointclouds-500')

    # Optimize hyperparameters


    # Single SMV Run
    #score = perform_svm(lidar, 0.7, 'poly', False)
    #print("Training Set Score: {:.2f}".format(score))
    """
    # Multiple SVM Runs
    out_learning_rate(lidar, perform_svm, 'poly')
    out_learning_rate(lidar, perform_svm, 'linear')
    out_learning_rate(lidar, perform_svm, 'sigmoid')
    out_learning_rate(lidar, perform_svm, 'rbf')
    """


    #visualize_features()
    #tsne_graph(lidar)

    # Single RF Run
    #rf_score = perform_rf(lidar, 0.7, 'entropy', True)
    #print("Random Forest Training Set Score: {:.2}".format(rf_score))

    # Multiple RF Runs
    #out_learning_rate(lidar, perform_rf, 'entropy')
    #out_learning_rate(lidar, perform_rf, 'gini')


    """
    plt.title('Learning Curve')
    plt.xlabel('Training Set Ratio')
    plt.ylabel('Training Set Score')
    plt.legend()
    plt.grid()
    plt.show()
    """
    #TODO:
    # Hyperparamter Tuning (A2 Intro slides)