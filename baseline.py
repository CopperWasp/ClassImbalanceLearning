import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections
from sklearn.model_selection import GridSearchCV
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
import time
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold, StratifiedKFold

# Classifier Libraries
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import collections
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, RandomizedSearchCV


import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
import matplotlib.patches as mpatches
from imblearn.under_sampling import NearMiss

# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
import warnings

import imblearn

def get_equal_sample(data, label):
    sample_size = min(len(data.loc[data[label] == 1]), len(data.loc[data['label'] == 0]))
    return data.groupby(label).apply(lambda x: x.sample(sample_size))


def plot_feature_correlations(data):
    sample = get_equal_sample(data, 'label')
    sub_sample_corr = sample.corr()
    plt.figure(figsize = (24,20))
    sns.heatmap(sub_sample_corr[(sub_sample_corr >= 0.2) | (sub_sample_corr <= -0.2)], cmap='coolwarm_r', annot_kws={'size':30})
    return plt


def get_numeric_features(data):
    return data.select_dtypes(include=['float64', 'int64'])


def cluster_visualize(data):
    sample = get_equal_sample(data, 'label')
    X = sample.drop('label', axis=1)
    y = sample['label']
    
    X_reduced_tsne = TSNE(n_components=2, random_state=42).fit_transform(X.values)
    X_reduced_pca = PCA(n_components=2, random_state=42).fit_transform(X.values)
    X_reduced_svd = TruncatedSVD(n_components=2, algorithm='randomized', random_state=42).fit_transform(X.values)

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24,6))
    f.suptitle('Clusters using Dimensionality Reduction', fontsize=14)
    blue_patch = mpatches.Patch(color='#0A0AFF', label='No Fraud')
    red_patch = mpatches.Patch(color='#AF0000', label='Fraud')
    # t-SNE scatter plot
    ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
    ax1.scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
    ax1.set_title('t-SNE', fontsize=14)
    ax1.grid(True)
    ax1.legend(handles=[blue_patch, red_patch])

    # PCA scatter plot
    ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
    ax2.scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
    ax2.set_title('PCA', fontsize=14)
    ax2.grid(True)
    ax2.legend(handles=[blue_patch, red_patch])

    # TruncatedSVD scatter plot
    ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 0), cmap='coolwarm', label='No Fraud', linewidths=2)
    ax3.scatter(X_reduced_svd[:,0], X_reduced_svd[:,1], c=(y == 1), cmap='coolwarm', label='Fraud', linewidths=2)
    ax3.set_title('Truncated SVD', fontsize=14)
    ax3.grid(True)
    ax3.legend(handles=[blue_patch, red_patch])

    return plt


# https://imbalanced-learn.readthedocs.io/en/stable/api.html#module-imblearn.over_sampling
def cv_grid_search(sampling, X, y, model_indices=False):
    log_reg_params = {"penalty": ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    knears_params = {"n_neighbors": list(range(2,5,1)), 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
    svc_params = {'C': [0.5, 0.7, 0.9, 1], 'kernel': ['rbf', 'poly', 'sigmoid', 'linear']}
    tree_params = {"criterion": ["gini", "entropy"], "max_depth": list(range(2,4,1)), "min_samples_leaf": list(range(5,7,1))}
    
    grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params)
    grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params)
    grid_svc = GridSearchCV(SVC(), svc_params)
    grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params)
    
    models = np.array([grid_log_reg, grid_knears, grid_svc, grid_tree])
    best_ests = []
    if model_indices:
        models = models[model_indices]
    
    for model in models:
        print("Model: " + str(model))
        # prepare initial train and test
        splitter = StratifiedKFold(n_splits=5, random_state=None, shuffle=False)
        for train_index, test_index in splitter.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
        # Turn into an array
        X_train, X_test = X_train.values, X_test.values 
        y_train, y_test = y_train.values, y_test.values
    
        # List to append the score and then find the average
        accuracy_lst, precision_lst = [], []
        recall_lst, f1_lst = [], []
        auc_lst = []
    
        for train, test in splitter.split(X_train, y_train):
            pipeline = imbalanced_make_pipeline(sampling, model)
            model = pipeline.fit(X_train[train], y_train[train])
            best_est = model.best_estimator_
            prediction = best_est.predict(X_train[test])
    
            accuracy_lst.append(pipeline.score(X_train[test], y_train[test]))
            precision_lst.append(precision_score(y_train[test], prediction))
            recall_lst.append(recall_score(y_train[test], prediction))
            f1_lst.append(f1_score(y_train[test], prediction))
            auc_lst.append(roc_auc_score(y_train[test], prediction))
    
    
        print("accuracy: {}".format(np.mean(accuracy_lst)))
        print("precision: {}".format(np.mean(precision_lst)))
        print("recall: {}".format(np.mean(recall_lst)))
        print("f1: {}".format(np.mean(f1_lst)))
        print("AUC: {}".format(np.mean(auc_lst)))
    
        best_ests.append(best_est)
    return best_ests
    


