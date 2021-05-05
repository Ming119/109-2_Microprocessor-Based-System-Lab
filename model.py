'''
model.py

109-2 283647 多媒體技術與應用
    期中專案一   剪刀石頭布

第十二組
    資工二 108590049 符芷琪
    資工二 108590050 李浩銘
'''

# Define DEBUG
DEBUG = False;

# Define PLOT;
PLOT = False;

import util;

import sys, os, cv2, time, glob, joblib;
import numpy as np;
import matplotlib.pyplot as plt;

from datetime import datetime;
from sklearn import svm;
from sklearn.cluster import KMeans;
from sklearn.model_selection import GridSearchCV;
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report;
from sklearn.preprocessing import StandardScaler;
from sklearn.decomposition import PCA;
from sklearn.pipeline import Pipeline;
from scipy.cluster.vq import *;



def dt(): return round(time.time() - t0, 2);

#####
# K-Means clustering
#####
def _KMeans_(features):
    '''
    # cv2.kmeans
    print('\tK-Means Clustering using `cv2.kmeans`');
    criteria = cv2.kmeans(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.1);
    flags = cv2.KMEANS_RANDOM_CENTERS;

    compactness, labels, centers = cv2.kmeans(features, util.K_MEANS_CLUSTERS, None, criteria, 20, flags);

    return centers;
    '''

    '''
    # sklearn.cluster.KMeans
    print('\tK-Means Clustering using `sklearn.cluster.KMeans`');
    kmeans = KMeans(n_clusters=util.K_MEANS_CLUSTERS, random_state=0).fit(features);

    return kmeans.cluster_centers_;
    '''


    # scipy.cluster.vq.kmeans
    print('\tK-Means Clustering using `scipy.cluster.vq.kmeans`');
    centers, variance = kmeans(features, util.K_MEANS_CLUSTERS);

    return centers;



def _calcVector_(path):
    print("+{}s: Calculating Features Vector...".format(dt()));
    centers = np.load(util.K_MEANS_PATH);

    data   = np.float32([]).reshape(0, util.K_MEANS_CLUSTERS);
    labels = np.float32([]);

    for i, folder in enumerate(os.listdir(path)):
        for file in glob.glob(path + folder + "/*.png"):
            img = cv2.imread(file, 0);
            imgFeature = util._SIFT_(img);
            imgVector  = util._FeatureVector_(imgFeature, centers);
            # plt.scatter(imgFeature[:,0], imgFeature[:,1], c='#{}{}{}'.format(i*10,i*20,i*30), marker='+');
            data   = np.append(data, imgVector, axis=0);
            labels = np.append(labels, i);


    print("+{}s: Calculating Features Vector Completed!".format(dt()));

    return (data, labels);


def _SaveCenter_(path):
    print("+{}s: K-Means Clustering...".format(dt()));
    features = np.float32([]).reshape(0, 128);
    labels = [];
    for i, folder in enumerate(os.listdir(path)):
        for file in glob.glob(path + folder + "/*.png"):
            if (DEBUG): print("+{}s: Reading image {}".format(dt(), file));

            img = cv2.imread(file, 0);
            imgFeature = util._SIFT_(img);

            features = np.append(features, imgFeature, axis=0);
            labels.append(i);

    centers = _KMeans_(features);
    print("+{}s: K-Means Clustering Completed!".format(dt()));

    if (PLOT):
        fig = plt.figure();
        ax = fig.add_subplot(projection='3d');

        pca_f = PCA(3).fit_transform(features);
        P = pca_f[:len(labels),:];
        R = pca_f[len(labels):len(labels)*2,:];
        S = pca_f[len(labels)*2:,];
        pca_f = PCA(3).fit_transform(centers);

        ax.scatter(P[:256,0], P[:256,1], P[:256,2], c='r', marker='+', label='PAPRE', alpha=0.3);
        ax.scatter(R[:256,0], R[:256,1], R[:256,2], c='g', marker='+', label='ROCK', alpha=0.3);
        ax.scatter(S[:256,0], S[:256,1], S[:256,2], c='b', marker='+', label='SCISSORS', alpha=0.3);
        ax.scatter(pca_f[:,0], pca_f[:,1], pca_f[:,2], c='y', alpha=0.5);
        plt.show();

    np.save(util.K_MEANS_PATH, centers);
    print("+{}s: Saved K-Means data at {}.".format(dt(), util.K_MEANS_PATH));



def _SVMTrain_(data, labels, GRID=False):
    if (PLOT):
        plt.hist(data);
        plt.legend(loc='best');
        plt.show();

    stdSlr = StandardScaler().fit(data);
    data = stdSlr.transform(data);

    if (PLOT):
        plt.hist(data);
        plt.legend(loc='best');
        plt.show();

    if (GRID):
        t0_train = time.time();
        print('+{}s: Defining grid search'.format(dt()));
        grid_params = {'C':[0.5,1.0,2.0,4.0,8.0],'kernel':["linear","poly","rbf"],'gamma':["scale","auto"]};

        print("+{}s: Grid Searching...".format(dt()));
        grid = GridSearchCV(svm.SVC(), grid_params, n_jobs=-1);
        grid.fit(data, labels);
        dt_train = time.time() - t0_train;

        print('+{}s: Cross-validation results:'.format(dt()));
        cvres = grid.cv_results_;
        print('\tscore\tstd\t\tparameters');
        for score, std, params in zip(cvres['mean_test_score'], cvres['std_test_score'], cvres['params']):
            print('\t{}, {}, {}'.format(round(score, 4), round(std, 5), params));

        print('\n\tGrid search best score: {}'.format(grid.best_score_));
        print('\tGrid search best parameters:');
        for key, value in grid.best_params_.items():
            print('\t\t{}: {}'.format(key, value));


        print('\n+{}s: Validating classifier on train set'.format(dt()));
        pred = grid.predict(data);
        score = f1_score(labels, pred, average='micro');
        print('Classifier f1-score on test set: {}'.format(score));
        print('\nConfusion matrix:');
        print(confusion_matrix(labels, pred));
        print('\nClassification report:');

        tn = [val for val in util.GESTURE.values()];
        print(classification_report(labels, pred, target_names=tn));

        joblib.dump((grid, stdSlr), util.MODEL_PATH);
        # joblib.dump(grid, util.MODEL_PATH);
    else:
        model = svm.SVC(C=2, gamma="scale", kernel="rbf");

        model.fit(data, labels);
        joblib.dump((model, stdSlr), util.MODEL_PATH);
        # joblib.dump(model, util.MODEL_PATH);


    print("+{}s: Training Finish.".format(dt()));



def _SVMTest_(path):
    # model, classes_names, stdSlr, centers, voc = joblib.load("bof.pkl");
    model, stdSlr = joblib.load(util.MODEL_PATH);
    # model = joblib.load(util.MODEL_PATH);
    centers = np.load(util.K_MEANS_PATH);

    data, labels = _calcVector_(path);
    data = stdSlr.transform(data);

    print('+{}s: Validating classifier on test set'.format(dt()));
    pred = model.predict(data);
    score = f1_score(labels, pred, average='micro');
    print('Classifier f1-score on test set: {}'.format(score));
    print('\nConfusion matrix:');
    print(confusion_matrix(labels, pred));
    print('\nClassification report:');

    tn = [val for val in util.GESTURE.values()];
    print(classification_report(labels, pred, target_names=tn));

    acc = accuracy_score(labels, pred);

    return (acc, pred);



t0 = time.time();

if (__name__ == "__main__"):
    now = datetime.now().strftime("%y%m%d_%H%M%S");
    sys.stdout = open(util.LOG_PATH.format(now), 'w+');

    train_path = util.TRAIN_PATH;
    test_path  = util.TEST_PATH;

    argv = sys.argv;
    print(argv[1:]);
    print('DEBUG Mode: {}\nPLOT Mode: {}\n'.format(DEBUG, PLOT));

    if (len(argv) > 1):
        if (argv[1] == "TRAIN"):
            _SaveCenter_(train_path);
            data, labels = _calcVector_(train_path);

            if (len(argv) == 3 and argv[2] == "GRID"): _SVMTrain_(data, labels, GRID=True);
            else: _SVMTrain_(data, labels);

        elif (argv[1] == "TEST"):
            _SVMTest_(test_path);

    else:
        print("doing nothing");

    sys.stdout.close();
