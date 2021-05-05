'''
util.py

109-2 283647 多媒體技術與應用
    期中專案一   剪刀石頭布

第十二組
    資工二 108590049 符芷琪
    資工二 108590050 李浩銘
'''

import cv2;
import numpy as np;
from scipy.cluster.vq import *;


# Define DEBUG
DEBUG = False;



# Define Gesture
PAPER    = 0;
ROCK     = 1;
SCISSORS = 2;

GESTURE = {PAPER: 'PAPER', ROCK: 'ROCK', SCISSORS: 'SCISSORS'};



# Define Path
MODEL_PATH   = "./model/model.pkl";
K_MEANS_PATH = "./model/kmeans.npy";
LOG_PATH     = "./log/{}.log";
TRAIN_PATH   = "./dataset/train/";
TEST_PATH    =  "./dataset/test/";


# Define K-Means_clusters
K_MEANS_CLUSTERS = 30;


#####
# SIFT
#####
def _SIFT_(img):
    sift = cv2.SIFT_create();
    kp, des = sift.detectAndCompute(img, None);

    return des;



def _FeatureVector_(features, centers):
    featureVector = np.zeros((1, K_MEANS_CLUSTERS), "float32");

    words, distance = vq(features, centers);

    for w in words:
        featureVector[0][w] += 1;

    # for feature in features:
    #     diff = np.tile(feature, (K_MEANS_CLUSTERS, 1)) - centers;
    #
    #     squareSum = (diff**2).sum(axis=1);
    #     dist = squareSum**0.5;
    #
    #     sortestDist = dist.argsort()[0];
    #
    #     featureVector[0][sortestDist] += 1;

    return featureVector;
