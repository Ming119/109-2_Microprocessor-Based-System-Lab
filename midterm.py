'''
midterm.py

109-2 283647 多媒體技術與應用
    期中專案一   剪刀石頭布

第十二組
    資工二 108590049 符芷琪
    資工二 108590050 李浩銘
'''

#
# Dataset
# http://www.laurencemoroney.com/rock-paper-scissors-dataset/
#

import util;

import random, sys, os, cv2, time, glob, joblib;
import numpy as np;
import matplotlib.pyplot as plt;
from sklearn import svm;
from skimage import io,transform;


def _Predict_(img):
    model, stdSlr = joblib.load(util.MODEL_PATH);
    # model = joblib.load(util.MODEL_PATH);
    centers = np.load(util.K_MEANS_PATH);

    features = util._SIFT_(img);
    featureVector = util._FeatureVector_(features, centers);

    featureVector = stdSlr.transform(featureVector);

    pred = model.predict(featureVector);

    return pred;

def ComputerMove():
    computer_choose = random.randint(0, 2);
    if (computer_choose == util.ROCK): computer_img = './tr.jpg';
    if (computer_choose == util.PAPER): computer_img = './tp.jpg';
    if (computer_choose == util.SCISSORS): computer_img = './ts.jpg';
    computer = cv2.imread(computer_img);
    computer = cv2.resize(computer, (300,300));

    return (computer_choose, computer);

def DrawWinLose(computer_choose, computer_img, predict, player):
    # Output Image
    combined = np.hstack((computer_img, player));

    text_wins = "WINS";
    text_lose = "LOSE";
    text_draw = "DRAW";
    # Compare computer_choose and predict
    # util.PAPER   = 0
    # util.ROCK    = 1
    # util.SCISSORS = 2
    # Computer Wins
    if (computer_choose == util.PAPER and predict == util.ROCK or \
        computer_choose == util.ROCK and predict == util.SCISSORS or \
        computer_choose == util.SCISSORS and predict == util.PAPER):

        cv2.putText(combined, text_wins, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA);
        cv2.putText(combined, text_lose, (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA);

    # Player Wins
    elif (computer_choose == util.ROCK and predict == util.PAPER or \
        computer_choose == util.SCISSORS and predict == util.ROCK or \
        computer_choose == util.PAPER and predict == util.SCISSORS):

        cv2.putText(combined, text_wins, (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA);
        cv2.putText(combined, text_lose, (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA);

    # Draw
    else:
        cv2.putText(combined, text_draw, (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA);

    return combined;


if (__name__ == "__main__"):
    argv = sys.argv;
    if (len(argv) > 1):
        # Computer Move
        computer_choose, computer_img = ComputerMove();

        # Player Move
        argv = sys.argv;
        if (len(argv) > 1):
            player_img = argv[1];
            # if (argv[1] == "ROCK"): player_img = './tr2.jpg';
            # if (argv[1] == "PAPER"): player_img = './tp2.jpg';
            # if (argv[1] == "SCISSOR"): player_img = './ts2.jpg';
        player = cv2.imread(player_img);
        player = cv2.resize(player, (300,300));
        player = cv2.flip(player, 1);



        # predict Player Move
        predict = _Predict_(cv2.cvtColor(player, cv2.COLOR_BGR2GRAY)).astype(np.int64);
        if (util.DEBUG):
            print(predict);
            print('Predict: ', util.GESTURE[predict[0]]);

        combined = DrawWinLose(computer_choose, computer_img, predict, player);

        cv2.imshow("output", combined);
        cv2.imwrite("output.jpg", combined);
        cv2.waitKey(0);
        cv2.destroyAllWindows();

    else:
        features = np.float32([]).reshape(0, 128);
        for label, folder in enumerate(os.listdir(util.TEST_PATH)):
            for _, file in enumerate(glob.glob(util.TEST_PATH + folder + "/*.png")):
                img = cv2.imread(file);
                pred = _Predict_(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY));

                if (pred != label):
                    computer_choose, computer_img = ComputerMove();
                    output = DrawWinLose(computer_choose, computer_img, pred, img);
                    cv2.imwrite('./doc/error output/output{}{}.jpg'.format(label, _), output);
                    cv2.waitKey(0);
                    cv2.destroyAllWindows();
