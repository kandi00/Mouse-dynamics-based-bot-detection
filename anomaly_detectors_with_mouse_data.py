"""Outlier detection using COPOD, KNN, PCA, LOF, OCSVM and IForrest anomaly detectors"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay
from pyod.models.copod import COPOD
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.iforest import IForest
import matplotlib.pyplot as plt
from pyod.utils.data import evaluate_print

title = ''
suptitle = ''
num_scores = 10

def detector(clf, human_train, human_test, bot_test, flag):
    # train the detector
    clf.fit(human_train)

    positive_scores = clf.decision_function(human_test)
    negative_scores = clf.decision_function(bot_test)

    ''''''
    ps = list()
    print("positive_scores: ", positive_scores)
    for i in range(0, len(positive_scores) - num_scores + 1):
        sum_scores = 0
        for j in range(i, i + num_scores):
            sum_scores = sum_scores + positive_scores[j]
        ps.append(sum_scores / num_scores)
    positive_scores = np.array(ps)
    print("positive_scores: ", positive_scores)


    # evaluate the model with synthetic data
    ps = list()
    print("negative_scores", negative_scores)
    for i in range(0, len(negative_scores) - num_scores + 1):
        sum_scores = 0
        for j in range(i, i + num_scores):
            sum_scores = sum_scores + negative_scores[j]
        ps.append(sum_scores / num_scores)
    negative_scores = np.array(ps)
    print("negative_scores", negative_scores)
    ''''''

    # print("POSITIVE: ", positive_scores)
    # print("NEGATIVE: ", negative_scores)

    # 0 - inlier; 1 - outlier
    zeros = np.zeros(len(positive_scores))
    ones = np.ones(len(negative_scores))
    y = np.concatenate((zeros, ones), axis=0)  # labels 0 or 1
    y_pred = np.concatenate((positive_scores, negative_scores), axis=0)  # scores
    # print('y_pred: ', y_pred)

    auc = roc_auc_score(y, y_pred)  # gorbe alatti terulet
    print("AUC: ", auc)
    evaluate_print('', y, y_pred)

    if flag:
        y_pred = 1 / (1 + y_pred)
    else:
        # Mivel az OCSVM eseteben vannak negativ ertekek is a score-ok koztt, pozitivat csinalok minden ertekbol, es beskalazom a [0, 1] intervallumba
        mini = min(y_pred)
        print('mini: ', mini)
        y_pred = 1 / (1 + y_pred + abs(mini))
    print('van-e negativ ertek: ', y_pred[y_pred <= 0])

    fpr, tpr, thresholds = roc_curve(y, y_pred, pos_label=0)  # pos_label - label of positive class

    """TNR, FPR, FNR, TPR """
    # from sklearn.metrics import confusion_matrix
    # tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    # my_tpr = tp / len(positive_scores)
    # my_fpr = fp / len(negative_scores)
    # nr = fn / len(positive_scores)
    # tnr = tn / len(negative_scores)
    # print("tpr: ", my_tpr, tpr)
    # print("fpr: ", my_fpr, fpr)
    """"""

    from sklearn import metrics
    roc_auc = metrics.auc(fpr, tpr)
    print("ROC AUC: ", roc_auc)

    return (fpr, tpr, auc)


def detector_(clf, human_train, human_test, bot_test):
    # train the detector
    clf.fit(human_train)

    """ get the prediction labels and outlier scores of the TRAINING data """
    y_train_pred = clf.labels_  # binary labels (0 - inlier; 1 - outlier)
    y_train_scores = clf.decision_scores_  # raw outlier scores

    """ get the prediction on the TEST data """
    y_test_pred_1 = clf.predict(human_test)
    y_test_pred_2 = clf.predict(bot_test)
    y_test_pred = np.concatenate((y_test_pred_1, y_test_pred_2))
    # test scores
    positive_scores = clf.decision_function(human_test)
    negative_scores = clf.decision_function(bot_test)
    y_test_scores = np.concatenate((positive_scores, negative_scores), axis=0)  # scores

    """" calculate y_train, y_test """
    # 0 - inlier; 1 - outlier
    y_train = np.zeros(len(human_train))
    zeros = np.zeros(len(positive_scores))
    ones = np.ones(len(negative_scores))
    y_test = np.concatenate((zeros, ones), axis=0)  # labels 0 or 1

    print("\nOn Test Data:")
    evaluate_print('', y_test, y_test_scores)


# train detectors
def trainDetector(clf, linestyle, detectorName, human_train, human_test, bot_test, flag):
    (fpr, tpr, roc_auc) = detector(clf, human_train, human_test, bot_test, flag)
    # plot
    plt.title(title, fontsize=12)
    plt.suptitle(suptitle, fontsize=14)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.plot(fpr, tpr, label=detectorName + ", auc=" + str(roc_auc)[0:4], linestyle=linestyle)

def train(df_3min, df_1min, df_bot):
    """ create numpy arrays from dataframes """
    """ human train """
    array = df_3min.values
    # array = df_3min
    shape_x = df_3min.shape[0]
    shape_y = df_3min.shape[1]
    human_train = array[0:shape_x, 0:shape_y]
    print(human_train)
    print("human train", human_train.shape)

    """human test """
    array = df_1min.values
    # array = df_1min
    shape_x = df_1min.shape[0]
    shape_y = df_1min.shape[1]
    human_test = array[0:shape_x, 0:shape_y]
    print(human_test)
    print("human test", human_test.shape)

    """ bot test """
    array = df_bot.values
    # array = df_bot
    shape_x = df_bot.shape[0]
    shape_y = df_bot.shape[1]
    bot_test = array[0:shape_x, 0:shape_y]
    print(bot_test)
    print("bot test", bot_test.shape)  # bot test shape ~= human test shape

    # detector_(COPOD(), human_train, human_test, bot_test)
    # detector_(KNN(), human_train, human_test, bot_test)
    # detector_(PCA(), human_train, human_test, bot_test)
    # detector_(LOF(), human_train, human_test, bot_test)
    # detector_(OCSVM(), human_train, human_test, bot_test)
    # detector_(IForest(), human_train, human_test, bot_test)

    trainDetector(COPOD(), "-", "COPOD", human_train, human_test, bot_test, True)  # Probabilistic
    trainDetector(KNN(), "dotted", "KNN", human_train, human_test, bot_test, True)  # Proximity-Based
    trainDetector(PCA(), "-.", "PCA", human_train, human_test, bot_test, True)  # Linear Model
    trainDetector(LOF(), ":", "LOF", human_train, human_test, bot_test, True)  # Proximity-Based
    trainDetector(OCSVM(), '--', "OCSVM", human_train, human_test, bot_test, False)  # Linear Model
    trainDetector(IForest(), 'dotted', "IFOREST", human_train, human_test, bot_test, True)  # Outlier Ensembles
    plt.legend(loc=0)
    plt.show()

def main():
    """Using detectors with extracted features"""
    global suptitle
    suptitle = "31 extracted features"
    df_human_3min = pd.read_csv('../csv_files/3min_extracted_features.csv')
    df_human_1min = pd.read_csv('../csv_files/1min_extracted_features.csv')
    df_bot_1 = pd.read_csv('../csv_files/bot_humanLike_extracted_features.csv')
    df_bot_2 = pd.read_csv('../csv_files/bot_humanLike_random_extracted_features.csv')
    df_bot_3 = pd.read_csv('../csv_files/bot_bezier_extracted_features.csv')
    df_bot_4 = pd.read_csv('../csv_files/bot_bezier_random_extracted_features.csv')

    global title
    title = "bot - Human like curve, human - 1min"
    train(df_human_3min, df_human_1min, df_bot_1)

    title = "bot - Human like curve by random points, human - 1min"
    train(df_human_3min, df_human_1min, df_bot_2)

    title = "bot - Cubic Bézier curve, human - 1min"
    train(df_human_3min, df_human_1min, df_bot_3)

    title = "bot - Cubic Bézier curve by random points, human - 1min"
    train(df_human_3min, df_human_1min, df_bot_4)

    """Using detectors with raw features"""
    suptitle = "128 dx + 128 dy raw features"

    # human train data
    df_human_3min = pd.read_csv('../csv_files/3min.csv')
    # human test data
    df_human_1min = pd.read_csv('../csv_files/1min.csv')
    # bot test data
    df_bot_1 = pd.read_csv('../csv_files/bot_humanLike.csv')
    df_bot_2 = pd.read_csv('../csv_files/bot_humanLike_random.csv')
    df_bot_3 = pd.read_csv('../csv_files/bot_bezier.csv')
    df_bot_4 = pd.read_csv('../csv_files/bot_bezier_random.csv')

    title = "bot - Human like curve, human - 1min"
    train(df_human_3min, df_human_1min, df_bot_1)

    title = "bot - Human like curve by random points, human - 1min"
    train(df_human_3min, df_human_1min, df_bot_2)

    title = "bot - Cubic Bézier curve, human - 1min"
    train(df_human_3min, df_human_1min, df_bot_3)

    title = "bot - Cubic Bézier curve by random points, human - 1min"
    train(df_human_3min, df_human_1min, df_bot_4)

if __name__ == "__main__":
    main()