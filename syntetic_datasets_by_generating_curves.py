import matplotlib.pyplot as plt
from pyclick._beziercurve import BezierCurve
import numpy as np
from pyclick.humancurve import HumanCurve
import csv
import pandas as pd
import math

def getHumanPoints(P1, P2, targetPoints_):
    hc = HumanCurve(P1, P2)
    Humanpoints = hc.generateCurve(targetPoints=int(targetPoints_))
    return Humanpoints


def dist(x1, y1, x2, y2):
    return int(math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)))


def bezier_random():
    xMax = 1700
    yMax = 1700
    nrOfBezierPoints = 129
    bezier_points = []
    for i in range(1, 5500):
        # random.randint - uniform distribution
        xP1, xP2, xP3, xP4 = np.random.randint(0, xMax, 4)
        yP1, yP2, yP3, yP4 = np.random.randint(0, yMax, 4)
        controlPoints3 = [(xP1, yP1), (xP2, yP2), (xP3, yP3), (xP4, yP4)]
        bezier_points += BezierCurve.curvePoints(nrOfBezierPoints, controlPoints3)
        #print(len(bezier_points)/i)
    print(len(bezier_points))
    with open('../csv_files/bezier_random_points_.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        for point in bezier_points:
            writer.writerow(point)

def human_curves_to_csv():
    df = pd.read_csv('../csv_files/kezdoEsVegpontok.csv')
    df.columns = ['x1', 'y1', 'x2', 'y2', 'nr_of_internal_points']

    with open('../csv_files/bot_humanLike_nan.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        column_names = []
        for i in range(1, 129):
            column_names += ['dx' + str(i)]
        for i in range(1, 129):
            column_names += ['dy' + str(i)]
        writer.writerow(column_names)

        for i in range(0, len(df)):
            # print(df['x'].iloc[i], df['y'].iloc[i])
            # print(df['x'].iloc[i+1], df['y'].iloc[i+1])
            # print(df['Index'][i+1] - df['Index'][i] + 1)
            # generate a curve
            humanPoints = getHumanPoints((df['x1'].iloc[i], df['y1'].iloc[i]), (df['x2'].iloc[i], df['y2'].iloc[i]), df['nr_of_internal_points'].iloc[i])
            humanPoints = np.array([(int(round(x, 0)), int(round(y, 0))) for x, y in humanPoints])
            # calculate distances for trajectories
            x = np.diff(humanPoints[:, 0])
            y = np.diff(humanPoints[:, 1])
            # append zeros to create trajectories with length = 129
            differenceOfLengths = 128 - len(x)
            empty_array = np.array(differenceOfLengths * [np.nan])
            x = np.append(x, empty_array)
            y = np.append(y, empty_array)
            xy = np.append(x, y)
            writer.writerow(xy)

def bezier_curves_to_csv():
    df = pd.read_csv('../csv_files/kezdoEsVegpontok.csv')
    df.columns = ['x1', 'y1', 'x2', 'y2', 'nr_of_internal_points']

    with open('../csv_files/bot_bezier_nan.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        column_names = []
        for i in range(1, 129):
            column_names += ['dx' + str(i)]
        for i in range(1, 129):
            column_names += ['dy' + str(i)]
        writer.writerow(column_names)

        for i in range(0, len(df), 1):
            # generate a curve
            xP1, xP2 = np.random.randint(0, 1700, 2)
            yP1, yP2 = np.random.randint(0, 1700, 2)
            controlPoints3 = [(df['x1'].iloc[i], df['y1'].iloc[i]), (xP1, yP1), (xP2, yP2), (df['x2'].iloc[i], df['y2'].iloc[i])]
            nrOfBezierPoints = df['nr_of_internal_points'][i]
            bezier_points = BezierCurve.curvePoints(nrOfBezierPoints, controlPoints3)
            bezier_points = np.array([(int(round(x, 0)), int(round(y, 0))) for x, y in bezier_points])
            # calculate distances for trajectories
            x = np.diff(bezier_points[:, 0])
            y = np.diff(bezier_points[:, 1])
            # append zeros to create trajectories with length = 129
            differenceOfLengths = 128 - len(x)
            empty_array = np.array(differenceOfLengths * [np.nan])
            x = np.append(x, empty_array)
            y = np.append(y, empty_array)
            xy = np.append(x, y)
            writer.writerow(xy)

def createDictionaryWithNumberOfInternalPointsAndDistances():
    df = pd.read_csv('../csv_files/number_of_intermediate_points_and_distances_1min.txt', header = None)
    print(df)
    df = df.sort_values(by=[1])
    ls = list()

    for i in range(0, len(df)):
        if df.iloc[i-1][1] == df.iloc[i][1]:
            continue
        subDf = df.where(df[1] == df.iloc[i][1])
        ls.append((df.iloc[i][1], subDf.mean()[0]))

    dictionary = {}
    i = 1
    nr = 1
    for (nrOfPoints, dist) in ls:
        if(i == len(ls)):
            tmp = 1
        else:
            tmp = ls[i][0] - ls[i-1][0]
        dictionary[nr] = int(dist)
        for j in range(nr+1, nr+tmp):
            dictionary[j] = int((dist + ls[i][1])/2)
        nr = nr+tmp
        i = i+1
    return dictionary


def random_human_curves_to_csv():
    with open('../csv_files/bot_humanlike_random_nan.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        column_names = []
        for i in range(1, 129):
            column_names += ['dx' + str(i)]
        for i in range(1, 129):
            column_names += ['dy' + str(i)]
        writer.writerow(column_names)

        xMax = 1834
        yMax = 1048
        dict = createDictionaryWithNumberOfInternalPointsAndDistances()
        for i in range(1, 5990):
            # random.randint - uniform distribution
            xP1, xP2 = np.random.randint(0, xMax, 2)
            yP1, yP2 = np.random.randint(0, yMax, 2)
            Humanpoints = getHumanPoints((xP1, yP1), (xP2, yP2), dict[dist(xP1, yP1, xP2, yP2)])
            Humanpoints = np.array([(int(round(x, 0)), int(round(y, 0))) for x, y in Humanpoints])
            x = np.diff(Humanpoints[:, 0])
            y = np.diff(Humanpoints[:, 1])
            # append zeros to create trajectories with length = 129
            differenceOfLengths = 128 - len(x)
            empty_array = np.array(differenceOfLengths * [np.nan])
            x = np.append(x, empty_array)
            y = np.append(y, empty_array)
            xy = np.append(x, y)
            writer.writerow(xy)

def random_bezier_curves_to_csv():
    with open('../csv_files/bot_bezier_random_nan.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        column_names = []
        for i in range(1, 129):
            column_names += ['dx' + str(i)]
        for i in range(1, 129):
            column_names += ['dy' + str(i)]
        writer.writerow(column_names)

        xMax = 1834
        yMax = 1048
        dict = createDictionaryWithNumberOfInternalPointsAndDistances()
        for i in range(1, 5990):
            # random.randint - uniform distribution
            xP1, xP2, xP3, xP4 = np.random.randint(0, xMax, 4)
            yP1, yP2, yP3, yP4 = np.random.randint(0, yMax, 4)
            controlPoints3 = [(xP1, yP1), (xP2, yP2), (xP3, yP3), (xP4, yP4)]
            bezier_points = BezierCurve.curvePoints(dict[dist(xP1, yP1, xP4, yP4)], controlPoints3)
            bezier_points = np.array([(int(round(x, 0)), int(round(y, 0))) for x, y in bezier_points])
            x = np.diff(bezier_points[:, 0])
            y = np.diff(bezier_points[:, 1])
            # append zeros to create trajectories with length = 129
            differenceOfLengths = 128 - len(x)
            empty_array = np.array(differenceOfLengths * [np.nan])
            x = np.append(x, empty_array)
            y = np.append(y, empty_array)
            xy = np.append(x, y)
            writer.writerow(xy)

def main():
    human_curves_to_csv()
    bezier_curves_to_csv()
    random_human_curves_to_csv()
    random_bezier_curves_to_csv()

if __name__ == "__main__":
    main()

