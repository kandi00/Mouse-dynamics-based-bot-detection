import numpy as np
import pandas as pd
import os
import csv
import math

def createTrajectoriesCsvFromHumanLikeMouseMovementPoints(csv):
    df = pd.read_csv(csv)
    (x, y) = calculateDistances(df)
    print("x", len(x))
    column_names = []
    for i in range(1, 129):
        column_names += ['dx' + str(i)]
    for i in range(1, 129):
        column_names += ['dy' + str(i)]

    x = [int(round(num, 0)) for num in x]
    y = [int(round(num, 0)) for num in y]

    with open('bezier_random_.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(column_names)
        pos = 0
        for i in range(0, int(len(x) / 128)):
            writer.writerow([i] + x[pos:pos + 128] + y[pos:pos + 128])
            pos += 129

def calculateDistances(trajectory):
    trajectory = trajectory[trajectory['state'] != 'Pressed']
    x = np.array(trajectory['x'])
    x = np.diff(x)
    y = np.array(trajectory['y'])
    y = np.diff(y)
    return (x, y)


def maxLengthOfTrajectories(df):
    # find the indexes of 'Press'
    indexesOfPressedArray = np.array(df.index[df['state'] == "Pressed"].tolist())
    # find the length of the trajectories from distances between indexes of Press
    trajectoryLengths = np.diff(indexesOfPressedArray) - 1
    # remove too short trajectories
    trajectoryLengths = np.delete(trajectoryLengths, np.where(trajectoryLengths <= 10))

    # Make statistics
    print("min: ", trajectoryLengths.min())
    print("max: ", trajectoryLengths.max())
    print("mean: ", trajectoryLengths.mean())
    print("std: ", trajectoryLengths.std())
    sortedArr = np.sort(trajectoryLengths)
    print("25%: ", sortedArr[int(len(sortedArr) / 4)])
    print("median: ", np.median(sortedArr))
    print("75%: ", sortedArr[int(len(sortedArr) * 3 / 4)])

    sum1 = sum(length <= 129 for length in sortedArr)
    sum2 = len(sortedArr)
    print(str((sum1 - 1) / sum2) + "%: ", sortedArr[int(len(sortedArr) * ((sum1 - 1) / sum2))])
    print(str(sum1 / sum2) + "%: ", sortedArr[int(len(sortedArr) * (sum1 / sum2))])

    # trajectory max length
    maxLen = np.max(trajectoryLengths)
    # maxLen = sortedArr[int(len(sortedArr)*9.9/10)]
    return (maxLen, indexesOfPressedArray)


def dist(x1, y1, x2, y2):
    return int(math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)))


def calculateDistancesInTrajectories(df, maxLen, indexesOfPressedArray, datafile_name):
    i = 0  # nr of iteration
    leftBoundary = indexesOfPressedArray[0]
    flag = False

    if datafile_name.__contains__('1_min'):
        flag = True
        kezdoEsVegpontok = pd.DataFrame(columns=['x1', 'y1', 'x2', 'y2', 'nr_of_internal_points'])
        output_file = open('../csv_files/number_of_intermediate_points_and_distances_1min.txt', 'w')

    for index in indexesOfPressedArray[1:]:
        # get end of trajectory
        rightBoundary = index
        # print("LEFT: ", leftBoundary, "right: ", rightBoundary)

        # trajectory too short (length < 10)
        if rightBoundary - leftBoundary < 10:
            leftBoundary = rightBoundary
            continue

        # trajectory too long (length > 129)
        if rightBoundary - leftBoundary > maxLen:
            trajectory = df[leftBoundary + 1:leftBoundary + 130]
        else:
            trajectory = df[leftBoundary + 1:rightBoundary]

        leftBoundary = rightBoundary

        if flag:
            kezdoEsVegpontok.loc[i] = [trajectory.iloc[0][1], trajectory.iloc[0][2],
                                       trajectory.iloc[len(trajectory) - 1][1], trajectory.iloc[len(trajectory) - 1][2],
                                       len(trajectory) - 2]

            distance = dist(float(trajectory.iloc[0][1]), float(trajectory.iloc[0][2]),
                            float(trajectory.iloc[len(trajectory) - 1][1]),
                            float(trajectory.iloc[len(trajectory) - 1][2]))
            # if distance > 2000:
            #     print(trajectory.iloc[0][1], trajectory.iloc[0][2], trajectory.iloc[len(trajectory) - 1][1], trajectory.iloc[len(trajectory) - 1][2])
            nr_of_intermediate_points_and_distance = (len(trajectory) - 2, distance)
            output_file.write(str(nr_of_intermediate_points_and_distance).strip('(').strip(')') + '\n')

        # calculation of distances and adding 0 or nan values
        (x, y) = calculateDistances(trajectory)
        differenceOfLengths = maxLen - len(x)
        empty_array = np.array(differenceOfLengths * [0])
        # empty_array = np.array(differenceOfLengths * [np.nan])
        x = np.append(x, empty_array)
        y = np.append(y, empty_array)

        if i == 0:
            xx = x
            yy = y
        elif i == 1:
            xx = np.append([xx], [x], axis=0)
            yy = np.append([yy], [y], axis=0)
        else:
            xx = np.append(xx, [x], axis=0)
            yy = np.append(yy, [y], axis=0)
        i += 1

    if flag:
        kezdoEsVegpontok.to_csv("../csv_files/kezdoEsVegpontok.csv", index=False)
        output_file.close()
    return (xx, yy)


def myLength(e):
    return len(e)


def createTrajectoriesCsvFromHumanMouseMovementPoints(datafile_name):
    # Load dataset
    # list of directories: from user1 to user120
    data_file_directories = os.listdir('../csv_files/sapimouse')
    data_file_directories.sort(key=myLength)

    # get filenames
    data_file_names = []
    for i in range(0, 120):
        data_file_names = data_file_names + ['../csv_files/sapimouse/' + data_file_directories[i] + '/' +
                                             [f for f in
                                              os.listdir('../csv_files/sapimouse/' + data_file_directories[i]) if
                                              datafile_name in f][0]]

    df_pressed_released = pd.DataFrame(np.array([[0, 'Left', 'Pressed', 0, 0], [0, 'Left', 'Released', 0, 0]]),
                                       columns=['client timestamp', 'button', 'state', 'x', 'y'])
    df = pd.concat([pd.concat([df_pressed_released, pd.read_csv(f)]) for f in data_file_names])
    # df = pd.read_csv('./sapimouse/user34/session_2020_06_09_1min.csv')

    # # Drop unuseful data
    df = df.drop(['client timestamp', 'button'], axis=1)
    df = df[df['state'] != 'Drag']
    df = df[df['state'] != 'Released']

    # # Reset indexes
    df = df.reset_index()
    df = df.drop(['index'], axis=1)

    (maxLen, indexesOfPressedArray) = maxLengthOfTrajectories(df)
    maxLen = 128
    (xx, yy) = calculateDistancesInTrajectories(df, maxLen, indexesOfPressedArray, datafile_name)
    column_names_x = []
    for i in range(1, maxLen + 1):
        column_names_x += ['dx' + str(i)]
    column_names_y = []
    for i in range(1, maxLen + 1):
        column_names_y += ['dy' + str(i)]

    final_df1 = pd.DataFrame(xx, columns=column_names_x)
    final_df2 = pd.DataFrame(yy, columns=column_names_y)
    print(final_df1)
    print(final_df2)
    final_df = pd.concat([final_df1, final_df2], axis=1)
    final_df = final_df.astype(float)
    print(final_df)
    # print to csv
    final_df.to_csv('../csv_files/' + datafile_name + ".csv", index=False)


if __name__ == "__main__":
    createTrajectoriesCsvFromHumanMouseMovementPoints('1min')
    createTrajectoriesCsvFromHumanMouseMovementPoints('3min')
