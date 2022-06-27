import pandas as pd
import numpy as np
import math

# returns histogram of each row
def calculate_histo(df):
    df = df.abs()
    bins = [0, 5, 10, 20, 30, 100]
    rows, columns = df.shape
    histos = list()

    for i in range(0, rows):
        df_row = df.iloc[i, :]
        df_row_dx = df_row.iloc[0:128]
        df_row_dy = df_row.iloc[128:256]

        count_dx, _ = np.histogram(df_row_dx, bins=bins)
        count_dy, _ = np.histogram(df_row_dy, bins=bins)

        count = np.array(count_dx / np.count_nonzero(~np.isnan(df_row_dx))) # devide by the number of non-NaN dx values
        count = np.append(count, count_dy / np.count_nonzero(~np.isnan(df_row_dy))) # devide by the number of non-NaN dy values
        histos.append(count)
    columns = ["histo_label" + str(i) for i in range(0, len(count))]
    result_df = pd.DataFrame(data=np.vstack(histos), columns=columns)
    '''result_df: 
    egz. the number of displacements (especially dxs and dys) between 0 and 5 - percentage
    dx - between 0-5,  dx between 5-10, ..., dy - between 20-30,  dy between 30-100 '''
    return result_df

def smoothness(df):
    rows = df.shape[0]
    x_array = df.iloc[:, 0:128].values
    y_array = df.iloc[:, 128:256].values
    features = list()

    for i in range(0, rows):
        dx1 = np.diff(x_array[i, :], axis=0) #second order differences
        dx1 = dx1[~np.isnan(dx1)] #remove nan values
        dx2 = np.diff(dx1, axis=0) #third order differences
        dx1 = np.abs(dx1)
        dx2 = np.abs(dx2)
        nx1 = np.count_nonzero(dx1)
        sx1 = np.sum(dx1)
        nx2 = np.count_nonzero(dx2)
        sx2 = np.sum(dx2)

        dy1 = np.diff(y_array[i, :], axis=0) #second order differences
        dy1 = dy1[~np.isnan(dy1)]  #remove nan values
        dy2 = np.diff(dy1, axis=0) #third order differences
        dy1 = np.abs(dy1)
        dy2 = np.abs(dy2)
        ny1 = np.count_nonzero(dy1)
        sy1 = np.sum(dy1)
        ny2 = np.count_nonzero(dy2)
        sy2 = np.sum(dy2)

        if nx1 == 0:
            nx1 = 1
        if ny1 == 0:
            ny1 = 1
        if nx2 == 0:
            nx2 = 1
        if ny2 == 0:
            ny2 = 1
        count = []
        count = np.append(count, sx1 / nx1)
        count = np.append(count, sx2 / nx2)
        count = np.append(count, sy1 / ny1)
        count = np.append(count, sy2 / ny2)
        features.append(count)

    columns = ["label_smoothness" + str(i) for i in range(0, len(count))]
    result_df = pd.DataFrame(data=np.vstack(features), columns=columns)
    return result_df


def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))


def movement_efficiency(df):
    rows = df.shape[0]
    dx_array = df.iloc[:, 0:128].values
    dy_array = df.iloc[:, 128:256].values
    efficiency = list()

    for i in range(0, rows):
        x = np.cumsum(np.append(0, dx_array[i][:127]))
        y = np.cumsum(np.append(0, dy_array[i][:127]))
        distance = dist(x[0], y[0], x[-1], y[-1])
        path = 0
        for k in range(1, 128):
            path += dist(x[k], y[k], x[k - 1], y[k - 1])

        try:
            efficiency.append(distance / path)
        except:
            efficiency.append(0.1)

    efficiency_array = np.array(efficiency)
    result_df = pd.DataFrame({"label_efficiency": efficiency_array})
    return result_df

def movement_efficiency_(df):
    rows = df.shape[0]
    dx_array = df.iloc[:, 0:128].values
    dy_array = df.iloc[:, 128:256].values
    efficiency = list()

    #for i in range(0, len(positive_scores) - num_scores + 1):
    for i in range(0, rows):
        counter = 0
        x = np.cumsum(np.append(0, dx_array[i][:127]))
        y = np.cumsum(np.append(0, dy_array[i][:127]))
        for j in range(0, len(x)-5):
            distance = dist(x[j], y[j], x[j+5], y[j+5])
            tmp = 0
            for k in range(j+1, j+5):
                tmp = tmp + dist(x[k], y[k], x[k - 1], y[k - 1])
            if tmp == distance and tmp != 0:
                counter = counter + 1
        efficiency.append(counter)

    efficiency_array = np.array(efficiency)
    result_df = pd.DataFrame({"label_efficiency": efficiency_array})
    return result_df

def createDataframeWithStatisticalFeatures(df):
    # Create new dataframe with new features
    data = {
        'min': df.min(axis=1),
        'first quartile': pd.Series(df.quantile([0.25], axis=1).transpose()[0.25].values),
        'median': df.median(axis=1),
        'third quartile': pd.Series(df.quantile([0.75], axis=1).transpose()[0.75].values),
        'max': df.max(axis=1),
        'mode': df.mode(axis=1, numeric_only=False, dropna=True).iloc[:, 0],
        'skewness': df.skew(axis=1, skipna=True),  # https://www.geeksforgeeks.org/python-pandas-dataframe-skew/
        # 'mean': df.mean(axis=1),
        # 'std': df.std(axis=1),
        # 'variance': df.var(axis=1),
        # 'kurtosis': df.kurtosis(axis=1, skipna=True), # https://www.geeksforgeeks.org/scipy-stats-kurtosis-function-python/
    }
    # Average and measure of central locations
    # data = {
    #     'mean': df.mean(axis=1),
    #     'mode': df.mode(axis=1, numeric_only=False, dropna=True).iloc[:, 0],
    #     'first quartile': pd.Series(df.quantile([0.25], axis=1).transpose()[0.25].values),
    #     'median': df.median(axis=1),
    #     'third quartile': pd.Series(df.quantile([0.75], axis=1).transpose()[0.75].values),
    # }
    # Measure of spread
    # data = {
    #     'variance': df.var(axis=1),
    #     'std': df.std(axis=1),
    #     'first quartile': pd.Series(df.quantile([0.25], axis=1).transpose()[0.25].values),
    #     'median': df.median(axis=1),
    #     'third quartile': pd.Series(df.quantile([0.75], axis=1).transpose()[0.75].values)
    # }
    return pd.DataFrame(data)

def feature_extraction(df, df_zeros, file_name):
    # seperate dataframe to x and y features
    column_names_x = []
    column_names_y = []
    for i in range(1, 129):
        column_names_x += ['dx' + str(i)]
        column_names_y += ['dy' + str(i)]

    dfX = df[column_names_x]
    dfY = df[column_names_y]

    # create statistical features
    """csv"""
    df_stat = createDataframeWithStatisticalFeatures(dfX)
    df_stat = pd.concat([df_stat, createDataframeWithStatisticalFeatures(dfY)], axis = 1)
    df_stat = pd.concat([df_stat, movement_efficiency(df_zeros)], axis = 1)
    # df_stat = pd.concat([df_stat, movement_efficiency_(df_zeros)], axis = 1)
    df_stat = pd.concat([df_stat, smoothness(df)], axis = 1)
    df_stat = pd.concat([df_stat, calculate_histo(df)], axis = 1)
    df_stat.to_csv(file_name, index=False)

def main():
    # bot test data
    df_bot = pd.read_csv('../csv_files/bot_bezier_random_nan.csv')  # for statistical features
    df_bot_zeros = pd.read_csv('../csv_files/bot_bezier_random.csv')  # for raw features
    feature_extraction(df_bot, df_bot_zeros, '../csv_files/bot_bezier_random_extracted_features.csv')

    df_bot = pd.read_csv('../csv_files/bot_humanLike_random_nan.csv')  # for statistical features
    df_bot_zeros = pd.read_csv('../csv_files/bot_humanLike_random.csv')  # for raw features
    feature_extraction(df_bot, df_bot_zeros, '../csv_files/bot_humanLike_random_extracted_features.csv')

    df_bot = pd.read_csv('../csv_files/bot_bezier_nan.csv')  # for statistical features
    df_bot_zeros = pd.read_csv('../csv_files/bot_bezier.csv')  # for raw features
    feature_extraction(df_bot, df_bot_zeros, '../csv_files/bot_bezier_extracted_features.csv')

    df_bot = pd.read_csv('../csv_files/bot_humanLike_nan.csv')  # for statistical features
    df_bot_zeros = pd.read_csv('../csv_files/bot_humanLike.csv')  # for raw features
    feature_extraction(df_bot, df_bot_zeros, '../csv_files/bot_humanLike_extracted_features.csv')

    # human train data
    df_human_3min = pd.read_csv('../csv_files/3min_nan.csv')
    df_human_3min_zeros = pd.read_csv('../csv_files/3min.csv')
    feature_extraction(df_human_3min, df_human_3min_zeros, '../csv_files/3min_extracted_features.csv')

    # human test data
    df_human_1min = pd.read_csv('../csv_files/1min_nan.csv')
    df_human_1min_zeros = pd.read_csv('../csv_files/1min.csv')
    feature_extraction(df_human_1min, df_human_1min_zeros, '../csv_files/1min_extracted_features.csv')

    # gan
    df_bot = pd.read_csv('../csv_files/synthetic_gan_3min_all_nan_hidden_dim_64.csv')  # for statistical features
    df_bot_zeros = pd.read_csv('../csv_files/synthetic_gan_3min_all_hidden_dim_64.csv')  # for raw features
    feature_extraction(df_bot, df_bot_zeros, '../csv_files/synthetic_gan_3min_all_extracted_features_hidden_dim_64.csv')

if __name__ == "__main__":
    main()