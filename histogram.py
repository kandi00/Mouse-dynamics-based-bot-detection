import pylab
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_histo(df):
     #df = df.abs()
     dx_values_df = df.values[:, 0:128]
     print(dx_values_df)
     dy_values_df = df.values[:, 128:256]
     print(dy_values_df)

     #histo for dx values
     #bins = [0, 5, 10, 20, 30, 100]
     bins = [-100, -30, -20, -10, -5 , 0, 5, 10, 20, 30, 100]
     #plt.ylim(0,400000)
     _ = plt.hist(dx_values_df[~np.isnan(dx_values_df)], bins=bins)  # arguments are passed to np.histogram
     plt.xlabel('dx elmozdulások hossza',fontsize=12)
     plt.ylabel('Gyakoriság',fontsize=12)
     plt.show()

     #histo for dy values
     #bins = [0, 5, 10, 20, 30, 100]
     bins = [-100, -30, -20, -10, -5 , 0, 5, 10, 20, 30, 100]
     #plt.ylim(0,350000)
     _ = plt.hist(dy_values_df[~np.isnan(dy_values_df)], bins=bins)  # arguments are passed to np.histogram
     plt.xlabel('dy elmozdulások hossza',fontsize=12)
     plt.ylabel('Gyakoriság',fontsize=12)
     plt.show()

def internal_points_histo(df):
     bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
     _ = plt.hist(df[0])  # arguments are passed to np.histogram
     plt.xlabel('Trajektória köztes pontjainak száma',fontsize=12)
     plt.ylabel('Gyakoriság',fontsize=12)
     plt.show()

def distances_histo(df):
     bins = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
     _ = plt.hist(df[1])  # arguments are passed to np.histogram
     plt.xlabel('Kezdő és végpontok közti távolság (pixelben)',fontsize=12)
     plt.ylabel('Gyakoriság',fontsize=12)
     plt.show()

def dist_vs_nr_of_internal_points(df):
     plt.scatter(*zip(*df.values), marker = '.')
     plt.xlabel('Trajektória köztes pontjainak száma',fontsize=12)
     plt.ylabel('Kezdő és végpontok közti távolság',fontsize=12)
     plt.show()


def main():
     df1 = pd.read_csv('../csv_files/1min_nan.csv')
     create_histo(df1)
     df2 = pd.read_csv('../csv_files/3min_nan.csv')
     create_histo(df2)
     create_histo(df1.append(df2))
     df = pd.read_csv('../csv_files/bot_bezier_nan.csv')
     create_histo(df)
     df = pd.read_csv('../csv_files/bot_bezier_random_nan.csv')
     create_histo(df)
     df = pd.read_csv('../csv_files/bot_humanLike_nan.csv')
     create_histo(df)
     df = pd.read_csv('../csv_files/bot_humanlike_random_nan.csv')
     create_histo(df)

     df1 = pd.read_csv('../csv_files/number_of_intermediate_points_and_distances_1min.txt', header = None)
     df2 = pd.read_csv('../csv_files/number_of_intermediate_points_and_distances_3min.txt', header = None)
     df = df1.append(df2)
     df = df.sort_values(by=[1])
     dist_vs_nr_of_internal_points(df)
     internal_points_histo(df1)
     distances_histo(df1)

if __name__ == "__main__":
    main()



