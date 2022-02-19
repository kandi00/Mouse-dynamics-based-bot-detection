import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.pyplot import figure

import pathlib
print(pathlib.Path(__file__).parent.resolve())

def plotBasedOnDistances(df_human, df_bot, fig_suptitle, fig_title):
     numpy_array_bot = df_bot.values
     numpy_array_human = df_human.values
     print(numpy_array_bot)
     print(numpy_array_human)
     for i in range(0, len(numpy_array_bot)):
          x_bot = np.cumsum(np.append(0, numpy_array_bot[i][:127]))
          print(len(x_bot))
          x_human = np.append(0, np.cumsum(numpy_array_human[i][:127]))
          print(len(x_human))
          y_bot = np.cumsum(np.append(0, numpy_array_bot[i][127:254]))
          print(len(y_bot))
          y_human = np.append(0, np.cumsum(numpy_array_human[i][127:254]))
          print(len(y_human))
          
          plt.suptitle(fig_suptitle, fontsize=12)
          plt.title(fig_title, fontsize=12)
          plt.plot(x_bot, y_bot, 'bo')
          plt.plot(x_human, y_human, 'go')
          plt.plot(x_bot[0], y_bot[0], 'ro')
          plt.plot(x_human[0], y_human[0], 'ro')
          plt.plot(x_human[len(x_human)-1], y_human[len(y_human)-1], 'ro')
          plt.plot(x_bot[len(x_bot)-1], y_bot[len(y_bot)-1], 'ro')
          plt.plot(x_bot, y_bot, 'b-')
          plt.plot(x_human, y_human, 'g-')
          plt.show()

def main():
     df_human = pd.read_csv('../csv_files/1min.csv', nrows=1)
     df_bot = pd.read_csv('../csv_files/bot_bezier.csv', nrows=1)
     plotBasedOnDistances(df_human, df_bot, 'Human mouse movement - green', 'Cubic Bézier curve - blue')
     df_bot = pd.read_csv('../csv_files/bot_humanLike.csv', nrows=1)
     plotBasedOnDistances(df_human, df_bot, 'Human mouse movement - green', 'Humanlike curve - blue')

if __name__ == "__main__":
    main()
