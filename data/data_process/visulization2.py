# plot the trajectories on one single figure


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import imageio

import sys
import os
currentPath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(currentPath)
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle

data=pickle.load(open('/home/fanuc/zhengwu/sampling_irl/data/real/xy/xy_list.pkl', 'rb'))


if __name__ == "__main__":
    plt.rcParams["figure.figsize"] = (17,10)
    for i in range(1, len(data)):
      traj_1 = data[0]['traj']
      # plt.xticks(np.arange(min(traj_1[0]), max(traj_1[0]), 1))
      # plt.yticks(np.arange(min(traj_1[1]), max(traj_1[1]), 1))
      plt.plot(np.array(traj_1[0]),np.array(traj_1[1]),c = "r", linewidth=3., alpha=1.)

      traj_2 = data[i]['traj']
      plt.plot(np.array(traj_2[0]),np.array(traj_2[1]),c = "r", linewidth=3., alpha=1.)

    # plt.title("visulization of dataset")
    # print(et,ot)
    # plt.legend([et,ot],["egocar trajectory", "adocar trajectory"],fontsize=15)
    # plt.show()
      plt.savefig('all_imgs/noninteractive_{}.png'.format(i))
      plt.close()
