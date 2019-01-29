import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i')
input = parser.parse_args().i

for file in os.listdir(input):
    if file.startswith('w_'):
        fig = plt.figure()
        ax = Axes3D(fig)

        # x and y
        x = np.arange(0, 28)
        y = np.arange(0, 28)
        X, Y = np.meshgrid(x, y)
        xx, yy = X.ravel(), Y.ravel()
        data = np.loadtxt(os.path.join(input, file))
        z = np.zeros_like(xx + yy)
        data = data.ravel()
        ax.set_zlim(-0.5, 0.5)
        ax.bar3d(xx, yy, z, 1, 1, data, shade=True)
        fname = 'figs/' + file
        fig.savefig(fname=fname)
        plt.close(fig)


