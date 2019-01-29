import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i')
input = parser.parse_args().i

fig = plt.figure()
ax = Axes3D(fig)

# x and y
x = np.arange(0, 28)
y = np.arange(0, 28)
X, Y = np.meshgrid(x, y)
xx, yy = X.ravel(), Y.ravel()
data = np.loadtxt(input)
z = np.zeros_like(xx + yy)
ax.plot_surface(X, Y, data, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
plt.show()

