import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


# 权值矩阵可视化动画展现代码

def get_polys(num):
    _x = np.arange(28)
    _y = np.arange(28)
    _xx, _yy = np.meshgrid(_x, _y)
    x, y = _xx.flatten(), _yy.flatten()
    z = 0
    dz = np.loadtxt('FullConnect/w_' + str(num)).flatten()
    dx = dy = 0.5
    x, y, z, dx, dy, dz = np.broadcast_arrays(np.atleast_1d(x), y, z, dx, dy, dz)
    polys = []
    for xi, yi, zi, dxi, dyi, dzi in zip(x, y, z, dx, dy, dz):
        polys.extend([
            ((xi, yi, zi), (xi + dxi, yi, zi),
             (xi + dxi, yi + dyi, zi), (xi, yi + dyi, zi)),
            ((xi, yi, zi + dzi), (xi + dxi, yi, zi + dzi),
             (xi + dxi, yi + dyi, zi + dzi), (xi, yi + dyi, zi + dzi)),

            ((xi, yi, zi), (xi + dxi, yi, zi),
             (xi + dxi, yi, zi + dzi), (xi, yi, zi + dzi)),
            ((xi, yi + dyi, zi), (xi + dxi, yi + dyi, zi),
             (xi + dxi, yi + dyi, zi + dzi), (xi, yi + dyi, zi + dzi)),

            ((xi, yi, zi), (xi, yi + dyi, zi),
             (xi, yi + dyi, zi + dzi), (xi, yi, zi + dzi)),
            ((xi + dxi, yi, zi), (xi + dxi, yi + dyi, zi),
             (xi + dxi, yi + dyi, zi + dzi), (xi + dxi, yi, zi + dzi)),
        ])
    return polys


def update_bar3d(num, bar3d):
    bar3d.set_verts(get_polys(num))


_x = np.arange(28)
_y = np.arange(28)
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.flatten(), _yy.flatten()
top = np.loadtxt('FullConnect/w_0').flatten()

fig = plt.figure()
ax = Axes3D(fig)
ax.set_zlim(-0.1, 0.1)

bar3d = ax.bar3d(x, y, 0, 0.5, 0.5, top, color='b', shade=False)
bar3d.set_alpha(0.8)

bar3d.set_edgecolor('#ff0000')

bar3d_anim = animation.FuncAnimation(fig, update_bar3d, 10, fargs=[bar3d], interval=20, blit=False)

plt.show()

