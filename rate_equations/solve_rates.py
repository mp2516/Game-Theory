import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def rate_solver(step):

    # x_list = np.arange(0, 1, step)
    # y_list = np.arange(0, 1, step)
    #
    # z = np.zeros((int(1/step), int(1/step)))
    # for x_num, x in enumerate(x_list):
    #     for y_num, y in enumerate(y_list):
    #         z[x_num,y_num] = x * y + (x + y) * (1 - x - y)
    #
    # plt.contour(x_list, y_list, z)
    # plt.show()


    def rate(x, y):
        return x * y + (x + y) * (1 - x - y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(0, 1, 0.01)
    X, Y = np.meshgrid(x, y)
    zs = np.array([rate(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()