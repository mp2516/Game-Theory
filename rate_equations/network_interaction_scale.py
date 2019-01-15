import numpy as np
import matplotlib.pyplot as plt

def interaction_scale():
    x = np.linspace(0, 5, 1000)
    y_exp = np.exp(-1* x)
    def sigmoid(a, s):
        return 1 / (1 + np.exp(s*(x - a)))
    y_line = - 0.2 * x + 1
    y_mean = 1 - 0.00000000001 * x

    plt.figure()
    plt.plot(x, sigmoid(2, 5))
    plt.plot(x, sigmoid(1, 10000))
    plt.plot(x, y_mean)
    x = np.linspace(1, 5, 1000)
    plt.plot(x, y_line)
    plt.xlabel('Distance away from Agent')
    plt.ylabel('Interaction Strength')



    plt.show()