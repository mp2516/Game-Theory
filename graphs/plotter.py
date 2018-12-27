import numpy as np
import matplotlib.pyplot as plt


def graph(formula, x_max):
    x = np.array(np.arange(0, x_max, 0.1))
    y = formula(x)  # <- note now we're calling the function 'formula' with x
    fig, ax = plt.subplots()
    plt.plot(x, y)
    plt.axis('equal')
    plt.axis([0, 10, 0, 10])
    plt.show()

graph(lambda x: (x-1)/18, 10)