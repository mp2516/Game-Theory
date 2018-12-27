# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 15:46:12 2018

@author: Dick
"""
import random

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def rate_equation():

    r_list = [0.2]
    p_list = [0.5]
    s_list = [0.3]

    def drdt1(r, p, s, alpha, beta):
        drdt = r * (1 - r - alpha * p - beta * s)
        return r + drdt + 0.001*random.uniform(-1, 1)
    def dpdt1(r, p, s, alpha, beta):
        dpdt = p * (1 - beta * r - p - alpha * s)
        return dpdt + p + 0.001*random.uniform(-1, 1)
    def dsdt1(r, p, s, alpha, beta):
        dsdt = s * (1 - alpha * r - beta * p - s)
        return dsdt + s + 0.001*random.uniform(-1, 1)

    def drdt2r(r, p, s, alpha, beta):
        drdt = r * (s - p)
        return r + 0.1 * drdt + 0.001*random.uniform(-1, 1)
    def dpdt2r(r, p, s, alpha, beta):
        dpdt = p * (r - s)
        return 0.1 * dpdt + p + 0.001*random.uniform(-1, 1)
    def dsdt2r(r, p, s, alpha, beta):
        dsdt = s * (p - r)
        return 0.1 * dsdt + s + 0.001*random.uniform(-1, 1)

    def drdt2(r, p, s, alpha, beta):
        drdt = r * (s - p)
        return r + 0.1 * drdt
    def dpdt2(r, p, s, alpha, beta):
        dpdt = p * (r - s)
        return 0.1 * dpdt + p
    def dsdt2(r, p, s, alpha, beta):
        dsdt = s * (p - r)
        return 0.1 * dsdt + s

    n = 0
    alpha = 0.999999
    beta = 0.999999

    model = [drdt2, dpdt2, dsdt2]


    while n < 5000:
        n+=1
        r_list.append(model[0](r_list[-1], p_list[-1], s_list[-1], alpha, beta))
        p_list.append(model[1](r_list[-2], p_list[-1], s_list[-1], alpha, beta))
        s_list.append(model[2](r_list[-2], p_list[-2], s_list[-1], alpha, beta))
        total = sum([r_list[-1], p_list[-1], s_list[-1]])
        if r_list[-1] < 0:
            r_list[-1] = 0
        if p_list[-1] < 0:
            p_list[-1] = 0
        if s_list[-1] < 0:
            s_list[-1] = 0
        if r_list[-1] > 1:
            r_list[-1] = 1
        if p_list[-1] > 1:
            p_list[-1] = 1
        if s_list[-1] > 1:
            s_list[-1] = 1

        r_list[-1] = r_list[-1]/total
        p_list[-1] = p_list[-1]/total
        s_list[-1] = s_list[-1]/total

    plt.plot(range(len(r_list)), r_list, label = "Rock")
    plt.plot(range(len(p_list)), p_list, label = "Paper")
    plt.plot(range(len(s_list)), s_list, label = "Scissors")
    plt.xlabel("Step number")
    plt.ylabel("Population density")
    plt.legend(loc="best")

    plt.ylim(-0.1, 1)
    plt.figure(figsize=(100,100))
    plt.show()

    #print("Rock \n", p_list)
    print(sum([r_list[-1], p_list[-1], s_list[-1]]))
    

