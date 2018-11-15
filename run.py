from game_theory.server import server

import cProfile
import sys

import numpy as np
import scipy
import matplotlib as plt
import ternary

from tqdm import trange
from game_theory.model import GameAgent


def fft_analysis():
    all_population_data = GameAgent.datacollector_populations.get_model_vars_dataframe()
    # surprisingly this iterates over columns, not rows
    for population_data in all_population_data:
        N = len(population_data)
        t_axis = np.linspace(0.0, 1.0 / (2.0), (int(N) / 2))
        y_axis = population_data - np.mean(population_data)
        y_axis_fft = scipy.fftpack.fft(y_axis)
        y_corrected = 2 / N * np.abs(y_axis_fft[0:np.int(N / 4)])
        t_corrected = t_axis[0:np.int(N / 4)]

        plt.figure(1)
        plt.plot(t_corrected, y_corrected, label='Dominant frequency = ' + str(round(t_corrected[np.argmax(y_corrected)], 4)) + ' $set^(-1)$')
        plt.xlabel('Frequency (set^-1)')
        plt.ylabel('FT of Population')
        plt.legend(loc='best')

        plt.figure(2, )
        plt.plot(np.arange(N), population_data)
        plt.xlabel('Set no')
        plt.ylabel('Population')

    plt.show()


def ternary_plot():
    figure, tax = ternary.figure(scale=1.0)
    tax.boundary()
    tax.gridlines(multiple=0.2, color="black")
    tax.set_title("Populations", fontsize=20)
    tax.left_axis_label("Scissors", fontsize=20)
    tax.right_axis_label("Paper", fontsize=20)
    tax.bottom_axis_label("Rock", fontsize=20)

    r_list_norm = [i / (l * l) for i in rock_list]
    p_list_norm = [i / (l * l) for i in paper_list]
    s_list_norm = [i / (l * l) for i in scissors_list]
    points = list(zip(r_list_norm, p_list_norm, s_list_norm))

    tax.plot(points, linewidth=2.0, label="Curve")
    tax.ticks(axis='lbr', multiple=0.2, linewidth=1)
    tax.legend()
    tax.show()

    print(np.argmax(yf))
    print("Dominant frequecy >> ", xf[np.argmax(yf)])



def run_model(config, n):
    model = GameAgent(config)
    for _ in trange(n):
        model.step()
    print("-" * 10 + "\nSimulation finished!\n" + "-" * 10)

    fft_analysis()
    # ternary_plot()

    for agent in model.schedule.agents:
        print("ID: {id}\n"
              "Average Score: {average_score}\n"
              "---------------------------".format(
            id=agent.unique_id,
            average_score=agent.total_score))

if len(sys.argv) > 1:
    file_name = 
    with open(file_name) as d:
        model_config = Config(d.read())

    number_of_steps = int(sys.argv[1])

    if len(sys.argv) > 2:
        cProfile.run('run_model(model_config, number_of_steps)')
    else:
        run_model(model_config, number_of_steps)


server.port = 8521 # The default
server.launch()