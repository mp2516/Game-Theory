# from game_theory.server import file_name

import cProfile
import sys

from mesa.batchrunner import BatchRunner
from tqdm import trange
from game_theory.model import RPSModel, PDModel
from game_theory.config import Config
import numpy as np
import matplotlib.pyplot as plt
from game_theory.analysis import fft_analysis, ternary_plot, calculate_extinction_time

import statistics


def run_model(config, batchrunning):
    if batchrunning['variable_output']:
        dependent_average = []
        dependent_y_err = []
        for variable in trange(batchrunning['start'], batchrunning['stop'], batchrunning['step']):
            config[batchrunning['variable']] = variable
            dependent = []
            for _ in range(batchrunning['num_sims_per_interval']):
                if config['game_type'] == "RPS":
                    model = RPSModel(config)
                else:
                    model = PDModel(config)
                model.run(batchrunning['num_steps'])
                dependent.append(calculate_extinction_time(model))
            dependent_average.append(statistics.mean(dependent))
            dependent_y_err.append(statistics.stdev(dependent))
        variable = np.arange(batchrunning['start'], batchrunning['stop'], batchrunning['step'])
        plt.figure()
        plt.errorbar(variable, dependent_average, yerr=dependent_y_err)
        plt.xlabel(batchrunning['variable'])
        plt.ylabel(batchrunning['dependent'])
        plt.show()
        print("-" * 10 + "\nSimulation finished!\n" + "-" * 10)
        # fft_analysis(model)
        # if config.game_mode == "Pure":
        #     labels = ["Pure Rock", "Pure Paper", "Pure Scissors"]
        #     ternary_plot(model, labels)
        # elif config.game_type == "Impure":
        #     labels = ["P(Rock)", "P(Paper)", "P(Scissors)"]
        #     ternary_plot(model, labels)


file_name = "game_theory/game_configs/rock_paper_scissors.json"
with open(file_name) as d:
    model_config = Config(d.read())

if model_config.parameters['simulation']:
    from game_theory.server import server
    server.port = 8521  # The default
    server.launch()
else:
    run_model(model_config.parameters, model_config.batchrunning)