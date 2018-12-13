# from game_theory.server import file_name

import cProfile
import sys

from mesa.batchrunner import BatchRunner
from tqdm import trange
from game_theory.model import RPSModel, PDModel
from game_theory.config import Config
import numpy as np
import matplotlib.pyplot as plt
from game_theory.analysis import collect_dependent_raw_data, calculate_dependent, calculate_dependent_error, histogram_results

import statistics


def run_model(config, batchrunning):
    if batchrunning['variable_output']:
        dependent = []
        dependent_error = []
        histogram_data = []
        for variable in range(batchrunning['start'], batchrunning['stop'], batchrunning['step']):
            config[batchrunning['variable_name']] = variable
            dependent_raw_data = []
            for num in trange(batchrunning['num_sims_per_interval']):
                if config['game_type'] == "RPS":
                    model = RPSModel(config)
                else:
                    model = PDModel(config)
                model.run(batchrunning['num_steps'])
                dependent_raw_data.append(collect_dependent_raw_data(model, batchrunning['dependent_name']))
            dependent.append(calculate_dependent(dependent_raw_data, batchrunning['dependent_name']))
            dependent_error.append(calculate_dependent_error(dependent_raw_data,
                                                             batchrunning['dependent_name'],
                                                             batchrunning['num_sim_batches']))
        #     histogram_data.append(dependent_raw_data)
        # histogram_results(histogram_data, batchrunning['num_steps'])

        plt.show()
        print(dependent_raw_data)
        variable = np.arange(batchrunning['start'], batchrunning['stop'], batchrunning['step'])
        print(variable)
        print(dependent)
        print(dependent_error)
        plt.figure()
        plt.scatter(variable, dependent, c='b')
        plt.errorbar(variable, dependent, yerr=dependent_error, elinewidth=0.2, ecolor='b')
        plt.xlabel(batchrunning['variable_name'])
        plt.ylabel(batchrunning['dependent_name'])
        plt.xlim([batchrunning['start'], batchrunning['stop']])
        plt.ylim([0, 1.05])
        # plt.savefig('graphs/batchruns/dimension_extinction_probability.png')
        plt.show()
        print("-" * 10 + "\nSimulation finished!\n" + "-" * 10)
        # fft_analysis(model)
        # if config.game_mode == "Pure":
        #     labels = ["Pure Rock", "Pure Paper", "Pure Scissors"]
        #     ternary_plot(model, labels)
        # elif config.game_type == "Impure":
        #     labels = ["P(Rock)", "P(Paper)", "P(Scissors)"]
        #     ternary_plot(model, labels)

a = [1.0, 1.0, 0.95, 0.99, 0.98, 0.71, 0.57, 0.28, 0.19, 0.12, 0.1, 0.02]
b = [0.0, 0.0, 0.05270462766947298, 0.031622776601683784, 0.04216370213557838, 0.11005049346146122, 0.09486832980505136, 0.16193277068654824, 0.15951314818673865, 0.13165611772087665, 0.10540925533894598, 0.04216370213557839]


file_name = "game_theory/game_configs/rock_paper_scissors.json"
with open(file_name) as d:
    model_config = Config(d.read())

if model_config.parameters['simulation']:
    from game_theory.server import server
    server.port = 8521  # The default
    server.launch()
else:
    run_model(model_config.parameters, model_config.batchrunning)