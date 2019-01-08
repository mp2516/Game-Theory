# from game_theory.server import file_name

import cProfile
import sys

from mesa.batchrunner import BatchRunner
from tqdm import trange
from game_theory.model import RPSModel
from game_theory.config import Config
import numpy as np
import matplotlib.pyplot as plt
from game_theory.analysis import * #branched, is_dead, collect_dependent_raw_data, calculate_dependent, calculate_dependent_error, histogram_results, count_vortices, fft_analysis, vortex_number_graph, vortex_number_ave, dominant_freq

import statistics


def run_model(config, batchrunning):
    if batchrunning['variable_output']:
        dependent = []
        dependent_error = []
        histogram_data = []
        all_dependent_raw = []
        for variable in np.arange(batchrunning['start'], batchrunning['stop'], batchrunning['step']):
            #np.arange(batchrunning['start'], batchrunning['stop'], batchrunning['step'])
            #np.logspace(batchrunning['start'], batchrunning['stop'], batchrunning['intervals'])
            config[batchrunning['variable_name']] = variable
            dependent_raw_data = []
            for num in trange(batchrunning['num_sims_per_interval']):
                if config['game_type'] == "RPS":
                    model = RPSModel(config)
                else:
                    model = PDModel(config)
                for i in range(batchrunning['num_steps']):
                    model.step()
                    if is_dead(model) or is_marching(model):
                        break
#                    if batchrunning['dependent_name'] == "bifurcation_time" and branched(model):
#                        break

#                model.run(batchrunning['num_steps'])
#                dependent_raw_data.append(collect_dependent_raw_data(model, batchrunning['dependent_name']))
#                if batchrunning['dependent_name'] == "bifurcation_time" and not is_dead(model):
#                    dependent_raw_data.append(collect_dependent_raw_data(model, batchrunning['dependent_name']))
#                else:
                dependent_raw_data.append(collect_dependent_raw_data(model, batchrunning['dependent_name']))
                print('\n', dependent_raw_data)
#                print("Number of Vortices ->", count_vortices(model))
#                vortex_number_graph(model)
#                fft_analysis(model)
#                print(dependent_raw_data)
#            vortex_number_ave(model)
#                print(dominant_freq(model))
#            print(dependent_raw_data)
            all_dependent_raw.append(dependent_raw_data)
            print(all_dependent_raw)
#            dependent_mean = statistics.mean(dependent_raw_data)
#            dependent_sd = statistics.stdev(dependent_raw_data)
            dependent_raw_data_tidy = [i for i in dependent_raw_data if i >= 0.07]
            dependent.append(calculate_dependent(dependent_raw_data_tidy, batchrunning['dependent_name']))
            dependent_error.append(calculate_dependent_error(dependent_raw_data_tidy,
                                                             batchrunning['dependent_name'],
                                                             batchrunning['num_sim_batches']))
            variable = np.arange(batchrunning['start'], batchrunning['stop'], batchrunning['step'])
            print(variable.tolist(), '\n length', len(variable.tolist()))
            print(dependent, '\n length', len(dependent))
            print(dependent_error)
            
#            histogram_data.append(dependent_raw_data)
#        histogram_results(histogram_data, batchrunning['num_steps'])
            plt.figure(figsize=(10,5))
            plt.scatter(variable[:len(dependent)], dependent, c='b')
            plt.errorbar(variable[:len(dependent)], dependent, yerr=dependent_error, elinewidth=0.2, ecolor='b')
            plt.xlabel(batchrunning['variable_name'])
            plt.ylabel(batchrunning['dependent_name'])
#            plt.xlim([batchrunning['start'] - 0.01, batchrunning['stop']])
            plt.xscale('log')
            plt.grid(True)
            plt.show()
#            plt.ylim([0, batchrunning['num_steps']])
            # plt.savefig('graphs/batchruns/dimension_extinction_probability.png')

        print(dependent_raw_data)
        variable = np.arange(batchrunning['start'], batchrunning['stop'], batchrunning['step'])
        print(str(batchrunning['variable_name']) + ' =', variable.tolist())
        print(str(batchrunning['dependent_name']) + ' =', dependent)
        print(str(batchrunning['dependent_name']) + '_sd' + ' =', dependent_error)
        print(histogram_data)
        plt.figure(figsize=(10,5))
        plt.scatter(variable, dependent, c='b')
        plt.errorbar(variable, dependent, yerr=dependent_error, elinewidth=0.2, ecolor='b')
        plt.xlabel(batchrunning['variable_name'])
        plt.ylabel(batchrunning['dependent_name'])
#        plt.xlim([batchrunning['start'] - 0.0001, batchrunning['stop']])
        plt.grid(True)
        plt.xscale('log')
#        plt.ylim([0, batchrunning['num_steps']])
        # plt.savefig('graphs/batchruns/dimension_extinction_probability.png')
        plt.rc('font', size=14)
        plt.show()
        fft_analysis(model)
        vortex_number_graph(model)
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
    server.port = 8523  # The default
    server.launch()
else:
    run_model(model_config.parameters, model_config.batchrunning)