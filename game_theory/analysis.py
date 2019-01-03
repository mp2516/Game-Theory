import numpy as np
import scipy.fftpack
from matplotlib import pyplot as plt
import pprint
import ternary
import random
import statistics
from game_theory.logger import logger

from tabulate import tabulate
from tqdm import trange

from game_theory.model import RPSModel


def chunk_it(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg
    return out


class BatchRunner:
    def __init__(self, config_batchrunning, config_model):

        self.variable_output = config_batchrunning['variable_output']

        self.num_steps = config_batchrunning['num_steps']
        self.variable_name = config_batchrunning['variable_name']
        self.dependent_name = config_batchrunning['dependent_name']
        self.start = config_batchrunning['start']
        self.stop = config_batchrunning['stop']
        self.step = config_batchrunning['step']
        self.num_sims_per_interval = config_batchrunning['num_sims_per_interval']
        self.num_sim_batches = config_batchrunning['num_sim_batches']

        self.config_model = config_model

        self.all_population_data = None
        self.all_score_data = None
        self.all_evolving_data = None

        if self.config_model['probability_mutation'] > 0:
            self.all_mutating_data = None

        self.dependent = []
        self.dependent_error = []
        self.dependent_raw_data = []
        self.histogram_data = []

        self.variable = np.arange(self.start, self.stop, self.step)

        self.num_steps = config_batchrunning['num_steps']
        self.transient_threshold = config_batchrunning['transient_threshold']

        self.colour_scheme = {"Rock": "red", "Paper": "blue", "Scissors": "green"}

    def run_model(self):
        if self.variable_output:
            for variable in range(self.start, self.stop, self.step):
                self.config_model[self.variable_name] = variable
                for _ in trange(self.num_sims_per_interval):
                    model = RPSModel(self.config_model)
                    model.run(self.num_steps)

                    self.all_population_data = model.datacollector_population.get_model_vars_dataframe()
                    self.all_score_data = model.datacollector_score.get_model_vars_dataframe()
                    self.all_evolving_data = model.datacollector_evolving_agents.get_model_vars_dataframe()

                    if self.config_model['probability_mutation'] > 0:
                        self.all_mutating_data = model.datacollector_mutating_agents.get_model_vars_dataframe()

                    self.dependent_raw_data.append(self.collect_dependent_raw_data(model))

                self.dependent.append(self.calculate_dependent())
                self.dependent_error.append(self.calculate_dependent_error())

                self.histogram_data.append(self.dependent_raw_data)

            self.histogram_results()
        else:
            model = RPSModel(self.config_model)
            model.run(self.num_steps)

            self.all_population_data = model.datacollector_population.get_model_vars_dataframe()
            self.all_score_data = model.datacollector_score.get_model_vars_dataframe()
            self.all_evolving_data = model.datacollector_evolving_agents.get_model_vars_dataframe()

            self.ternary_plot(model)
            self.fft_analysis()


    def plot_scatter(self):
        plt.figure()
        self.variable = np.arange(self.start, self.stop, self.step)
        plt.scatter(self.variable, self.dependent, c='b')
        plt.errorbar(self.variable, self.dependent, yerr=self.dependent_error, elinewidth=0.2, ecolor='b')
        plt.xlabel(self.variable_name)
        plt.ylabel(self.dependent_name)
        plt.xlim([self.start, self.stop])
        plt.ylim([0, 1.05])
        plot_name = str(self.variable_name) + '_' + str(self.dependent_name) + '_' + str(
            self.num_steps) + 'n' + '_' + str(self.num_sims_per_interval) + 'sims'
        plt.savefig('graphs/batchruns/' + plot_name + '.png')
        plt.show()

    def collect_dependent_raw_data(self, model):
        self.all_population_data = model.datacollector_populations.get_model_vars_dataframe()
        if self.dependent_name in {"extinction_probability", "extinction_time"}:
            return self.calculate_extinction_time(model)
        elif self.dependent_name in {"environment_death"}:
            self.all_evolving_data = model.datacollector_evolving_agents.get_model_vars_dataframe()
            step_num = self.calculate_environment_death(model)
            if step_num == np.inf:
                step_num = self.calculate_extinction_time(model)
            return step_num

    def calculate_dependent(self):
        if self.dependent_name in {"extinction_probability", "environment_death"}:
            return len(list(filter(lambda a: a != np.inf, self.dependent_raw_data))) / self.num_sims_per_interval

    def calculate_dependent_error(self):
        if self.dependent_name in {"extinction_probability", "environment_death"}:
            split_extinction_prob = []
            for split in chunk_it(self.dependent_raw_data, self.num_sim_batches):
                split_extinction_prob.append(len(list(filter(lambda a: a != np.inf, split))) / len(split))
            return statistics.stdev(split_extinction_prob)

    def calculate_extinction_time(self, model):
        # we only need to loop through one population data set
        extinction_time = {}
        for population_label in self.all_population_data:
            population = self.all_population_data[population_label]
            for step_num, i in enumerate(population):
                if i == (model.dimension ** 2) or i == 0:
                    # there is one homogenous population
                    # if it is homogenous we can break out of the loop
                    extinction_time[population_label] = step_num
                    break

        return False, np.inf

    def calculate_dominance_percentage_extinction(self, model):
        extinction = False
        for population_data_label in self.all_population_data:
            population = self.all_population_data[population_data_label]
            for step_num, i in enumerate(population):
                if i == 0:
                    extinction_step = step_num
                    extinction = True
                    break
        if extinction:
            reverse_population = population[0:extinction_step].reverse()
            reverse_maximum_index = reverse_population.index(max(reverse_population))
            maximum_index = extinction_step - reverse_maximum_index
            percentage_dominance = population[maximum_index] / (model.dimension ** 2)
            return percentage_dominance
        else:
            return None




    def calculate_stable_transient(self, model):
        for population_data in self.all_population_data:
            populations = self.all_population_data[population_data]
            for step_num, i in enumerate(populations):
                if np.ceil(populations[0] * self.transient_threshold) >= populations[i]\
                        or np.floor(populations[0] * (1 - self.transient_threshold)) <= populations[i]:
                    return step_num

    def calculate_environment_death(self):
        for evolving_agents in self.all_evolving_data:
            evolving = self.all_evolving_data[evolving_agents]
            break
        for step_num, i in enumerate(evolving):
            if i == 0 and step_num > 0:
                return step_num
        return np.inf

    def fft_analysis(self):
        for population_data_label in self.all_population_data:
            color = self.colour_scheme[population_data_label]
            population_data = self.all_population_data[population_data_label]
            y_fft = 2 / self.num_steps\
                          * np.abs(scipy.fftpack.fft(population_data - np.mean(population_data))
                                   [0:np.int(self.num_steps / 4)])
            t_corrected = np.linspace(0.0, 0.5, (self.num_steps / 2))[0:np.int(self.num_steps / 4)]

            # plot the fourier transform
            plt.figure(1, )
            plt.plot(t_corrected, y_fft,
                     label='Dominant frequency = '
                           + str(round(t_corrected[np.argmax(y_fft)], 4))
                           + ' $set^(-1)$'
                           + str(population_data_label),
                     color=color)
            plt.xlabel('Frequency (set^-1)')
            plt.ylabel('FT of Population')
            plt.legend(loc='best')

            # plot the actual populations
            plt.figure(2, )
            plt.plot(np.arange(self.num_steps), population_data,
                     label=str(population_data_label) + ' : ' + color,
                     color=color)
            plt.xlabel('Step Number')
            plt.ylabel('Population')
            plt.legend(loc='best')
        plt.show()

    def histogram(self, model):
        all_population_scores = model.datacollector_scores.get_model_vars_dataframe()
        indexes = all_population_scores.columns
        current_population_scores = []
        bins = np.arange((-1) * model.num_moves_per_set * 8, model.num_moves_per_set * 8)
        for a in indexes:
            population_scores = all_population_scores[a]
            current_population_scores.append(population_scores[model.number_of_steps - 1])
        labels = [a for a in all_population_scores.columns]
    #    print(labels)
        plt.figure(3,)
        plt.hist(current_population_scores, bins, label=labels, stacked=True)
        plt.xlabel("Score")
        plt.ylabel("Score density")
        plt.legend(loc="best")
        for population_scores in all_population_scores:
           print("population scores>>")
           print(all_population_scores[population_scores])

        for population_scores in all_population_scores:
            print("population scores>>", all_population_scores[population_scores])
            plt.hist(all_population_scores[population_scores], bins=4)
        bins=model.num_moves_per_set*16+1
        plt.show()

    def ternary_plot(self, model):
        points = self.all_population_data.values
        points.tolist()
        points = [(row[0] / (model.height * model.width),
                   row[1] / (model.height * model.width),
                   row[2] / (model.height * model.width)) for row in points]
        pp = pprint.PrettyPrinter(indent=4)
        print("The normalised populations:")
        pp.pprint(points)

        fig, tax = ternary.figure(scale=1.0)
        tax.boundary()
        tax.gridlines(multiple=0.1, color="black")
        tax.set_title("Populations", fontsize=20)
        tax.left_axis_label("Scissors", fontsize=20)
        tax.right_axis_label("Paper", fontsize=20)
        tax.bottom_axis_label("Rock", fontsize=20)

        tax.plot(points, linewidth=2.0, label="Curve")
        tax.ticks(axis='lbr', multiple=0.2, linewidth=1, tick_formats="%.1f")

        tax.clear_matplotlib_ticks()
        tax.legend()
        tax.show()

    def histogram(self):
        # all_population_scores = model.datacollector_population_scores.get_model_vars_dataframe()
        indexes = self.all_population_scores.columns
        current_population_scores = []
        bins = np.arange((-1) * model.num_moves_per_set * 8, model.num_moves_per_set * 8)
        for a in indexes:
            population_scores = all_population_scores[a]
            current_population_scores.append(population_scores[model.number_of_steps - 1])
        labels = [a for a in all_population_scores.columns]
        print(labels)
        plt.figure(3)
        plt.hist(current_population_scores, bins, label=labels, stacked=True)
        plt.xlabel("Score")
        plt.ylabel("Score density")
        plt.legend(loc="best")
        for population_scores in all_population_scores:
            print("population scores>>")
            print(all_population_scores[population_scores])

    def histogram_results(self, raw_data, num_steps):
        # filtered_raw_data = list(filter(lambda a: a != np.inf, raw_data))
        # print(filtered_raw_data)
        bins = np.arange(num_steps+1)
        # bins = np.linspace(0, num_steps+1, 30)
        plt.hist(raw_data, bins, label=("2", "3", "4"))
        plt.xlabel("Extinction Time")
        plt.ylabel("Number of sims")
        plt.legend(loc="best")

    def pie_chart(self, model):
        current_populations = []
        for population_data in self.all_population_data:
            populations = self.all_population_data[population_data]
            current_populations.append(populations[model.number_of_steps])
        #   current_populations = [a[model.number_of_steps - 1] for a in [population_data for population_data in all_population_data]]
        plt.figure(4, )
        plt.pie(current_populations, labels=[a for a in all_population_data.columns])