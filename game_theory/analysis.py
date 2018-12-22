import numpy as np
import scipy.fftpack
from matplotlib import pyplot as plt
import ternary
import random
import statistics
from game_theory.config import Config



def fft_analysis(model):
    all_population_data = model.datacollector_populations.get_model_vars_dataframe()
    # surprisingly this iterates over columns, not rows
    dominant_frequencies = []
    for population_data in all_population_data:
        N = len(all_population_data[population_data])
#        print(N)
#        print(all_population_data[population_data])
        t_axis = np.linspace(0.0, 1.0 / (2.0), (int(N) / 2))
        y_axis = all_population_data[population_data] - np.mean(all_population_data[population_data])
        y_axis_fft = scipy.fftpack.fft(y_axis)
        y_corrected = 2 / N * np.abs(y_axis_fft[0:np.int(N / 4)])
        t_corrected = t_axis[0:np.int(N / 4)]

        plt.figure(1)
        plt.plot(t_corrected, y_corrected, label='Dominant frequency = ' + str(round(t_corrected[np.argmax(y_corrected)], 4)) + ' $set^(-1)$' + str(population_data))
        plt.xlabel('Frequency (set^-1)')
        plt.ylabel('FT of Population')
        plt.legend(loc='best')

        plt.figure(2, )
        plt.plot(np.arange(N), all_population_data[population_data])
        plt.xlabel('Set no')
        plt.ylabel('Population')
        dominant_frequencies.append(t_corrected[np.argmax(y_corrected)])

    plt.show()
    print("Dominant frequency >> ", statistics.mean(dominant_frequencies[:3]))
    return statistics.mean(dominant_frequencies[:3])

def vortex_number_graph(model):
    all_vortex_data = model.datacollector_no_vortices.get_model_vars_dataframe()
    plt.figure(3,)
    plt.plot(range(len((all_vortex_data["Number of Vortices"]))), all_vortex_data["Number of Vortices"])
    plt.xlabel('Step no')
    plt.ylabel('Number of Vortices')
    plt.show()
    
def vortex_number_ave(model):
    all_vortex_data = model.datacollector_no_vortices.get_model_vars_dataframe()
    ave_end_vortex_number = statistics.mean(all_vortex_data["Number of Vortices"][all_vortex_data.index[-20:-1]])
    ave_end_vortex_number_sd = statistics.stdev(all_vortex_data["Number of Vortices"][all_vortex_data.index[-20:-1]])
    print("Average number of vortices at end ->", ave_end_vortex_number, "±", ave_end_vortex_number_sd)
    return ave_end_vortex_number

def dominant_freq(model):
    all_population_data = model.datacollector_populations.get_model_vars_dataframe()
    # surprisingly this iterates over columns, not rows
    dominant_frequencies = []
    for population_data in all_population_data:
        N = len(all_population_data[population_data])
        t_axis = np.linspace(0.0, 1.0 / (2.0), (int(N) / 2))
        y_axis = all_population_data[population_data] - np.mean(all_population_data[population_data])
        y_axis_fft = scipy.fftpack.fft(y_axis)
        y_corrected = 2 / N * np.abs(y_axis_fft[0:np.int(N / 4)])
        t_corrected = t_axis[0:np.int(N / 4)]
        dominant_frequency = t_corrected[np.argmax(y_corrected)]
        dominant_frequencies.append(t_corrected[np.argmax(y_corrected)])
    print("Average dominant frequency ->", statistics.mean(dominant_frequencies[:4]))
    return statistics.mean(dominant_frequencies[:3])

def ternary_plot(model, labels):
    all_population_data = model.datacollector_populations.get_model_vars_dataframe()
    for index, data in all_population_data.iterrows():
        list_norm = [(i / model.height ** 2) for i in all_population_data[population_data]]
        print(list_norm)
        points = list(zip(list_norm))
        print(points)

    figure, tax = ternary.figure(scale=1.0)
    tax.boundary()
    tax.gridlines(multiple=0.2, color="black")
    tax.set_title("Populations", fontsize=20)
    tax.left_axis_label("Scissors", fontsize=20)
    tax.right_axis_label("Paper", fontsize=20)
    tax.bottom_axis_label("Rock", fontsize=20)

    tax.plot(points, linewidth=2.0, label="Curve")
    tax.ticks(axis='lbr', multiple=0.2, linewidth=1)
    tax.legend()
    tax.show()


def histogram(model):
    all_population_scores = model.datacollector_population_scores.get_model_vars_dataframe()
    indexes = all_population_scores.columns
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


def histogram_results(raw_data, num_steps):

    # filtered_raw_data = list(filter(lambda a: a != np.inf, raw_data))
    # print(filtered_raw_data)
    bins = np.arange(num_steps+1)
    # bins = np.linspace(0, num_steps+1, 30)
    plt.hist(raw_data, bins, label=("2", "3", "4"))
    plt.xlabel("Extinction Time")
    plt.ylabel("Number of sims")
    plt.legend(loc="best")


def pie_chart(model):
    all_population_data = model.datacollector_populations.get_model_vars_dataframe()
    current_populations = []
    for population_data in all_population_data:
        populations = all_population_data[population_data]
        current_populations.append(populations[model.number_of_steps])
    #   current_populations = [a[model.number_of_steps - 1] for a in [population_data for population_data in all_population_data]]
    plt.figure(4, )
    plt.pie(current_populations, labels=[a for a in all_population_data.columns])
    
def count_vortices(model):
    positions = []
    vortex_count = 0
    agent_list = [agent for agent in model.schedule.agents]
    if model.periodic_BC:
        for agent in model.schedule.agents:
            neighbor_list = [neighbor for neighbor in agent.neighbors]
            if agent.strategy == neighbor_list[1].strategy and agent.strategy != neighbor_list[0].strategy and agent.strategy != neighbor_list[3].strategy and neighbor_list[0].strategy != neighbor_list[3].strategy:
                vortex_count += 1
            elif agent.strategy == neighbor_list[3].strategy and agent.strategy != neighbor_list[0].strategy and agent.strategy != neighbor_list[1].strategy and neighbor_list[0].strategy != neighbor_list[1].strategy:
                vortex_count += 1
            elif agent.strategy == neighbor_list[3].strategy and agent.strategy != neighbor_list[6].strategy and agent.strategy != neighbor_list[5].strategy and neighbor_list[5].strategy != neighbor_list[6].strategy:
                vortex_count += 1
            elif agent.strategy == neighbor_list[6].strategy and agent.strategy != neighbor_list[3].strategy and agent.strategy != neighbor_list[5].strategy and neighbor_list[3].strategy != neighbor_list[5].strategy:
                vortex_count += 1
            elif agent.strategy == neighbor_list[6].strategy and agent.strategy != neighbor_list[4].strategy and agent.strategy != neighbor_list[7].strategy and neighbor_list[4].strategy != neighbor_list[7].strategy:
                vortex_count += 1
            elif agent.strategy == neighbor_list[4].strategy and agent.strategy != neighbor_list[1].strategy and agent.strategy != neighbor_list[2].strategy and neighbor_list[1].strategy != neighbor_list[2].strategy:
                vortex_count += 1
            elif agent.strategy == neighbor_list[4].strategy and agent.strategy != neighbor_list[6].strategy and agent.strategy != neighbor_list[7].strategy and neighbor_list[6].strategy != neighbor_list[7].strategy:
                vortex_count += 1
            elif agent.strategy == neighbor_list[1].strategy and agent.strategy != neighbor_list[2].strategy and agent.strategy != neighbor_list[4].strategy and neighbor_list[2].strategy != neighbor_list[4].strategy:
                vortex_count += 1
                
    elif not model.periodic_BC:
        for agent in model.schedule.agents:
            if agent.pos[0] != 0 and agent.pos[0] != model.dimension-1 and agent.pos[1] != 0 and agent.pos[1] != model.dimension-1:
                neighbor_list = [neighbor for neighbor in agent.neighbors]
                if agent.strategy == neighbor_list[1].strategy and agent.strategy != neighbor_list[0].strategy and agent.strategy != neighbor_list[3].strategy and neighbor_list[0].strategy != neighbor_list[3].strategy:
                    vortex_count += 1
                elif agent.strategy == neighbor_list[3].strategy and agent.strategy != neighbor_list[0].strategy and agent.strategy != neighbor_list[1].strategy and neighbor_list[0].strategy != neighbor_list[1].strategy:
                    vortex_count += 1
                elif agent.strategy == neighbor_list[3].strategy and agent.strategy != neighbor_list[6].strategy and agent.strategy != neighbor_list[5].strategy and neighbor_list[5].strategy != neighbor_list[6].strategy:
                    vortex_count += 1
                elif agent.strategy == neighbor_list[6].strategy and agent.strategy != neighbor_list[3].strategy and agent.strategy != neighbor_list[5].strategy and neighbor_list[3].strategy != neighbor_list[5].strategy:
                    vortex_count += 1
                elif agent.strategy == neighbor_list[6].strategy and agent.strategy != neighbor_list[4].strategy and agent.strategy != neighbor_list[7].strategy and neighbor_list[4].strategy != neighbor_list[7].strategy:
                    vortex_count += 1
                elif agent.strategy == neighbor_list[4].strategy and agent.strategy != neighbor_list[1].strategy and agent.strategy != neighbor_list[2].strategy and neighbor_list[1].strategy != neighbor_list[2].strategy:
                    vortex_count += 1
                elif agent.strategy == neighbor_list[4].strategy and agent.strategy != neighbor_list[6].strategy and agent.strategy != neighbor_list[7].strategy and neighbor_list[6].strategy != neighbor_list[7].strategy:
                    vortex_count += 1
                elif agent.strategy == neighbor_list[1].strategy and agent.strategy != neighbor_list[2].strategy and agent.strategy != neighbor_list[4].strategy and neighbor_list[2].strategy != neighbor_list[4].strategy:
                    vortex_count += 1


    agent_pos = [agent.pos for agent in model.schedule.agents]
    return vortex_count/2


def collect_dependent_raw_data(model, dependent_name):
    if dependent_name == "extinction_probability" or dependent_name == "extinction_time":
        return calculate_extinction_time(model)
    elif dependent_name == "environment_death" or dependent_name == "biodiversity_lifetime":
        step_num = calculate_environment_death(model)
        if step_num == np.inf:
            step_num = calculate_extinction_time(model)
        return step_num


def calculate_dependent(raw_data, dependent_name):
    if dependent_name == "extinction_probability" or dependent_name == "environment_death":
        return calculate_dependent_prob(raw_data)
    if dependent_name == "biodiversity_lifetime":
        return calculate_dependent_ave(raw_data)


def calculate_dependent_error(raw_data, dependent_name, num_sim_batches):
    if dependent_name == "extinction_probability" or dependent_name == "environment_death":
        return calculate_dependent_prob_sd(raw_data, num_sim_batches)
    elif dependent_name == "biodiversity_lifetime":
        return calculate_dependent_ave_sd(raw_data, num_sim_batches)


def calculate_extinction_time(model):
    """
    :param model:
    :return: np.inf if the population remains stable for the whole simulation, step_num if the population reaches
        extinction after a certain period of time.
    """
    all_population_data = model.datacollector_populations.get_model_vars_dataframe()
    # we only need to loop through one population data set
    for population_data in all_population_data:
        populations = all_population_data[population_data]
        break
    for step_num, i in enumerate(populations):
        if i == (model.dimension ** 2) or i == 0:
            # there is one homogenous population
            # if it is homogenous we can break out of the loop
            return step_num
    return np.inf


def calculate_stable_transient(model):

    all_population_data = model.datacollector_populations.get_model_vars_dataframe()
    for population_data in all_population_data:
        populations = all_population_data[population_data]
        for step_num, i in enumerate(populations):
            if np.ceil(populations[0] * model.transient_threshold) >= populations[i]\
                    or np.floor(populations[0] * (1 - model.transient_threshold)) <= populations[i]:
                return step_num


def calculate_dependent_prob(raw_data):
    dependent_successful = list(filter(lambda a: a != np.inf, raw_data))
    return len(dependent_successful) / len(raw_data)

def calculate_dependent_ave(raw_data):
    dependent_successful = list(filter(lambda a: a != np.inf, raw_data))
    if len(dependent_successful) > 1:
        return sum(dependent_successful) / len(dependent_successful)
    return 0

def calculate_dependent_prob_sd(raw_data, num_sim_batches):

    def chunk_it(seq, num):
        avg = len(seq) / float(num)
        out = []
        last = 0.0
        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg
        return out

    # raw_data_filtered = list(filter(lambda a: a != np.inf, raw_data))
    split_raw_data = chunk_it(raw_data, num_sim_batches)
    split_extinction_prob = []
    for split in split_raw_data:
        split_extinction_prob.append(calculate_dependent_prob(split))
    return statistics.stdev(split_extinction_prob)

def calculate_dependent_ave_sd(raw_data, num_sim_batches):
    dependent_successful = list(filter(lambda a: a != np.inf, raw_data))
    if len(dependent_successful) > 1:
        return statistics.stdev(dependent_successful)
    return 0


def calculate_environment_death(model):
    all_mutating_agents = model.datacollector_mutating_agents.get_model_vars_dataframe()
    for mutating_agents in all_mutating_agents:
        mutating = all_mutating_agents[mutating_agents]
        break
    for step_num, i in enumerate(mutating):
        if i == 0 and step_num > 0:
            return step_num
    return np.inf
