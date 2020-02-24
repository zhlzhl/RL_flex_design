import numpy as np
import os
import pickle
from spinup.heuristics.flexibilityheuristics import *
from best_structures.exp_settings import *


def load_experiment(experiment):
    dir = os.getcwd()
    files = os.listdir(dir)
    files.sort()
    best_structures_dict = dict()

    for file in files:
        if os.path.isfile(file) and (experiment in file):
            file_path = os.path.join(dir, file)
            with open(file_path, 'rb') as f:
                best_performance, best_structure, epoch = pickle.load(f)

                # split the file name to get the target_arc of the experiment. example file name is:
                # best_eval_performance_n_structure_Flexibility10x10T16-v0.pkl. the target_arc is the 2-digit value
                # after 10x10T
                target_arc_substring = file.split(experiment + 'T', 1)[1]
                target_arc = target_arc_substring[0:2]

                # add the best_structure to the best_structure dict, key is target_arc in str format
                if target_arc in best_structures_dict:
                    best_structures_dict[target_arc].append(best_structure)
                else:
                    best_structures_dict[target_arc] = [best_structure]

                print("Experiment: {} | target_arc: {} | best_performance: {} | epoch: {}"
                      .format(experiment, target_arc, best_performance, epoch))

    return best_structures_dict


def gen_use_subtract(initial_design, capacity_vec, demand_vec, profit_mat, num_arcs):
    num_drop = int(sum(sum(initial_design))) - num_arcs

    assert num_drop > 0, "number of arcs to drop {} is not larger than zero. ".format(num_drop)

    design = initial_design.copy()

    for k in range(num_drop):
        (tosubtract, profitperf) = subtract(design, capacity_vec, demand_vec, profit_mat)
        design[tosubtract[0], tosubtract[1]] = 0
    return design


if __name__ == "__main__":
    experiment = EXPERIMENT  # e.g. 10x10
    best_structures = load_experiment(experiment)

    n = int(experiment.split('x')[0])  # e.g. 10
    profit_mat = np.ones((n, n))
    full_flex = np.ones((n, n))
    mean_c = 100 * np.ones(n)  # capacity vector
    mean_d = sum(mean_c) / n * np.ones(n)  # mean demand vector
    sd_d = mean_d * 0.8  # standard dAeviation vector (if needed)
    num_samples = 5000

    cap_vec = np.tile(np.array(mean_c), (num_samples, 1)).transpose()
    demand_train = np.zeros((n, num_samples))
    demand_test = np.zeros((n, num_samples))

    np.random.seed(10)
    for i in range(num_samples):
        demand = np.random.normal(mean_d, sd_d)
        # truncate demand at two standard deviations
        demand = np.maximum(demand, 0)
        demand = np.minimum(demand, mean_d + 2 * sd_d)
        demand_train[:, i] = demand

        demand = np.random.normal(mean_d, sd_d)
        # truncate demand at two standard deviations
        demand = np.maximum(demand, 0)
        demand = np.minimum(demand, mean_d + 2 * sd_d)
        demand_test[:, i] = demand

    num_arcs_iter = sorted(best_structures.keys(), reverse=True)
    subtract_structs = {}
    last_flex = full_flex
    drop_perf_dir = os.getcwd() + '/drop_benchmarks'
    if not os.path.exists(drop_perf_dir):
        os.mkdir(drop_perf_dir)

    # check whether drop structure have created or not. if exists, load it, otherwise, generate it.
    for index in num_arcs_iter:
        pickle_file = "{}/drop_performances_{}T{}".format(drop_perf_dir, experiment, index)
        if os.path.exists(pickle_file):
            with open(pickle_file, 'rb') as f:
                subtract_structs[index] = pickle.load(f)
                print("Loaded DROP structure for exp {} target_arc {}".format(experiment, index))
        else:
            print("Creating structure using DROP heuristic for exp {} target_arc {}".format(experiment, int(index)))
            subtract_structs[index] = gen_use_subtract(last_flex, cap_vec, demand_train, profit_mat, int(index))
            last_flex = subtract_structs[index]
            with open(pickle_file, 'wb') as f:
                pickle.dump(subtract_structs[index], f)
            print("Created and saved.")

        last_flex = subtract_structs[index]

    lei_perfs = {}
    drop_perfs = {}
    print('Performance evaluating ... ')
    for index, struc in best_structures.items():
        lei_perfs[index] = evaluate(struc, cap_vec, demand_test, profit_mat)
        drop_perfs[index] = evaluate(subtract_structs[index], cap_vec, demand_test, profit_mat)
        print("index: {} | RL: {} | drop: {}".format(index, lei_perfs[index], drop_perfs[index]))

    with open("RSvsDrop.pickle", 'wb') as f:
        pickle.dump((lei_perfs, drop_perfs), f)
    print("Performance evaluation is done. Saved to RSvsDrop.pickle.")

    fig, ax = plt.subplots()
    ax.plot(list(lei_perfs.keys()), list(lei_perfs.values()))
    ax.plot(list(drop_perfs.keys()), list(drop_perfs.values()), 'r')

    # ax.set(xlabel='sparsity', ylabel='avg profit', title='Comparison')
    ax.grid()

    fig.savefig("RSvsDROP.png")
    print('Plotted and saved figuer to RSvsDROP.png.')
    plt.show()
