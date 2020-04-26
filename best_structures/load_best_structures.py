import numpy as np
import os
import pickle
from best_structures.exp_settings import *


def load_experiment(experiment):
    dir = os.getcwd()
    files = os.listdir(dir)
    files.sort()
    best_structures_dict = dict()

    for file in files:
        if experiment in file:
            file_path = os.path.join(dir, file)

            if os.path.isfile(file_path):
                with open(file_path, 'rb') as f:
                    best_performance, best_structure, epoch = pickle.load(f)

                    # split the file name to get the target_arc of the experiment. example file name is:
                    # best_eval_performance_n_structure_Flexibility10x10T16-v0.pkl. the target_arc is the 2-digit value
                    # after 10x10T
                    target_arc_substring = file.split(experiment + 'T', 1)[1]
                    target_arc = target_arc_substring[0:2]

                    # add the best_structure to the best_structure dict, key is target_arc in str format
                    best_structures_dict[target_arc] = best_structure

                    if 'ENV3' in file:
                        print("Experiment: {} | target_arc: {} | {} | epoch: {} | best_performance: {} "
                              .format(experiment, target_arc, 'ENV3', epoch, best_performance))
                    elif 'ENV4' in file:
                        print("Experiment: {} | target_arc: {} | {} | epoch: {} | best_performance: {}"
                              .format(experiment, target_arc, 'ENV4', epoch, best_performance))
                    else:
                        print("Experiment: {} | target_arc: {} | epoch: {} | best_performance: {}"
                              .format(experiment, target_arc, best_performance, epoch))

    return best_structures_dict


if __name__ == "__main__":
    best_structures = load_experiment(EXPERIMENT)
