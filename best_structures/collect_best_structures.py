import numpy as np
import os
import pickle
from pathlib import Path
import shutil
from best_structures.exp_settings import *


# looking for filenames starting with "best_eval_performance_n_structure", in sub-directory "simple_save999999" of
# sub-directory of specified directory


def collect_and_copy_best_structures():
    dir = os.getcwd()
    output_dir = dir

    path = Path(dir)

    data_dir = os.path.join(path.parent, 'data')

    count = 0
    for root, dir_list, _ in os.walk(data_dir):
        for dir in dir_list:
            # if DIR_IDENTIFIER in dir:
            if all([identifier in dir for identifier in DIR_IDENTIFIER]):
                if all([identifier not in dir for identifier in EXCLUDE]):
                    for sub_root, sub_dir_list, _ in os.walk(os.path.join(root, dir)):  # level of a particular exp_s0 dir
                        if len(sub_dir_list) > 0:
                            for sub_dir in sub_dir_list:
                                for ss_root, ss_dir_list, _ in os.walk(
                                        os.path.join(sub_root, sub_dir)):  # level of 'sample_save's
                                    for ss_dir in ss_dir_list:
                                        if ss_dir in SAVED_MODEL_DIR_NAME:
                                            for sss_root, sss_dir_list, sss_file_list in os.walk(
                                                    os.path.join(sub_root, sub_dir)):
                                                for file in sss_file_list:
                                                    if FILE_IDENTIFIER in file:
                                                        # found a file, copy it to the best_structure folder
                                                        file_to_copy = os.path.join(sss_root, file)

                                                        if 'ENV' in file_to_copy:
                                                            output_file = os.path.join(output_dir,
                                                                                       "{}_ENV{}_{}.pkl".format(
                                                                                           file.split('.pkl')[0],
                                                                                           _get_env_version(file_to_copy),
                                                                                           count))
                                                        else:
                                                            output_file = os.path.join(output_dir,
                                                                                       "{}_{}.pkl".format(
                                                                                           file.split('.pkl')[0],
                                                                                           count))
                                                        count += 1
                                                        shutil.copy2(file_to_copy, output_file)
                                                        print('{} | copied {}'.format(count, file_to_copy))
                                                        print('    to {}'.format(output_file))


def _get_env_version(file):
    env_version = file.split('ENV')[1]
    env_version = env_version.split('_')[0]
    return env_version


if __name__ == "__main__":
    collect_and_copy_best_structures()
    print("done")
