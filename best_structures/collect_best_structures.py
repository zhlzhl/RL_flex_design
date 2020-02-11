import numpy as np
import os
import pickle
from pathlib import Path
import shutil


# looking for filenames starting with "best_eval_performance_n_structure", in sub-directory "simple_save999999" of
# sub-directory of specified directory


def collect_and_copy_best_structures():
    dir = os.getcwd()
    output_dir = dir

    path = Path(dir)

    data_dir = os.path.join(path.parent, 'data')

    for root, dir_list, _ in os.walk(data_dir):
        for dir in dir_list:
            if DIR_IDENTIFIER in dir:
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
                                                    shutil.copy2(file_to_copy, output_dir)
                                                    print('copied {}'.format(file_to_copy))


# a unique string to identify the set of directories to look into
DIR_IDENTIFIER = 'CENV0-8'
# a unique string to identify the name of the files to be copied
FILE_IDENTIFIER = 'best_eval_performance_n_structure'
# the name of the directory where the best structures are saved
SAVED_MODEL_DIR_NAME = ('simple_save999999', 'custom_save999999')

if __name__ == "__main__":
    collect_and_copy_best_structures()
    print("done")
