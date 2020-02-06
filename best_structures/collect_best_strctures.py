import numpy as np
import os
import pickle
from pathlib import Path
import shutil


# looking for filenames starting with "best_eval_performance_n_structure", in sub-directory "simple_save999999" of
# sub-directory of specified directory

# a unique string to identify the set of directories to look into
dir_identifier = 'CENV'
file_identifier = 'best_eval_performance_n_structure'

dir = os.getcwd()
output_dir = dir


path = Path(dir)

data_dir = os.path.join(path.parent, 'data')

for root, dir_list, _ in os.walk(data_dir):
    for dir in dir_list:
        if dir_identifier in dir:
            for sub_root, sub_dir_list, _ in os.walk(os.path.join(root, dir)):  # level of a particular exp_s0 dir
                if len(sub_dir_list) > 0:
                    for sub_dir in sub_dir_list:
                        for ss_root, ss_dir_list, _ in os.walk(os.path.join(sub_root, sub_dir)): # level of 'sample_save's
                            for ss_dir in ss_dir_list:
                                if ss_dir == 'simple_save999999':
                                    for sss_root, sss_dir_list, sss_file_list in os.walk(os.path.join(sub_root, sub_dir)):
                                        for file in sss_file_list:
                                            if file_identifier in file:
                                                # found a file, copy it to the best_structure folder
                                                file_to_copy = os.path.join(sss_root, file)
                                                shutil.copy2(file_to_copy, output_dir)


