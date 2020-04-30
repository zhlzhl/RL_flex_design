import numpy as np
import os
import pickle
from pathlib import Path
import shutil
import pickle
from spinup.FlexibilityEnv_input.inputload import load_FlexibilityEnv_input
from spinup.FlexibilityEnv.FlexibilityEnv import expected_sales_for_structure

if __name__ == "__main__":
    exp_dir = "10x10a"
    cwd = os.getcwd()

    exp_path = os.path.join(cwd.split('best_structures')[0],'data', exp_dir)

    count = 0
    for root, dirs, files in os.walk(exp_path):
        for dir in dirs:
            if "10x10" in dir and "10x10a" not in dir:
                src = os.path.join(root, dir)
                new_dir = dir.replace('10x10', '10x10a')
                dst = os.path.join(root, new_dir)
                os.rename(src, dst)
                count += 1
                print(count)

        for file in files:
            if "best_eval_performance_n_structure" in file and '10x10' in file and '10x10a' not in file:
                src = os.path.join(root, file)
                new_file = file.replace('10x10', '10x10a')
                dst = os.path.join(root, file)
                os.rename(src, dst)
                count += 1
                print('renamed file {}'.format(new_file))
                print(count)
