import os

files = os.listdir(os.getcwd())

suffix = '_s88888'

for file in files:
    if os.path.isfile(file) and "best_eval_performance_n_structure" in file:
        file_name = file.split('.')
        file_name[0] += suffix
        new_file = file_name[0] + '.' + file_name[1]

        os.rename(file, new_file)
