import os
import os.path as osp


EXPERIMENT = '10x10'

# a unique string to identify the set of directories to look into
DIR_IDENTIFIER = ('ENV', EXPERIMENT)
EXCLUDE = ()
# a unique string to identify the name of the files to be copied
FILE_IDENTIFIER = 'best_eval_performance_n_structure'
# the name of the directory where the best structures are saved
SAVED_MODEL_DIR_NAME = ('simple_save999999')