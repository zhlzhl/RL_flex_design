import os
import os.path as osp

# Where experiment outputs are saved by default:
DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),'data')

# Whether to automatically insert a date and time stamp into the names of
# save directories:
FORCE_DATESTAMP = False

# Whether to render FlexibilityEnv Viewer during evaluation
FORCE_RENDER = False

# Force Not to save model every --save_freq but only perform evaluation if --do_checkpoint_eval is set as True
# the change is made in logx.py save_state function
# if hasattr(self, 'tf_saver_elements'):
#     if FORCE_NO_MODEL_SAVE:
#         pass
#     else:
#         self._tf_simple_save(itr)
FORCE_NO_MODEL_SAVE = False

# Whether GridSearch provides automatically-generated default shorthands:
DEFAULT_SHORTHAND = True

# Tells the GridSearch how many seconds to pause for before launching 
# experiments.
WAIT_BEFORE_LAUNCH = 2
