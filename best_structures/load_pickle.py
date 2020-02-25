import os
import pickle
import numpy as np

file = "RSvsDrop.pickle"

with open(file, 'rb') as f:
    rl_perf, drop_perf = pickle.load(f)

for perf in rl_perf.values():
    print(perf)

rl_perfs = np.zeros((len(rl_perf.keys()), 2))
drop_perfs = np.zeros((len(rl_perf.keys()), 2))
t = 0

for target_arc, perf in rl_perf.items():
    rl_perfs[t, 0] = int(target_arc)
    drop_perfs[t, 0] = int(target_arc)
    rl_perfs[t, 1] = perf
    drop_perfs[t, 1] = drop_perf[target_arc]
    t += 1
