#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 21:20:01 2020

@author: yehuawei
"""

import numpy as np
import os
import pickle


def load_FlexibilityEnv_input(file):

    with open(file, 'rb') as f:
        m,n = pickle.load(f)
        mean_c= pickle.load(f)
        mean_d = pickle.load(f)
        sd_d=pickle.load(f)
        profit_mat=pickle.load(f)
        target_arcs=pickle.load(f)
        fixed_costs=pickle.load(f)
        flex_0=pickle.load(f)

    return m, n, mean_c, mean_d, sd_d, profit_mat, target_arcs, fixed_costs, flex_0


if __name__ == "__main__":
    input_file = "input_ran10x10a_cv0.8.pkl"
    file = os.path.join(os.getcwd(), input_file)

    m, n, mean_c, mean_d, sd_d, profit_mat, target_arcs, fixed_costs, flex_0 = load_FlexibilityEnv_input(file)

    print((m, n))
    print(mean_c)
    print(mean_d)
    print(sd_d)
    print(profit_mat)
    print('target_arcs: {}'.format(target_arcs))
    print(fixed_costs)
    print(flex_0)