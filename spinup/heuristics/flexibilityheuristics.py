#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 13:24:27 2020

@author: yehuawei
"""

import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import time

from gurobipy import * 


def add_usedualmargin(initial_design, capacity_vec, demand_vec, profit_mat):
#Set up the basic structure of the second stage problem
    
    X = initial_design.copy()
    num_capacity, num_samples = capacity_vec.shape
    num_demand = len(demand_vec)
    
    plants = range(num_capacity)
    products = range(num_demand)
        
    m = Model('MaxProfit')
    f = {}; cs ={}; ct={}; #f is the flow variable, b is the lost sales variable, cs is the (supply) plant constraints, ct is the (demand) product constraints
    ub={}
    for j in range(num_demand):
        for i in range(num_capacity):
            f[i,j] = m.addVar(name='f_%s' %i + '%s' %j, ub=X[i,j]*10e30)
            f[i,j].setAttr(GRB.attr.Obj, profit_mat[i,j])
    m.update()
    for i in range(num_capacity):
        cs[i] = m.addConstr(quicksum(f[i,j] for j in products) <= 0, name='cs_%s' %i)
    for j in range(num_demand):
        ct[j] = m.addConstr(quicksum(f[i,j] for i in plants) <= 0, name='ct_%s' %j)
    m.update()
    m.setAttr("ModelSense", GRB.MAXIMIZE)
    #m.params.method = 0
    m.setParam('OutputFlag', 0)
    
    A = np.zeros((num_capacity, num_demand))

    samp_profit = np.zeros(num_samples)
    margins = np.zeros((num_capacity, num_demand))
    
    for s in range(num_samples):    
        for j in range(num_demand):
            ct[j].setAttr(GRB.attr.RHS, demand_vec[j,s])
        for i in range(num_capacity):
            cs[i].setAttr(GRB.attr.RHS, capacity_vec[i,s])
        m.optimize()
        samp_profit[s]=m.objVal
        
        for i in range(num_capacity):
            for j in range(num_demand):
                margins[i,j] += max(0, profit_mat[i,j] - cs[i].Pi - ct[j].Pi)
    
    arc_index = {}
    max_margin = -1    
    for i in range(num_capacity):
        for j in range(num_demand):
            if margins[i,j] > max_margin:
                arc_index[0] = i
                arc_index[1] = j
                max_margin = margins[i,j]

    #print(dual_cap_margins)        
    return arc_index, np.average(samp_profit)

def subtract(initial_design, capacity_vec, demand_vec, profit_mat):
#Set up the basic structure of the second stage problem
    
    X = initial_design.copy()
    num_capacity, num_samples = capacity_vec.shape
    num_demand = len(demand_vec)
    
    plants = range(num_capacity)
    products = range(num_demand)
        
    m = Model('MaxProfit')
    f = {}; cs ={}; ct={}; #f is the flow variable, b is the lost sales variable, cs is the (supply) plant constraints, ct is the (demand) product constraints
    ub={}
    for j in range(num_demand):
        for i in range(num_capacity):
            f[i,j] = m.addVar(name='f_%s' %i + '%s' %j, ub=X[i,j]*10e30)
            f[i,j].setAttr(GRB.attr.Obj, profit_mat[i,j])
    m.update()
    for i in range(num_capacity):
        cs[i] = m.addConstr(quicksum(f[i,j] for j in products) <= 0, name='cs_%s' %i)
    for j in range(num_demand):
        ct[j] = m.addConstr(quicksum(f[i,j] for i in plants) <= 0, name='ct_%s' %j)
    m.update()
    m.setAttr("ModelSense", GRB.MAXIMIZE)
    m.setParam('OutputFlag', 0)
    
    A = np.zeros((num_capacity, num_demand))

    samp_profit = np.zeros(num_samples)
    for s in range(num_samples):    
        for j in range(num_demand):
            ct[j].setAttr(GRB.attr.RHS, demand_vec[j,s])
        for i in range(num_capacity):
            cs[i].setAttr(GRB.attr.RHS, capacity_vec[i,s])
        m.optimize()
        samp_profit[s]=m.objVal
        
        for i in range(num_capacity):
            for j in range(num_demand):
                A[i,j]+=f[i,j].x
    
    nonzero_arcs = np.nonzero(X)
    num_nonzeros = len(nonzero_arcs[0])
    arc_index = {}
    min_flow = np.sum(demand_vec)*np.max(profit_mat)
    
    for i in range(num_nonzeros):
        if min_flow > A[nonzero_arcs[0][i], nonzero_arcs[1][i]]*profit_mat[nonzero_arcs[0][i], nonzero_arcs[1][i]]:
            arc_index[0] =nonzero_arcs[0][i]
            arc_index[1] =nonzero_arcs[1][i]
            min_flow = A[nonzero_arcs[0][i], nonzero_arcs[1][i]]*profit_mat[nonzero_arcs[0][i], nonzero_arcs[1][i]]
        
    return arc_index, np.average(samp_profit)


def greedy(initial_design, capacity_vec, demand_vec, profit_mat, original_profit=0):
#Set up the basic structure of the second stage problem
    
    X = initial_design.copy()
    num_capacity, num_samples = capacity_vec.shape
    num_demand = len(demand_vec)
        
    m = Model('MaxProfit')
    f = {}; cs ={}; ct={}; #f is the flow variable, b is the lost sales variable, cs is the (supply) plant constraints, ct is the (demand) product constraints
    ub={}
    for j in range(num_demand):
        for i in range(num_capacity):
            f[i,j] = m.addVar(name='f_%s' %i + '%s' %j, ub=X[i,j]*10e30)
            f[i,j].setAttr(GRB.attr.Obj, profit_mat[i,j])
    m.update()
    
    for i in range(num_capacity):
        cs[i] = m.addConstr(quicksum(f[i,j] for j in products) <= 0, name='cs_%s' %i)
    for j in range(num_demand):
        ct[j] = m.addConstr(quicksum(f[i,j] for i in plants) <= 0, name='ct_%s' %j)
    m.update()
    m.setAttr("ModelSense", GRB.MAXIMIZE)
    m.setParam('OutputFlag', 0)
    
    
    incremental_profit = -1
    
    it_arcs = np.nditer(X, flags=['multi_index'])
    while not it_arcs.finished:
        new_profit = original_profit
        samp_profit = np.zeros(num_samples)
        if X[it_arcs.multi_index[0],it_arcs.multi_index[1]]==0:
            f[it_arcs.multi_index[0],it_arcs.multi_index[1]].setAttr(GRB.attr.UB, 10e30)                          
            for s in range(num_samples):    
                for j in range(num_demand):
                    ct[j].setAttr(GRB.attr.RHS, demand_vec[j,s])
                for i in range(num_capacity):
                    cs[i].setAttr(GRB.attr.RHS, capacity_vec[i,s])
                m.optimize()
                samp_profit[s]=m.objVal
            new_profit = np.average(samp_profit)
            f[it_arcs.multi_index[0],it_arcs.multi_index[1]].setAttr(GRB.attr.UB, 0)             

        if(new_profit-original_profit>incremental_profit):
            incremental_profit = new_profit-original_profit
            arc_index = it_arcs.multi_index
            print("new profit:", original_profit+incremental_profit)
        
        it_arcs.iternext()
        
    
    return arc_index, original_profit+incremental_profit


def _evaluate_single_structure(design, capacity_vec, demand_vec, profit_mat):
    X = design.copy()
    num_capacity, num_samples = capacity_vec.shape
    num_demand = len(demand_vec)

    plants = range(num_capacity)
    products = range(num_demand)

    m = Model('MaxProfit')
    f = {};
    cs = {};
    ct = {};  # f is the flow variable, b is the lost sales variable, cs is the (supply) plant constraints, ct is the (demand) product constraints
    ub = {}
    for j in range(num_demand):
        for i in range(num_capacity):
            f[i, j] = m.addVar(name='f_%s' % i + '%s' % j, ub=X[i, j] * 10e30)
            f[i, j].setAttr(GRB.attr.Obj, profit_mat[i, j])
    m.update()
    for i in range(num_capacity):
        cs[i] = m.addConstr(quicksum(f[i, j] for j in products) <= 0, name='cs_%s' % i)
    for j in range(num_demand):
        ct[j] = m.addConstr(quicksum(f[i, j] for i in plants) <= 0, name='ct_%s' % j)
    m.update()
    m.setAttr("ModelSense", GRB.MAXIMIZE)
    m.setParam('OutputFlag', 0)

    A = np.zeros((num_capacity, num_demand))

    samp_profit = np.zeros(num_samples)
    for s in range(num_samples):
        for j in range(num_demand):
            ct[j].setAttr(GRB.attr.RHS, demand_vec[j, s])
        for i in range(num_capacity):
            cs[i].setAttr(GRB.attr.RHS, capacity_vec[i, s])
        m.optimize()
        samp_profit[s] = m.objVal

    return np.average(samp_profit)


def evaluate(design, capacity_vec, demand_vec, profit_mat):
#Set up the basic structure of the second stage problem
    if isinstance(design, list):
        perf_list = []
        for structure in design:
            perf_list.append(_evaluate_single_structure(structure, capacity_vec, demand_vec, profit_mat))

        print("performance list: {}".format(perf_list))
        perf = max(perf_list)

    else:
        perf = _evaluate_single_structure(design, capacity_vec, demand_vec, profit_mat)

    return np.average(perf)