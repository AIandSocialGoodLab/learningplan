'''
 Version: 0419
 Utility Maximizing Learning Plan Simulations
    # Brute force, Greedy, ILP Solver
    # Support Additive cost function
'''
import networkx as nx
import numpy as np
import random,copy, time, json, os, argparse, csv, datetime
from gurobipy import *
from UMLP_solver import *
import utils

# INPUT description:
# G <- a DAG object representing n knowledge points' dependencies
# B <- a number describing total budget
# C <- a row vector of length n describing cost of learning Ki
# U <- a row vector of length n describing the value of learning Ki
# type <- type of cost function

def generate_random_dag(nodes, density):
    #Generate a random Directed Acyclic Graph (DAG) with a given number of nodes and edges.
    G = nx.DiGraph()
    edges = density * nodes * (nodes - 1)
    for i in range(nodes):
        G.add_node(i)
    for i in range(nodes**2):
        a = random.randint(0,nodes-1)
        b = a
        while b==a:
            b = random.randint(0,nodes-1)
        
        if G.has_edge(a,b): 
            G.remove_edge(a,b)
        else:
            G.add_edge(a,b)
            current_degree = sum(dict(G.degree()).values())
            if not (nx.is_directed_acyclic_graph(G) and current_degree <= edges):
                G.remove_edge(a,b)
    return G


def get_index (A):
    res = 0
    for kp in A:
        res += 2**kp
    return res 

def generate_cost(G, cost_type = "add"):
    # cost_type has add, monotone
    N = G.order()
    C0 = np.random.uniform(1,10,N)
    if cost_type == "add":
        return C0
    elif cost_type == "monotone":
        C = np.zeros((2**N, N))
        C[0] = C0
        def generate_part_cost(C, A, i):
            maximum_cost = cost(C,[],i, cost_type)
            for i in range(len(A)):
                curA = A[:i]+A[i+1:]
                curCost = cost(C,curA, i, cost_type)
                if curCost <= maximum_cost:
                    maximum_cost = curCost
            return np.random.uniform(maximum_cost*(1-1.0/N), maximum_cost)

        def all_subsets(N, x):

            return itertools.combinations(list(range(N)), x)


        for x in range(N):
            subsets = all_subsets(N,x)
            for subset in subsets:
                for i in range(N):
                    index = get_index(subset)
                    C[index][i] = generate_part_cost(C, subset, i)
    return C

def generate_utility(G):
    N = G.order()
    return np.random.uniform(1,10, N)


def simulate():
    args = utils.process_args(vars(utils.parser.parse_args()))
    print(args)
    Ns, densities, solvers, budgets, nsim, costType, verbose, loadPrev, standardize = args
    result_dict = []
    result_colnums_names = ['N','Density','Solver','Budget','Cost',
                            'Time_avg','Time_sd','Sol_avg','Sol_sd']
    total_simulations = utils.getTotalSimulation([Ns, densities, budgets, costType])
    total_simulations *= nsim
    progress = 0
    loadPrev_outer = loadPrev

    try:
        for N in Ns:
            for density in densities:
                for budget in budgets:
                    for cost in costType:
                        sols = np.zeros((nsim,len(solvers)))
                        times = np.zeros((nsim,len(solvers)))
                        if loadPrev:
                            try:
                                print ("\nLoading previously saved test instances...")
                                try:
                                    update_cost = False
                                    sims,new_budget = utils.load_saved_instance(N,density,budget,cost)
                                except:
                                    print("Need to update costs...")
                                    update_cost = True
                                    sims,new_budget = utils.load_saved_instance(N,density,budget,None)
                            except:
                                print ("Failed to load... Creating new instances...")
                                sims = []
                                loadPrev = False
                        else:
                            print ("Creating new instances...")
                            sims = []

                        for sim in range(nsim):
                            if loadPrev and sim < len(sims):
                                changed_instance = False
                                G,B,U,C = sims[sim]
                                if update_cost:
                                    print("\nUpdating costs...")
                                    C = generate_cost(G, cost)
                                    sims[sim] = G,B,U,C
                                    changed_instance = True
                                if new_budget:
                                    print("\nReusing test cases but with different budget...")
                                    B = 5 * G.order() * budget
                            else:
                                changed_instance = True
                                G = generate_random_dag(N, density)
                                B = 5 * N * budget
                                U = generate_utility(G)
                                C = generate_cost(G, cost)
                                sims.append((G,B,U,C))
                            for solver_index in range(len(solvers)):
                                solver = solvers[solver_index]
                                if solver == "ilp":
                                    if cost == "monotone":
                                        C_ilp = C[0]
                                        s_time, s_sol = ilp_time(G,C[0],B,U)
                                    elif cost == "add":
                                        s_time, s_sol = ilp_time(G,C,B,U)
                                elif solver == "bf":
                                    s_time, s_sol = brute_force_time(G,C,B,U,cost)
                                elif solver == "gd":
                                    s_time, s_sol = greedy_time(G,C,B,U,cost)
                                elif solver == "gd2":
                                    s_time, s_sol = greedy2_time(G,C,B,U,cost)
                                sols[sim,solver_index] = s_sol
                                times[sim,solver_index] = s_time
                            progress += 1
                            if verbose: utils.update_progress(progress/total_simulations)
                        if changed_instance or new_budget:
                            print ("\nTest instances saved for future use.")
                            utils.save_instance(sims,N,density,budget,cost)

                        result_dict.extend(utils.generate_result_dict(N, density, budget, 
                                                                      cost, solvers, sols, times,
                                                                      standardize))
                        loadPrev = loadPrev_outer

        utils.export(result_colnums_names, result_dict)
    except KeyboardInterrupt:
        utils.export(result_colnums_names, result_dict)




if __name__ == '__main__':
    simulate()

