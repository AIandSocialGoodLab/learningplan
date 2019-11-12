'''
 Version: 0419
 Various Util Functions
'''
import networkx as nx
import numpy as np
import random,copy, time, json, os, argparse, collections,pickle, datetime, csv
from functools import reduce
from gurobipy import *
from UMLP_solver import *


# INPUT description:
# G <- a DAG object representing n knowledge points' dependencies
# B <- a number describing total budget
# C <- a row vector of length n describing cost of learning Ki
# U <- a row vector of length n describing the value of learning Ki
# type <- type of cost function
small_n = 10e-5

parser = argparse.ArgumentParser(description = 'UMLP simulator')
parser.add_argument('--n', default = '[30,30,10]',type=str, help='Specify the range of nodes in [start,end,step] format.')
parser.add_argument('--density', default = '[0.2,0.2,0.1]', type=str, 
    help='Specify the range of edge density in [start,end,step] format.')
parser.add_argument('--nsim', default = 30, type = int, help = 'Specify number of simulations.')
parser.add_argument('--verbose',default = False, type=bool, help="Print progress?")
parser.add_argument('--solver',default = '[bf]', type=str, 
    help="Specify solver types in [x,y,...] form (bf: Brute Force; gd: Greedy; ilp: Integer Linear Program)")
parser.add_argument('--maxlearnP',default = '[0.166,0.166,0.1]', type=str, 
    help="Specify the range of maximum fraction of knowledge points that user can learn in [start, end,step] format.")
parser.add_argument('--costType',default = '[add]', type=str, 
    help="Specify cost type in [x,y,...] form (add: additive; monotone: monotone; sub: submodular)")
parser.add_argument('--loadPrev',default = True, type=bool, help="Load previously created test instances?")
parser.add_argument('--standardize',default = False, type=bool, help="Standardize solution. Only valid if one of the solver is greedy")

def process_args(p):
	def splitAndStrip(s):
		l = s.split(",")
		l = list(map(lambda x: x.replace("[","").replace("]",""),l))
		return l

	nsim = p['nsim']
	verbose = p['verbose']
	costType = p['costType']
	loadPrev = p['loadPrev']
	standardize = p['standardize']
	p.pop('nsim')
	p.pop('verbose')
	p.pop('loadPrev')
	p.pop('standardize')

	
	arg_vals = list(map(splitAndStrip, list(p.values())))
	Ns, densities, solvers, budgets, costType = arg_vals
	try:
		assert(len(Ns) == 3 and len(densities) == 3 and len(budgets)==3)
	except:
		raise AssertionError('Input form of Ns, densities, solvers, or budgets invalid. Please check you arguments (Hint: put them in list form).')

	try:
	    Ns = np.arange(int(Ns[0]),int(Ns[1])+int(Ns[2]),step=int(Ns[2]))
	    densities = np.arange(float(densities[0]),float(densities[1])+small_n,
	                          step=float(densities[2]))
	    budgets = np.arange(float(budgets[0]),float(budgets[1])+small_n,
	                        step=float(budgets[2]))
	except:
		raise Exception("Number conversion error! Please check your arguments.")

	for s in solvers:
		if s not in ['bf','gd','ilp','gd2']:
			raise AssertionError('Unrecognized solver type!')

	if standardize and ("gd" not in solvers or "gd2" not in solvers) and len(solvers) == 1:
		raise AssertionError('To get the standardized solution quality comparison, you at least have to have two solvers and one of them must be greedy')

	for t in costType:
		if t not in ['add','monotone','sub']:
			raise AssertionError('Unrecognized cost function type!')


	return [Ns, densities, solvers, budgets, nsim, costType, verbose, loadPrev, standardize]

def generate_result_dict(N, density, budget, cost, solvers, sols, times, standardize):
	if standardize:
		greedy_index = solvers.index("gd")
		if "bf" in solvers: optimal_index = solvers.index("bf")
		else: optimal_index = solvers.index("ilp")
		sols = sols/sols[:,optimal_index][:,None]
	sols_means = sols.mean(axis=0)
	sols_sds = sols.std(axis=0)
	times_means = times.mean(axis=0)
	times_sds = times.std(axis=0)
	result = []
	for solver_idx in range(len(solvers)):
		d = {"N":N, "Density":density, 'Solver':solvers[solver_idx],"Budget":budget, "Cost":cost,
			'Time_avg':times_means[solver_idx],'Time_sd':times_sds[solver_idx],'Sol_avg':sols_means[solver_idx],
			'Sol_sd':sols_sds[solver_idx]}
		result.append(d)
	return result

def bfs_helper(G,nodes_depth,seen,queue):
	depth = 0 
	while queue:
		vertex,d = queue.popleft()
		if d > depth: depth += 1
		for node in list(G.neighbors(vertex)):
			if node not in seen:
				seen.add(node)
				queue.append((node,depth+1))
				nodes_depth[node] = depth+1
			elif node in seen and nodes_depth[node] < depth+1:
				nodes_depth[node] = depth+1
				queue.append((node,depth+1))
	return nodes_depth, seen, queue

def bfs_depth(G):
	nodes_in_degree = list(G.in_degree())
	roots_with_degree = list(filter(lambda x: x[1] == 0, nodes_in_degree))
	roots = list(map(lambda x: x[0],roots_with_degree))
	nodes_depth = list(np.arange(G.order()))
	seen, queue = set(), collections.deque()
	for root in roots:
		nodes_depth[root] = 0
		queue.append((root, 0))
		nodes_depth, seen, queue = bfs_helper(G,nodes_depth,seen,queue)
	return nodes_depth

def max_dist(G, v, d):
	v_in_edges = G.in_edges(v)
	v_in_nodes = list(map(lambda x: x[0],v_in_edges))
	dist_v_in_nodes = d[v_in_nodes]
	if len(v_in_nodes) == 0:
		return 0
	else:
		return max(dist_v_in_nodes) + 1

def longest_path(G):
	N = G.order()
	dist = np.zeros(N)
	topo_sorted = list(nx.topological_sort(G))
	for i in range(N):
		node = topo_sorted[i]
		dist[node] = max_dist(G, node, dist)
	return list(dist)


def update_progress(progress):
    barLength = 30 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    progress_print = round(progress*100,3)
    text = "\rPercent: [{0}] {1}% {2}".format( "="*block + " "*(barLength-block), progress_print, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def getTotalSimulation(Ls):
	def map0to1(i): return 1 if i == 0 else i
	def mult(x1, x2): return x1 * x2
	lens = list(map(lambda l: len(l),Ls))
	lens_nozero = list(map(lambda i : map0to1(i), lens))
	simulations = reduce(mult, lens_nozero)
	return simulations

def load_saved_instance(N,density,budget,cost):
	all_filenames = os.listdir("simulation/probelm_instance")
	new_budget = False
	if cost == None:
		pickle_in = open('simulation/probelm_instance/%s_%s_%s.pickle' % (N,round(density,5),round(budget,5)), 'rb')
	else:
		# print('%s_%s_' % (N, round(density,5)))
		all_ok = list(filter(lambda x: '%s_%s_' % (N, round(density,5)) in x, all_filenames))
		all_ok = list(filter(lambda x: x.startswith('%s_%s_' % (N, round(density,5))), all_filenames))
		all_ok = list(filter(lambda x: '_%s' % (cost) in x, all_ok))
		# print(all_ok)
		if len(all_ok) != 0:
			new_budget = True
			pickle_in = open('simulation/probelm_instance/'+all_ok[0], 'rb')
		else:
			pickle_in = open('simulation/probelm_instance/%s_%s_%s_%s.pickle' % (N,round(density,5),round(budget,5),cost), 'rb')
	sims_data = pickle.load(pickle_in)
	return sims_data,new_budget

def save_instance(sims,N,density,budget,cost):
	pickle_out = open('simulation/probelm_instance/%s_%s_%s_%s.pickle' % (N,round(density,5),round(budget,5),cost), 'wb')
	pickle.dump(sims, pickle_out)


def export(column_names, data):
	now = datetime.datetime.now()
	csv_file = "simulation/simulation_" + now.strftime("%Y-%m-%d-%H-%M") + ".csv"
	try:
		with open(csv_file, 'w') as csvfile:
			writer = csv.DictWriter(csvfile, fieldnames=column_names)
			writer.writeheader()
			for d in data:
				writer.writerow(d)
	except IOError:
		print("I/O error") 