'''
 Version: 0419
 Utility Maximizing Learning Plan Solver
  	# Brute force, Greedy, ILP Solver
    # Support Additive cost function
'''
import networkx as nx
import numpy as np
import random,copy, time, json, collections, operator
from gurobipy import *
import utils

# G <- a DAG object representing n knowledge points' dependencies
# B <- a number describing total budget
# C <- a row vector of length n describing cost of learning Ki
# U <- a row vector of length n describing the value of learning Ki
# type <- type of cost function

################################################################
######################## ILP ###################################
################################################################
def ilp_setup(G, B, C, U):
	N = G.order() #number of nodes
	Ns = np.arange(0, N)
	m = Model("milp") #create model
	m.setParam( 'OutputFlag', False )
	 
	X = m.addVars(N, vtype = GRB.BINARY) #add N binary variables
	
	for kp_i in G.nodes():
		# for each node in G
		for inedge in G.in_edges(kp_i):
			# for all of the nodes that have incoming edge to it
			kp_j = inedge[0]
			# add dependency contraint
			m.addConstr(X[kp_j] - X[kp_i] >= 0,
						name = "x%i,x%i"%(kp_i, kp_j))


	CoeffC = dict(zip(X, C)) 
	# add budget constraint
	m.addLConstr(X.prod(CoeffC), GRB.LESS_EQUAL, B) 
	

	CoeffV = dict(zip(X, U)) 
	# set objective function
	m.setObjective(X.prod(CoeffV), GRB.MAXIMIZE) 

	return m

def ilp_time(G,C,B,U):
	m = ilp_setup(G, B, C, U)
	start = time.time()
	m.optimize()
	# for v in m.getVars():
	# 	print (v.varName, v.x)
	end = time.time()
	sol = m.objVal
	return end - start, sol

################################################################
####################### Helper Fns. ############################
################################################################
def get_index (A):
    res = 0
    for kp in A:
        res += 2**kp
    return res 

def cost(C, A, i, cost_type = "add"):
    #inputs:
    # C: cost array
    # A: set of acquired kps
    # i: index of the kp that we want to know the cost of
    # output: cost of k_i
    if cost_type == "add":
        return C[i]
    elif cost_type == "monotone":
        return C[get_index(A)][i]

def get_actions(S, G, C, cost_type = "add"):
    #inputs:
    # S: current state
    # G: kps graph object
    # C: cost array
    # cost_type: cost function cost_type
    #output: all avaliable actions in state S
    A, B = S
    n = len(A)
    KPs = np.arange(G.order()).reshape((-1,1))
    
    def prereq_cleared(k_i):
        in_edges = np.array(list(G.in_edges(k_i[0])))
        if len(in_edges) == 0:
            return True
        else:
            prereqs = in_edges[:,0]
            prereqs_cleared = np.isin(prereqs, np.array(A))
        return np.all(prereqs_cleared)
    
    def query_cost(k_i):
        return cost(C, A, k_i, cost_type)
    
    costs = (np.apply_along_axis(query_cost, 1, KPs)).T[0]
    under_budget = (np.apply_along_axis(query_cost, 1, KPs)).T[0] <= B
    pre_req_cleared = np.apply_along_axis(prereq_cleared, 1, KPs)
    not_learned = np.logical_not(np.isin(KPs, np.array(A)).T[0])
    actions = KPs[np.logical_and(np.logical_and(under_budget, pre_req_cleared),
                              not_learned)].reshape(1,-1)[0]
    costs = costs[actions]
    
    return actions, costs

def take_action(action, cost, S):
    #inputs:
    # action: whihc kp to learn
    # cost: the corresponding cost
    # S: current state
    #output: new state S'
    (A,B) = S
    A_new = copy.copy(A)
    A_new.append(action)
    B_new = B - cost
    S_new = (A_new, B_new)
    return S_new


def compute_utility(S, U):
    (A,B) = S
    return sum(U[A])

################################################################
####################### Brute Force ############################
################################################################
def brute_force(G, C, B, U, cost_type = "add"):
    #brute force graph traversal with BFS
    A = []
    B = B
    Q = collections.deque([(A,B)])
    global_max = ((A,B), 0)
    while len(Q) != 0:
        S = Q.popleft()
        actions, costs = get_actions(S, G, C, cost_type) 
        action_indicies = list(range(len(actions)))
        take_action_i = lambda i: take_action(actions[i], costs[i], S)
        newstates = list(map(take_action_i, action_indicies))
        newstates_temp = copy.deepcopy(newstates)
        if len(newstates) > 0:
            for newstate in newstates:
                if newstate in Q:
                    newstates_temp.remove(newstate)
                else:
                    Q.append(newstate)
        
            newstates_temp_indicies = list(range(len(newstates_temp)))
            compute_utility_i = lambda i: compute_utility(newstates_temp[i], U)
            utilities = list(map(compute_utility_i, newstates_temp_indicies))
            current_max_index = np.argmax(utilities)
            current_max_seq = newstates_temp[current_max_index]
            current_max_u = utilities[current_max_index]
            if current_max_u > global_max[1]:
                global_max = (current_max_seq, current_max_u)
    return global_max

def brute_force_time(G, C, B, U, cost_type = "add"):
	start = time.time()
	result = brute_force(G, C, B, U, cost_type)
	end = time.time()
	sol = result[1]
	return end - start, sol

################################################################
########################### Greedy #############################
################################################################
def greedy(G, C, B, U, cost_type = "add"):
	#greedy search
	A = []
	B = B
	global_max = ((A,B), 0)
	nodes = list(np.arange(G.order()))
	depth_order = utils.longest_path(G)
	depth_utility_order = list(zip(nodes, depth_order, -U))
	greedy_order = sorted(depth_utility_order, key = operator.itemgetter(1, 2))
	# print(greedy_order)
	seq = []
	utility = 0
	# print(C)
	while B > 0 and len(greedy_order) > 0:
		node, depth, u_node = greedy_order.pop(0)
		if cost_type == "add":
			c_node = cost(C, seq, node, cost_type)
		elif cost_type == "monotone":
			c_node = cost(C, seq, node, cost_type)
		if B - c_node < 0: break
		else:
			seq.append(node)
			utility -= u_node
			B -= c_node
	return (seq, utility, B)


def greedy_time(G, C, B, U, cost_type = "add"):
	start = time.time()
	result = greedy(G, C, B, U, cost_type)
	end = time.time()
	sol = result[1]
	return end - start, sol

################################################################
################### Monotone Greedy ############################
################################################################
def greedy2(G,C,B,U, cost_type = "monotone"):
    frontier = {n for n,d in G.out_degree() if d==0}
    visited = []
    unvisited = list(G.nodes())
    utility = 0
    while len(unvisited)!=0:
        minimum = 10
        cur_node = 0

        for node in frontier:
            if cost_type == "add":
                c = cost(C,unvisited,node,cost_type)
            elif cost_type == "monotone":
                c = cost(C,unvisited,node,cost_type)
            u = U[node]
            if u - c < minimum:
                minimum = u-c
                cur_node = node 
        visited.append(cur_node)
        #print(frontier, cur_node)
        frontier.remove(cur_node)
        
        unvisited.remove(cur_node)
        for (x,y) in list(G.in_edges(cur_node)):
            all_visited = True
            for (_,n) in list(G.out_edges(x)):
                if n not in visited:
                    all_visited = False
                    break
            if x not in visited and all_visited:
                frontier.add(x)
        
    seq = visited[::-1]
    index = 0
    c = 0
    while index < len(seq):
        if cost_type == "add":
            c = cost(C,seq[:index],seq[index],cost_type)
        elif cost_type == "monotone":
            c = cost(C,seq[:index],seq[index],cost_type)
        if B < c: break
        B -= c
        utility += U[seq[index]] 
        index += 1
    return (seq[:index], utility, B)

def greedy2_time(G, C, B, U, cost_type = "monotone"):
	start = time.time()
	result = greedy2(G, C, B, U, cost_type)
	end = time.time()
	sol = result[1]
	return end - start, sol
