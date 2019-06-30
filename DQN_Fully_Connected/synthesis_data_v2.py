#Ziang's Data Generation Code
#No need to generate test data
#The model will be tested by running on the graph and comparing with ILP
import random
import networkx as nx
import pandas as pd

# Copy from Mike"s previous work
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

def save_dag(G):
    dag_df = nx.convert_matrix.to_pandas_edgelist(G)
    dag_df = dag_df.astype('int32')
    dag_df.to_csv('prerequisites.csv', index=False)

def generate_data(N, M, MIN_TIME, MAX_TIME, MIN_UTIL, MAX_UTIL, DIFF_VAR, DENSITY, MIN_LEARNINGSTEP, MAX_LEARNINGSTEP):
    TIME_CONSUMED = [random.randint(MIN_TIME, MAX_TIME) for i in range(M)]
    UTIL_OBTAINED = [random.randint(MIN_UTIL, MAX_UTIL) for i in range(M)]
    DEPENDENCY = generate_random_dag(M, DENSITY)

    #Examples
    empty_dict = {'ID':[], 'Trace of Learning':[], 'Learning Time':[]}
    gen_df = pd.DataFrame(empty_dict)
    for i in range(N):
        avail_kps = [n for n,d in DEPENDENCY.in_degree() if d == 0]
        visited = []
        learning_sequence = []
        learning_time = []
        steps = random.randint(MIN_LEARNINGSTEP, MAX_LEARNINGSTEP)
        for s in range(steps):
            cur_node = avail_kps[random.randint(0, len(avail_kps) - 1)]
            time = random.randint(TIME_CONSUMED[cur_node] - DIFF_VAR, TIME_CONSUMED[cur_node] + DIFF_VAR)
            learning_sequence.append(cur_node)
            learning_time.append(time)
            visited.append(cur_node)
            avail_kps.remove(cur_node)
            for (_,y) in DEPENDENCY.out_edges(cur_node):
                avail = True
                for (z,_) in DEPENDENCY.in_edges(y):
                    if z not in visited:
                        avail = False
                        break
                if avail: 
                    avail_kps.append(y)
        new_row = pd.DataFrame([[i, learning_sequence, learning_time]], columns=['ID', 'Trace of Learning', 'Learning Time'])
        gen_df = gen_df.append(new_row)

    #Save data to files
    save_dag(DEPENDENCY)
    gen_df.to_csv('generated_data.csv', index=False)
    with open('costs.txt', 'w') as cost_file:
        for i in range(M):
            cost_file.write('%d\n' % TIME_CONSUMED[i])
    with open('utils.txt', 'w') as util_file:
        for i in range(M):
            util_file.write('%d\n' % UTIL_OBTAINED[i]) #Utility is fixed
    with open('num_nodes.txt', 'w') as num_nodes_file: #Write number of KPs
        num_nodes_file.write('%d\n' % M)


N = 1000 #number of student
M = 25 #number of knowledge points
MIN_TIME = 5
MAX_TIME = 100
MIN_UTIL = 5
MAX_UTIL = 100
DIFF_VAR = 2
DENSITY = 0.4
MIN_LEARNINGSTEP = 5
MAX_LEARNINGSTEP = 20
generate_data(N, M, MIN_TIME, MAX_TIME, MIN_UTIL, MAX_UTIL, DIFF_VAR, DENSITY, MIN_LEARNINGSTEP, MAX_LEARNINGSTEP)

