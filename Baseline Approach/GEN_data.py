
import csv
import random
import networkx as nx

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
"""
Feature space should include Student id, ethnicity, gender, age, etc.
Data should include set of knowledge points students already know; Trace of learning: e.g. (D, 3mins), (E, 5mins) .etc. 
"""
N = 100 #number of student
M = 25 #number of knowledge points
MIN_TIME = 5
MAX_TIME = 100
TIME_CONSUMED = [random.randint(MIN_TIME,MAX_TIME) for i in range(M)]
DIFF_VAR = 2
DENSITY = 0.4
DEPENDENCY = generate_random_dag(M, DENSITY)


"""
ETHNICITY = ["Asian", "Arab", "Australian Indigenous", "Black or African American", "Hispanic or Latino", 
"Maori", "Native American or Alaska Native", "Native Hawaiian or Other Pacific Islander", "White or Cucasian"]
"""
ETHNICITY = ["Asian", "Arab", "Black or African American", "Hispanic or Latino", "White or Cucasian"]
GENDER = ["Male", "Female"]
AGEMIN = 18
AGEMAX = 25

MIN_LEARNINGSTEP = 5
MAX_LEARNINGSTEP = 20

with open("generated_data.csv", mode="w") as generated_data:
    dwriter = csv.writer(generated_data, delimiter=",")
    dwriter.writerow(["ID", "ethnicity", "gender", "age", "Trace of learning", "Learning Time"])
    for i in range(N):
        ethnicity = ETHNICITY[random.randint(0,len(ETHNICITY) - 1)]
        gender = GENDER[random.randint(0,len(GENDER) - 1)]
        age =random.randint(AGEMIN, AGEMAX)
        avail_kps = [n for n,d in DEPENDENCY.in_degree() if d == 0]
        visited = []
        learning_sequence = []
        learning_time = []
        steps = random.randint(MIN_LEARNINGSTEP, MAX_LEARNINGSTEP)
        for s in range(steps):
            cur_node = avail_kps[random.randint(0, len(avail_kps) - 1)]
            time = random.randint(TIME_CONSUMED[cur_node]-DIFF_VAR, TIME_CONSUMED[cur_node]+DIFF_VAR)
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
        dwriter.writerow([str(i), ethnicity, gender, str(age), learning_sequence, learning_time])
