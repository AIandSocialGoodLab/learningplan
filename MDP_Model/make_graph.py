import csv, copy, mdptoolbox, json
import numpy as np

history = []
actions = []

all_actions = set()
NUM_PLEVELS = 3
with open("./action.txt") as json_file:
    action_dict = json.load(json_file)
    related_entries = action_dict["related_entries"]


def str2list(s):
    return [int(i) for i in s.strip('[]').split(', ')]

def dict2list(s, length):
    res = [-1]*length
    if s!="":
        for entry in s.strip('{}').split(', '):
            res[int(entry[0])] = int(entry[-1])
    return res

def encode_state(state):
    res = 0
    for i in range(len(state)):
        res += state[i] * NUM_PLEVELS**(len(state)-1-i)
    return res
def decode_state(num):
    res = [0] * NUM_KC
    for i in range(NUM_KC):
        res[-1-i] = num % 5
        num = num/5
    return res

def process_history(cur_history):
    lo_l = copy.deepcopy(cur_history)
    hi_l = copy.deepcopy(cur_history)
    cur_state = copy.deepcopy(cur_history[0])
    for state in lo_l:
        for i in range(len(state)):
            if state[i] == -1:
                state[i] = cur_state[i]
            else:
                tmp = state[i]
                state[i] = cur_state[i]
                cur_state[i] = tmp

    cur_state = copy.deepcopy(cur_history[-1])
    for state in hi_l[::-1]:
        for i in range(len(state)):
            if state[i] == -1:
                state[i] = cur_state[i]
            else:
                cur_state[i] = state[i]
    return lo_l, hi_l


def gen_all_between(lo, hi):
    res = [lo]
    for i in range(5):
        if lo[i] < hi[i]:
            new_lo = copy.deepcopy(lo)
            new_lo[i] = hi[i]
            cp = gen_all_between(new_lo, hi)
            for k in range(lo[i], hi[i]):
                cur_cp = copy.deepcopy(cp)
                for j in range(len(cur_cp)):
                    cur_cp[j][i] = k + 1
                res += cur_cp
            break
    return res



def state_match(s1,s2):
    for i in range(NUM_KC):
        if s1[i]!=-1 and s1[i]!=s2[i]:
            return False 
    return True


def gen_all_paths(trace):
    lo = trace[0]
    hi = [-1]*NUM_KC
    between = trace[1:]
    all_paths = []
    if len(between) <= 1:
        return [trace]
    else:
        for state in between:
            count = 0
            for i in range(NUM_KC):
                if hi[i]==-1 and state[i]!=-1:
                    hi[i] = state[i]
                    count += 1
                    if count == NUM_KC:
                        break
        all_choices = gen_all_between(lo,hi)
        for state in all_choices:
            if state_match(between[0], state):
                cur_trace = copy.deepcopy(between)
                cur_trace[0] = state
                cur_all_paths = gen_all_paths(cur_trace)
                all_paths += [[lo]+path for path in cur_all_paths]

        return all_paths

def state_match(s1,s2):
    for i in range(NUM_KC):
        if s1[i]!=-1 and s1[i]!=s2[i]:
            return False 
    return True

def find_all_possibilities(all_paths, actions):
    state_copy = copy.deepcopy(all_paths)
    for trace in state_copy:
        for i in  range(len(trace)):
            trace[i] = encode_state(trace[i])
    res = dict()

    for i in range(len(actions)):
        res[actions[i]] = dict()
        for trace in state_copy:
            if trace[i] not in res[actions[i]]:
                res[actions[i]][trace[i]] = [0]*(NUM_PLEVELS**NUM_KC)
            res[actions[i]][trace[i]][trace[i+1]] += (1.0/len(state_copy))
    for action in actions:
        for trace in res[action]:
            res[action][trace] = np.asarray(res[action][trace])
            res[action][trace] *= 1.0/(sum(res[action][trace]))
    return res

def update_P(action_index, state, state_vec):
    action_count =  ACTION_WEIGHT[action_index]
    w_new = 1.0/(action_count+1)
    w_old = 1 - w_new
    ACTION_WEIGHT[action_index] += 1

    P[action_index][state] = w_new * state_vec + w_old * P[action_index][state] 

    return

def increase_dis(initial_state, state, target_state):
    res = 0
    for i in range(NUM_KC):
        if initial_state[i] > target_state[i]:
            initial_state[i] = target_state[i]
        if state[i] > target_state[i]:
            state[i] = target_state[i]
        res += state[i] - initial_state[i]
    return res

def reach_goal(s, goal):
    for i in range(NUM_KC):
        if s[i] < goal[i]:
            return False
    return True



with open('MDPdatasheet.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    first_line = True
    cur_history = []
    cur_action = []
    cur_student = -1
    length = -1
    for row in csv_reader:
        if first_line:
            first_line = False
            continue
        all_actions.add(row[2])
        if int(row[1]) == cur_student:
            cur_action.append(row[2])
            if (row[2] == "Final Exam"):
                cur_history.append(str2list(row[-1]))
            else:
                cur_history.append(dict2list(row[-1], length))
        else:
            cur_student = int(row[1])
            if cur_student!=0:
                final_state = cur_history[-1]
                cur_history = cur_history[:-1]
                cur_history[-1] = final_state
                history.append(cur_history)
                actions.append(cur_action[1:-1])
            cur_history = [str2list(row[-1])]
            length = len(cur_history[0])
            cur_action = [row[2]]

NUM_KC = length 

all_actions = sorted(list(all_actions))
all_actions.remove("Final Exam")
all_actions.remove("Prior Assessment Test")
NUM_ACTIONS = len(all_actions)
ACTIONS = all_actions
print(ACTIONS)
print(NUM_KC, NUM_ACTIONS)



state_num = encode_state([2,1,2,2,1])


try:
    P = np.load("P.npy")
    print("successfully load P!")
except:
    print ("start finding P")
    P = []
    for _ in range(NUM_ACTIONS + 1):
        P.append(np.identity(NUM_PLEVELS**NUM_KC))
    P = np.asarray(P, dtype = np.float64)
    print(P.shape)
    ACTION_WEIGHT = np.zeros(NUM_ACTIONS)


    for i in range(len(history)):
        print(i)
        cur_history = history[i]
        all_paths = gen_all_paths(cur_history)
        cur_actions = actions[i]
        possibilities = find_all_possibilities(all_paths, cur_actions)
        for action in possibilities:
            action_index = all_actions.index(action)
            for state in possibilities[action]:
                state_vec = possibilities[action][state]
                update_P(action_index,state,state_vec)
        


    print(P)
    for i in range(len(P)):
        for j in range(len(P[i])):
            dif = 1.0 - np.sum(P[i,j])
            if dif != 0:
                for k in range(len(P[i,j])):
                    if P[i,j,k]!=0:
                        P[i,j,k] = P[i,j,k] + dif
                        break
    np.save('P.npy', P)
            


try:
    R = np.load("R" + str(state_num) + ".npy")
    print("successfully load R!")

except:
    print("Genertaing R!")
    R = -np.ones((NUM_PLEVELS**NUM_KC, len(all_actions) + 1))
    for i in range(NUM_PLEVELS**NUM_KC):
        if reach_goal(decode_state(i), decode_state(state_num)):
            R[i, -1] = 0
        else:
            R[i, -1] = -float('Inf')

    np.save("R" + str(state_num) + ".npy", R)
print(P.shape, R.shape)
print(R)

ql = mdptoolbox.mdp.PolicyIteration(P, R, 0.999999, max_iter = 5000) 
ql.run()
print(ql.iter)
print(ql.policy)
file_name = str(state_num)+'.npy'
np.save(file_name, ql.policy)
        

    



    

