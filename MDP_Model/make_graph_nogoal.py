import csv, copy, mdptoolbox, json
import numpy as np
history = []
actions = []
all_actions = set()
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
        res += state[i] * 5**(len(state)-1-i)
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

def gen_all_between(lo, hi, related_entry):
    res = [lo]
    for i in range(NUM_KC):
        if i in related_entry and lo[i] < hi[i]:
            new_lo = copy.deepcopy(lo)
            new_lo[i] = hi[i]
            cp = gen_all_between(new_lo, hi, related_entry)
            for k in range(lo[i], hi[i]):
                cur_cp = copy.deepcopy(cp)
                for j in range(len(cur_cp)):
                    cur_cp[j][i] = k + 1
                res += cur_cp
            break
    return res

def state_match(s1,s2):
    for i in range(5):
        if s1[i]!=-1 and s1[i]!=s2[i]:
            return False 
    return True
def update_P(action_index, lo, hi, real_state, related_entry):
    action_count =  ACTION_WEIGHT[action_index]
    w_new = 1.0/(action_count+1)
    w_old = 1 - w_new
    ACTION_WEIGHT[action_index] += 1

    states_between = gen_all_between(lo,hi,[0,1,2,3,4])
    for state in states_between:
        all_between = gen_all_between(state, hi, related_entry)
        p_entries = np.zeros(NUM_KC**5)
        valid_set = []
        total = 0
        for j in range(len(all_between)):
            cur_between_state = all_between[j]
            if state_match(real_state, cur_between_state):
                total += 1
                valid_set.append(j)

        if total!=0:
            for index in valid_set:
                p_entries[encode_state(all_between[index])] = 1.0/total
                
        else:
            p_entries[encode_state(state)] = 1.0
        P[action_index][encode_state(state)] = w_new * p_entries + w_old * P[action_index][encode_state(state)] 

    return

def increase_dis(initial_state, state):
    res = 0
    for i in range(NUM_KC):
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
                actions.append(cur_action)
            cur_history = [str2list(row[-1])]
            length = len(cur_history[0])
            cur_action = [row[2]]

NUM_KC = length 
NUM_ACTIONS = len(all_actions)
all_actions = sorted(list(all_actions))
ACTIONS = all_actions
print(ACTIONS)
#print(sorted(list(set(all_actions))))
print(NUM_KC, NUM_ACTIONS)


haveR = False 
try:
    P = np.load("P.npy")
    print("successfully load P!")
except:
    P = []
    for _ in range(NUM_ACTIONS):
        P.append(np.identity(NUM_KC**5))
    P = np.asarray(P, dtype = np.float64)
    print(P.shape)
    ACTION_WEIGHT = np.zeros(NUM_ACTIONS)


    for i in range(len(history)):
        print(i)
        cur_history = history[i]
        cur_actions = actions[i]
        lo_l, hi_l = process_history(cur_history)
        for j in range(len(cur_history)):
            action = cur_actions[j]
            action_index = all_actions.index(action)
            lo, hi = lo_l[j], hi_l[j]
            if action == "Prior Assessment Test" or action == "Final Exam":
                related_entry = []
            else:
                related_entry = related_entries[action]
            real_state = cur_history[j]
            #print(lo, hi , real_state)
            update_P(action_index, lo, hi, real_state, related_entry)


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
            
"""
        if not mdptoolbox.util.isStochastic(P[i]):
            print ("dif:", (np.abs(P[i].sum(axis=1) -np.ones(P[i].shape[0]))).max() )
            print ("threhold", 10*np.spacing(np.float64(1)))

        

    LEARNING_GOALS = NUM_KC**5
    R = np.ones((len(all_actions), LEARNING_GOALS))
    ql = mdptoolbox.mdp.QLearning(P, R.transpose(), 0.8) 
    ql.run()
"""
    #LEARNING_GOALS = NUM_KC**5


    

#for state_num in range(LEARNING_GOALS):

try:
    R = np.load("R_nogoal.npy")
    haveR = True
    print("successfully load R!")

except:
    print("Genertaing R!")


if True:
    if not haveR:
        R = np.zeros((len(all_actions), NUM_KC**5))
        for action in range(len(all_actions)):
            print(action)
            for state in range(NUM_KC**5):
                cur_r = 0
                for f_state in range(len(P[action][state])):
                    if P[action][state][f_state] != 0:
                        cur_r += (P[action][state][f_state] * 
                        increase_dis(decode_state(state), decode_state(f_state)))
                if action >= 25 and cur_r != 0:
                    cur_r += 0.05  
                R[action][state] = cur_r

        np.save("R_nogoal.npy", R)

    ql = mdptoolbox.mdp.PolicyIteration(P, R.transpose(), 0.96, n_iter = 10000) 
    ql.run()
    print(ql.Q)
    print(ql.policy)
    file_name = 'nogoal.npy'
    np.save(file_name, ql.policy)
    

    

