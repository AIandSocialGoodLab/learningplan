import csv, copy, mdptoolbox, json
import numpy as np 

NUM_KC = 5

P = []
for _ in range(42):
    P.append(np.identity(5**5))
P = np.asarray(P, dtype = np.float64)
print(P.shape)


def gen_all_valid_states(s):
	res = []
	not_exact = False
	for i in range(len(s)):
		if s[i] < 0:
			not_exact = True
			for j in range(5):
				cur_s = copy.deepcopy(s)
				cur_s[i] = j
				res += gen_all_valid_states(cur_s)
				
			break
	if not_exact:
		return res
	else:
		return [s]

def form_end_state(s1, s2):
	for i in range(len(s2)):
		if s2[i] == -1:
			s2[i] = s1[i]
	return s2

def encode_state(state):
    res = 0
    for i in range(len(state)):
        res += state[i] * 5**(len(state)-1-i)
    return res

def decode_state(num):
    res = [0] * 5
    for i in range(5):
        res[-1-i] = num % 5
        num = num/5
    return res


def increase_dis(initial_state, state, target_state):
    res = 0
    for i in range(5):
        if initial_state[i] > target_state[i]:
            initial_state[i] = target_state[i]
        if state[i] > target_state[i]:
            state[i] = target_state[i]
        res += state[i] - initial_state[i]
    return res

all_actions = ACTIONS = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '3', '4', '5', '6', '7', '8', '9', 'AT25', 'AT26', 'AT27', 'AT28', 'AT29', 'AT30', 'AT31', 'AT32', 'AT33', 'AT34', 'AT35', 'AT36', 'AT37', 'AT38', 'AT39', 'Final Exam', 'Prior Assessment Test']
with open("./action.txt") as json_file:
    action_dict = json.load(json_file)
    #print(action_dict["actions"])
    for i in range(len(ACTIONS)):
    	key = ACTIONS[i]
    	if key in action_dict["actions"]:
    		for raw_state in action_dict["actions"][key]:
    			state = [int(j) for j in raw_state.strip('[]').split(',')]
    			for exact_state in gen_all_valid_states(state):
    				for result_states in action_dict["actions"][key][raw_state]:
    					end_state = form_end_state(exact_state, result_states['end_state'])
    					p = result_states['p']
    					s1 = encode_state(exact_state)
    					s2 = encode_state(end_state)
    					P[i,s1,s1] = 0
    					P[i,s1,s2] = p



for i in range(len(P)):
	for j in range(len(P[i])):
		dif = 1.0 - np.sum(P[i,j])
		if dif != 0:
			for k in range(len(P[i,j])):
				if P[i,j,k]!=0:
					P[i,j,k] = P[i,j,k] + dif 
					break


np.save('optimal_P.npy', P)

state_num = encode_state([4,2,3,1,4])
#state_num = encode_state([3,4,2,3,4])
#state_num = encode_state([2,2,4,3,3])
#state_num = encode_state([4,4,2,3,4])
#state_num = encode_state([2,2,4,4,3])
#state_num = encode_state([3,4,2,1,4])
#state_num = encode_state([4,3,2,3,3])
#state_num = encode_state([4,4,4,3,4])
#state_num = encode_state([2,3,4,1,3])
#state_num = encode_state([3,4,2,2,4])
#state_num = encode_state([2,4,4,4,3])
"""
haveR = False
try:
    R = np.load("R" + str(state_num) + ".npy")
    haveR = True
    print("successfully load R!")

except:
    print("Genertaing R!")

if True:
    if not haveR:
        R = np.zeros((len(all_actions), 5**5))
        for action in range(len(all_actions)):
            print(action)
            for state in range(5**5):
                cur_r = 0
                for f_state in range(len(P[action][state])):
                    if P[action][state][f_state] != 0:
                        cur_r += (P[action][state][f_state] * 
                        increase_dis(decode_state(state), decode_state(f_state), decode_state(state_num)))
                if action >= 25 and cur_r != 0:
                    cur_r += 0.05  
                R[action][state] = cur_r

        np.save("R" + str(state_num) + ".npy", R)
"""
def reach_goal(s, goal):
    for i in range(NUM_KC):
        if s[i] < goal[i]:
            return False
    return True
if True:
    R = -np.ones((5**5, len(all_actions)))
    """
    for action in range(len(all_actions)):
        print action , sum(R[action])
    """
    for i in range(5**5):
        if reach_goal(decode_state(i), decode_state(state_num)):
            R[i, -1] = 0
    ql = mdptoolbox.mdp.QLearning(P, R, 1.1, n_iter = 1000000) 
    ql.run()
    print(ql.Q)
    print(ql.policy)
    file_name = str(state_num)+'optimal.npy'
    np.save(file_name, ql.policy)
    
