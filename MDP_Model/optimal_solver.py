import csv, copy, mdptoolbox, json
import numpy as np 



def gen_all_valid_states(s):
	res = []
	not_exact = False
	for i in range(len(s)):
		if s[i] < 0:
			not_exact = True
			for j in range(NUM_PLEVELS):
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
		res += state[i] * NUM_PLEVELS**(len(state)-1-i)
	return res

def decode_state(num):
	res = [0] * NUM_KC
	for i in range(NUM_KC):
		res[-1-i] = num % NUM_PLEVELS
		num = num/NUM_PLEVELS
	return res


def increase_dis(initial_state, state, target_state):
	res = 0
	for i in range(NUM_KC):
		if initial_state[i] > target_state[i]:
			initial_state[i] = target_state[i]
		if state[i] > target_state[i]:
			state[i] = target_state[i]
		res += state[i] - initial_state[i]
	return res

all_actions = ACTIONS =  ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'AT10', 
		   'AT11', 'AT12', 'AT13', 'AT14', 'AT15', 'AT16', 'AT17', 'AT18', 'AT19', 'final']
P = []
for _ in range(len(ACTIONS)):
	P.append(np.identity(NUM_PLEVELS**NUM_KC))
P = np.asarray(P, dtype = np.float64)
print(P.shape)

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

state_num = encode_state([2,2,2,2,2])


def reach_goal(s, goal):
	for i in range(NUM_KC):
		if s[i] < goal[i]:
			return False
	return True



try:
	R = np.load("R" + str(state_num) + "optimal.npy")
	haveR = True
	print("successfully load R!")

except:
	print("Genertaing R!")
	R = -np.ones((NUM_PLEVELS**NUM_KC, len(all_actions)))
	print(R.shape)
	for i in range(NUM_PLEVELS**NUM_KC):
		if reach_goal(decode_state(i), decode_state(state_num)):
			R[i, -1] = 0
	np.save("R" + str(state_num) + "optimal.npy", R)
ql = mdptoolbox.mdp.PolicyIteration(P, R, 0.999,  max_iter = 1000) 
ql.run()
print(ql.iter)
print(ql.policy)
file_name = str(state_num)+'optimal.npy'
np.save(file_name, ql.policy)
