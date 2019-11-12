import pandas as pd, copy, numpy as np, mdptoolbox, math

settings = open("settings.txt", 'r')
NUM_KC = int(settings.readline())
NUM_PLEVELS = int(settings.readline())
settings.close()

def str2list(s):
	return [int(i) for i in s.strip('[]').split(', ')]

def dict2list(s):
	res = [-1]*NUM_KC
	if s!="{}":
		for entry in s.strip('{}').split(', '):
			res[int(entry[0])] = int(entry[-1])
	return res

def gen_all_between(lo, hi, relate_kc):
	if relate_kc == []:
		return [lo]
	else:
		cur_kc = relate_kc[0]
		res = []
		results = gen_all_between(lo, hi, relate_kc[1:])
		for num in range(lo[cur_kc], hi[cur_kc]+1):
			cop = copy.deepcopy(results)
			for inst in cop:
				inst[cur_kc] = num 
				res.append(inst)
		return res 


def state_match(s1,s2):
	for i in range(NUM_KC):
		if s1[i]!=-1 and s1[i]!=s2[i]:
			return False 
	return True

def reach_goal(s1, s2):
  for i in range(NUM_KC):
  	if s1[i] < s2[i]:
  		return False
  return True

def action_match(lo, state, action):
	for i in range(NUM_KC):
		if state[i] > lo[i] and i not in ACTIONS[action]:
			return False 
	return True

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

def gen_all_paths(trace, actions, action_related_kcs):
	lo = trace[0]
	hi = [-1]*NUM_KC
	between = trace[1:]
	all_paths = []
	action = actions[0]
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
		all_choices = gen_all_between(lo, hi, action_related_kcs[action])
		#print(lo, hi, action, all_choices)
		for state in all_choices:
			if state_match(between[0], state):
				cur_trace = copy.deepcopy(between)
				cur_trace[0] = state
				cur_all_paths = gen_all_paths(cur_trace, actions[1:], action_related_kcs)
				all_paths += [[lo]+path for path in cur_all_paths]

		return all_paths


def generate_transition_matrix(history, actions, all_actions, action_related_kcs):
	num_actions = len(all_actions)
	P = np.zeros((num_actions + 2, NUM_PLEVELS**NUM_KC + 1, NUM_PLEVELS**NUM_KC + 1))
	mat = np.zeros(( NUM_PLEVELS**NUM_KC + 1, NUM_PLEVELS**NUM_KC + 1))
	mat[-1] = np.ones(NUM_PLEVELS**NUM_KC + 1)
	P[-1] = mat.T 
	P[-2] = mat.T
	count = np.zeros(num_actions)
	for i in range(len(history)):
		print(i)
		cur_history = history[i]
		cur_actions = actions[i]
		paths = gen_all_paths(cur_history, cur_actions, action_related_kcs)
		cur_weights = dict()
		for action in cur_actions:
			cur_weights[action] = np.zeros((NUM_PLEVELS**NUM_KC + 1, NUM_PLEVELS**NUM_KC + 1))
		single_w = 1.0/len(paths)
		
		for path in paths:
			encode_path = []
			for state in path:
				encode_path.append(encode_state(state))
			for j in range(len(cur_actions)):
				cur_action = cur_actions[j]
				(s1,s2) = (encode_path[j], encode_path[j+1])
				cur_weights[cur_action][s1,s2] += single_w

		for action in cur_weights:
			ind = all_actions.index(action)
			P[ind] = (count[ind] * P[ind] + cur_weights[action])/(count[ind] + np.sum(cur_weights[action]))
			count[ind] += np.sum(cur_weights[action]) 
	for i in range(len(P)):
		P[i,-1,-1] = 1
	return P

def main():

	try: 
		P = np.load("P.npy")
		actionset = pd.read_csv("action.csv")
		ACTIONS = dict()
		for row in range(actionset.shape[0]):
			ACTIONS[actionset["action"][row]] = str2list(actionset["related_kc"][row])
		ACTION_LIST = []
		for action in ACTIONS:
			ACTION_LIST.append(action)

		ACTION_LIST = sorted(ACTION_LIST)
		print(ACTION_LIST)
		print("Successfully Load P!")

	except:
		datasheet = pd.read_csv('MDPdatasheet.csv')
		actionset = pd.read_csv("action.csv")
		ACTIONS = dict()
		for row in range(actionset.shape[0]):
			ACTIONS[actionset["action"][row]] = str2list(actionset["related_kc"][row])

		#Truncate datasheet to only include the trajectories
		#marked as training and validation
		test_train_split = open("test_train_split.txt", "r")
		train_percent = int(test_train_split.readline())
		validation_percent = int(test_train_split.readline())
		test_train_split.close()
		trajectory_count = int(datasheet.iloc[-1]["Student_ID"]) + 1
		train_num = math.floor(trajectory_count * (train_percent + validation_percent)/100)

		history = []
		actions = []
		trajectories_completed = 0
		for row in range(datasheet.shape[0]):
			if trajectories_completed >= train_num:
				break
			cur_act = datasheet["Action_Types"][row]
			try:
				cur_hist = str2list(datasheet["Cur_Proficiency"][row])
			except:
				cur_hist = dict2list(datasheet["Cur_Proficiency"][row])
			if cur_act == "Prior Assessment Test":
				cur_history = [cur_hist]
				cur_actions = []
			elif cur_act == "Final Exam":
				cur_history[-1] = cur_hist
				history.append(cur_history)
				actions.append(cur_actions)
				trajectories_completed += 1
			else:
				cur_actions.append(cur_act)
				cur_history.append(cur_hist)


		ACTION_LIST = []
		for action in ACTIONS:
			ACTION_LIST.append(action)

		ACTION_LIST = sorted(ACTION_LIST)
		print(ACTION_LIST)

		P = generate_transition_matrix(history, actions, ACTION_LIST, ACTIONS)
		
		for i in range(len(P)):
			for j in range(len(P[i])):
				dif = 1.0 - np.sum(P[i,j]) 
				if dif > 0.1:
					P[i,j,j] = 1.0
				dif = 1.0 - np.sum(P[i,j]) 
				if dif != 0:
					for k in range(len(P[i,j])):
						if P[i,j,k]!=0:
							P[i,j,k] = P[i,j,k] + dif
							break


		np.save('P.npy', P)
		print("Generate and Save P!")
	print(P)

if __name__ == "__main__":
	main()
