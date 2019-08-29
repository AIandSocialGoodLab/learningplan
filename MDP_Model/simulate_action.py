import getopt, sys, math, random, json, csv, copy

ACTION_PATH =  "./action.txt"#last action is "assessment test"

with open(ACTION_PATH) as json_file:
	all_actions = json.load(json_file)
	action_set = []
	for action in all_actions["actions"]:
		action_set.append(action)
		for state in all_actions["actions"][action]:
			continue
	num_actions = len(action_set)
	related_entries = all_actions["reveal_plevels"]



def state_match(state, statei):
	s2 = [int(i) for i in statei.strip('[]').split(',')]
	for i in range(5):
		if state[i]!= s2[i] and s2[i]!=-1:
			return False 
	return True

def transit(state, action):
	stateindex = ""
	#print(all_actions["actions"][action])
	for statei in all_actions["actions"][action]:
		#print(all_actions["actions"][action])
		#print("compare!", state, statei)
		if state_match(state, statei):
			stateindex = statei
			break
	if stateindex != "":
		final_states = copy.deepcopy(all_actions["actions"][action][stateindex])
		p0 = random.random()
		final_state = []
		for fstate in final_states:
			p0 -= fstate['p']
			if p0 <= 0:
				final_state = fstate['end_state']
		for i in range(5):
			if final_state[i] == -1:
				final_state[i] = state[i]
	else:
		print("Prerequisit Not MET")
		final_state = state 

	if  len(all_actions["reveal_plevels"][action]) == 2:
		related_entries = []
	else:
		related_entries = copy.deepcopy(all_actions["reveal_plevels"][action])
	reveal_states = [-1]*5
	for e in related_entries:
		reveal_states[e] = final_state[e]
	return final_state, reveal_states
	
#print(state_match([1,0,2,1,1], "[-1, -1, 2, -1, -1]"))

