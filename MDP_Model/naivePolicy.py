from parseDatasheet import encode_state, decode_state, str2list
import numpy as np
import sys

sys.dont_write_bytecode = True

def genNaivePolicy(goal, num_kc, num_plevels, revealKC, actions):
	num_states = num_plevels** num_kc
	def reachGoal(state):
		res = []
		for i in range(num_kc):
			if state[i] < goal[i]:
				res.append(i)
		return res 
	def intersection(l1, l2):
		return list(set(l1) & set(l2))
	policy = []
	for enc_state in range(num_states):
		action_p = np.zeros(len(actions)+1)
		state = decode_state(enc_state)
		unreach_kc = reachGoal(state)
		related_actions = []
		for action in actions:
			if intersection(revealKC[action],unreach_kc) != []:
				related_actions.append(action)
		for action in related_actions:
			action_p[actions.index(action)] += 1.0/len(related_actions)
		policy.append(action_p)
	return np.array(policy) 




