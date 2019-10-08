import copy, random
import numpy as np
from simulate_action import transit
NUM_KC = 5
NUM_PLEVELS = 3
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

def gen_proficiency_level():
	p_level = [0]*NUM_KC
	for i in range(NUM_KC):
		p_level[i] = int(random.random()**2*3)
	return p_level

GOAL = [2,2,2,2,2]
ACTIONS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'AT10', 
		   'AT11', 'AT12', 'AT13', 'AT14', 'AT15', 'AT16', 'AT17', 'AT18', 'AT19', 'final']
try:
	P = np.load('P.npy')
	print("successfully load p!")
except:
	print "no transition probability has been made!"
	exit(1)

file_name = str(encode_state(GOAL))+".npy"
try:
	policy = np.load(file_name)
	print("successfully load policy!")
except:
	print "no policy for this state is available"
	exit(1)
def balance(cur_state, s1, s2):
	total_sum = 0
	for i in range(len(cur_state)):
		total_sum += cur_state[i]
	s = copy.deepcopy(s2)
	if total_sum == 0:
		for i in range(NUM_KC):
			if s[i] == -1:
				s[i] = s1[i]
		cur_state[encode_state(s)] = 1.0
	else:
		cur_state = cur_state/total_sum

	return cur_state

def transition(cur_state, cur_action, alpha = 0.00001):
	new_state = np.zeros(NUM_PLEVELS**NUM_KC)
	for i in range(len(cur_state)):
		if cur_state[i]!=0:
			new_state += P[cur_action][i] * cur_state[i]
	total_sum = 0
	valid_states = []
	for i in range(len(new_state)):
		if new_state[i] >= alpha:
			total_sum += new_state[i]
		else:
			new_state[i] = 0
	new_state = new_state/total_sum
	return new_state

def over(s1,s2):
	for i in range(NUM_KC):
		if s1[i] < s2[i]:
			return False
	return True

def reach_goal(cur_state, threhold = 0.85):
	p = 0
	for i in range(len(cur_state)):
		state = decode_state(i)
		if over(state, GOAL):
			#print(state)
			p += cur_state[i]
	return p > threhold

def state_match(s1,s2):
	for i in range(NUM_KC):
		if s2[i]!=-1 and s1[i]!=s2[i]:
			return False 
	return True

def real_reach_goal(s1):
	for i in range(NUM_KC):
		if s1[i] < GOAL[i]:
			return False 
	return True

for maximum_step in range(2,12):
	total_step = 0
	total_reach = 0
	for ite in range(1000):
		if ite % 100 == 0:
			print ite
		START_P = gen_proficiency_level()
		reach_target = False
		cur_state = np.zeros(NUM_PLEVELS**NUM_KC)
		cur_state[encode_state(START_P)] = 1.0
		action_history = []

		step = 0
		real_state = START_P
		while not reach_target and step < maximum_step:
			step += 1
			cur_action = policy[np.argmax(cur_state)]
			#print(cur_action)
			if cur_action !=20:
				real_state, reveal_state = transit(real_state, ACTIONS[cur_action])
			else:
				break
			
			#print(real_state, reveal_state)
			cur_state = transition(cur_state, cur_action)
			argm = np.argmax(cur_state)
			if reveal_state != [-1]*NUM_KC:
				for st in range(len(cur_state)):
					if not state_match(decode_state(st), reveal_state):
						cur_state[st] = 0

			cur_state = balance(cur_state, decode_state(argm), reveal_state)
			#print(cur_state)
			action_history.append((real_state, ACTIONS[cur_action]))
			
			if reach_goal(cur_state):
				reach_target = True


		#print(START_P, GOAL, action_history)
		total_step += step
		#print(real_state)
		if real_reach_goal(real_state):
			#print(real_state)
			total_reach += 1
		else:
			#print(real_state)
			total_reach += (sum(real_state)/float(sum(GOAL)))
	print(total_step/1000.0, total_reach/1000.0)