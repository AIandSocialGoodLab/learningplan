import numpy as np
from simulate_action import transit
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
START_P = [2,0,1,3,4]

ACTIONS = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '3', '4', '5', '6', '7', '8', '9', 'AT25', 'AT26', 'AT27', 'AT28', 'AT29', 'AT30', 'AT31', 'AT32', 'AT33', 'AT34', 'AT35', 'AT36', 'AT37', 'AT38', 'AT39', 'Final Exam', 'Prior Assessment Test']
try:
	P = np.load('P.npy')
	print("successfully load p!")
except:
	print "no transition probability has been made!"
	exit(1)

file_name ="nogoal.npy"
try:
	policy = np.load(file_name)
	print("successfully load policy!")
except:
	print "no policy for this state is available"
	exit(1)
def balance(cur_state):
	total_sum = 0
	for i in range(len(cur_state)):
		total_sum += cur_state[i]
	cur_state = cur_state/total_sum
	return cur_state

def transition(cur_state, cur_action, alpha = 0.00001):
	new_state = np.zeros(5**5)
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
	for i in range(5):
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
	print(cur_state[cur_state.argmax()], decode_state(cur_state.argmax()))
	print(p)
	return p > threhold

def state_match(s1,s2):
    for i in range(5):
        if s2[i]!=-1 and s1[i]!=s2[i]:
            return False 
    return True


cur_state = np.zeros(5**5)
cur_state[encode_state(START_P)] = 1.0
action_history = []
step = 0
real_state = START_P
while step < 10:
	step += 1
	cur_action = policy[cur_state.argmax()]
	real_state, reveal_state = transit(real_state, ACTIONS[cur_action])
	print("realstate v.s. reveal_state", real_state, reveal_state)
	print("calculated_state", cur_state.argmax())
	if reveal_state != [-1]*5:
		for st in range(len(cur_state)):
			if not state_match(decode_state(st), reveal_state):
				cur_state[st] = 0
	cur_state = balance(cur_state)
	#print(cur_action)
	action_history.append(ACTIONS[cur_action])
	cur_state = transition(cur_state, cur_action)

print(START_P, action_history)