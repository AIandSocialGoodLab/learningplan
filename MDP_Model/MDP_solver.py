import numpy as np
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
START_P = [1,1,1,1,1]
GOAL = [4,2,3,1,4]
ACTIONS = ['0', '1', '10', '11', '12', '14', '15', 
'17', '18', '19', '2', '20', '21', '22', '23', '24', 
'25', '26', '27', '28', '29', '3', '30', '31', '32', 
'33', '34', '35', '36', '37', '38', '39', '4', '40', 
'41', '43', '44', '45', '46', '47', '48', '49', '5', 
'6', '8', '9', 'AT51', 'AT52', 'AT53', 'AT54', 'AT55', 
'AT56', 'AT57', 'AT58', 'AT59', 'AT60', 'AT62', 'AT63', 
'AT64', 'AT65', 'AT66', 'AT67', 'AT68', 'AT69', 
'Final Exam', 'Prior Assessment Test']
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

def transite(cur_state, cur_action):
	new_state = np.zeros(5**5)
	for i in range(len(cur_state)):
		if cur_state[i]!=0:
			new_state += P[cur_action][i] * cur_state[i]
	return new_state

def over(s1,s2):
	for i in range(5):
		if s1[i] > s2[i]:
			return False
	return True

def reach_goal(cur_state, threhold = 0.5):
	p = 0
	for i in range(len(cur_state)):
		state = decode_state(i)
		if over(state, GOAL):
			p += cur_state[i]
	print(cur_state, p)
	return p > threhold

reach_target = False
cur_state = np.zeros(5**5)
cur_state[encode_state(START_P)] = 1.0
action_history = []
while not reach_target:
	cur_action = policy[cur_state.argmax()]
	action_history.append(ACTIONS[cur_action])
	cur_state = transite(cur_state, cur_action)
	if reach_goal(cur_state):
		reach_target = True

print(START_P, GOAL, action_history)