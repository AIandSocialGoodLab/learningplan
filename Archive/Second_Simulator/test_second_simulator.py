import ast, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
Purpose of this file:
- Evaluate the transition matrix by computing the likelihood of each transition
(i.e. the probability of any observation given the history) in the historical data,
according to this transition matrix. The evaluation metric is the average of the logs
of these likelihoods.

The algorithm for evaluating the transition matrix (computing the likelihoods) is as
follows:
- For each trajectory, what we do is start with a belief state according to the specified 
initial proficiency. Now, at each time step, we multiply the belief state by the
transition matrix to obtain a new belief state. We use this new belief state to calculate
the probability of the observed proficiency levels. We finally update this belief state
by setting the probabilities of impossible proficiency level combinations to 0, and then
normalizing. We move to the next iteration.
"""

settings = open("settings.txt", 'r')
NUM_KC = int(settings.readline())
NUM_PLEVELS = int(settings.readline())
settings.close()

#Functions from parseDatasheet.py
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

def state_match(s1,s2):
    for i in range(NUM_KC):
        if s1[i]!=-1 and s1[i]!=s2[i]:
            return False 
    return True

#This tells us the proficiencies revealed by each action.
action_related = pd.read_csv("action.csv")
reveal_dict = {}
for i in range(action_related.shape[0]):
	action_name = action_related.iloc[i]["action"]
	reveal_dict[action_name] = action_related.iloc[i]["related_kc"]

#Get list of actions
#Use same order as in parseDatasheet.csv
actions_sorted = []
for action in reveal_dict:
	actions_sorted.append(action)
actions_sorted = sorted(actions_sorted)
print(actions_sorted)

action_index = {}
for i in range(len(actions_sorted)):
	action_index[actions_sorted[i]] = i

#Use test_train_split.txt to determine which trajectory to start from.
#Find index of the (train_num)^th trajectory, and discard the trajectories 
#which come before.
test_train_split = open("test_train_split.txt", "r")
train_percent = int(test_train_split.readline())
validation_percent = int(test_train_split.readline())
test_train_split.close()

#Find index of the (train_num)^th trajectory, and discard the trajectories which come before.
historical_data = pd.read_csv("MDPdatasheet.csv")
trajectory_count = int(historical_data.iloc[-1]["Student_ID"]) + 1
train_num = math.floor(trajectory_count * (train_percent + validation_percent)/100)
start_index = -1
for i in range(historical_data.shape[0]):
	if historical_data.iloc[i]['Student_ID'] == train_num:
		start_index = i
		break

historical_test_data = historical_data.iloc[start_index:]

#Find average log-likelihood of all the transitions.
#Use belief state to summarize all of the history until a time step.
#Only count transitions where some KCs were actually observed.
transition_matrix = np.load("P.npy")
sum_log_likelihoods = 0
num_transitions = 0
belief_state = np.zeros(NUM_PLEVELS ** NUM_KC + 1)

for i in range(historical_test_data.shape[0]):
	current_action = historical_test_data.iloc[i]['Action_Types']
	if current_action == 'Prior Assessment Test':
		#Reset belief state at the start of episode
		new_initial_prof = ast.literal_eval(historical_test_data.iloc[i]['Cur_Proficiency'])
		state_index = encode_state(new_initial_prof)
		belief_state.fill(0)
		belief_state[state_index] = 1

		#Debug
		print("================================================")
	elif current_action == 'Final Exam':
		#Belief state is not updated --- the assumption of the model
		#is that final exam simply moves student to terminal state.
		observation = ast.literal_eval(historical_test_data.iloc[i]['Cur_Proficiency'])
		encoded_final = encode_state(observation)
		final_likelihood = belief_state[encoded_final]
		sum_log_likelihoods += np.log(final_likelihood)
		num_transitions += 1

		#Debug
		print("Final likelihood: " + str(np.log(final_likelihood)))
		print("================================================")
	else:
		action_transition_matrix = transition_matrix[action_index[current_action]]

		#Note: The rows of action_transition_matrix represent the starting
		#state, and the columns represent the ending state!
		belief_state = np.matmul(belief_state, action_transition_matrix)

		#If assessment test, then update our log-likelihood average.
		#Also update belief state by assigning 0 probability to impossible
		#states, and normalizing the rest.
		if current_action[:2] == 'AT':
			revealed_proficiencies = ast.literal_eval(historical_test_data.iloc[i]['Cur_Proficiency'])

			#Convert from dictionary to list
			possible_states = [-1] * NUM_KC
			for kc in range(NUM_KC):
				if kc in revealed_proficiencies:
					possible_states[kc] = revealed_proficiencies[kc]

			likelihood = 0
			for state_index in range(NUM_PLEVELS ** NUM_KC):
				decoded_state = decode_state(state_index)
				if state_match(decoded_state, possible_states):
					likelihood += belief_state[state_index]
				else:
					#This state is not possible after new observation
					belief_state[state_index] = 0

			print(np.log(likelihood))
			sum_log_likelihoods += np.log(likelihood)
			num_transitions += 1

			#Normalize the belief state
			sum_beliefs = 0
			for state_index in range(NUM_PLEVELS ** NUM_KC):
				sum_beliefs += belief_state[state_index]
			for state_index in range(NUM_PLEVELS ** NUM_KC):
				belief_state[state_index] /= sum_beliefs

print("Overall Average Log-likelihood: " + str(sum_log_likelihoods/num_transitions))