import json, ast
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

#From simulator.py
NUM_KC = 5
NUM_PROFICIENCY_LEVEL = 5

#Hardcoded settings from MDP_solver.py
ACTIONS = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20',
 '21', '22', '23', '24', '3', '4', '5', '6', '7', '8', '9', 'AT25', 'AT26', 'AT27', 'AT28', 'AT29',
  'AT30', 'AT31', 'AT32', 'AT33', 'AT34', 'AT35', 'AT36', 'AT37', 'AT38', 'AT39', 'Final Exam', 'Prior Assessment Test']

#Functions from MDP_solver.py
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

def state_match(s1,s2):
    for i in range(5):
        if s2[i]!=-1 and s1[i]!=s2[i]:
            return False 
    return True

action_index = {}
for i in range(len(ACTIONS)):
	action_index[ACTIONS[i]] = i

#This tells us the proficiencies revealed
#by each action.
with open("action.txt") as action_json:
	actions_full_dict = json.load(action_json)
	reveal_dict = actions_full_dict["reveal_plevels"]
	reveal_dict["Final Exam"] = [0, 1, 2, 3, 4]

#This tells us which trajectory to start from for testing.
with open("first_test_trajectory.txt") as test_file:
	first_trajectory = int(test_file.read())

#Discard the trajectories which come before the first one used for testing.
historical_data = pd.read_csv("datasheet.csv")
num_episodes = 0
start_index = -1
for i in range(historical_data.shape[0]):
	if num_episodes == first_trajectory and historical_data.iloc[i]['Action_Types'] == 'Prior Assessment Test':
		start_index = i
		break
	if historical_data.iloc[i]['Action_Types'] == 'Prior Assessment Test':
		num_episodes += 1

historical_test_data = historical_data.iloc[start_index:]

#Find average log-likelihood
#Only count transitions where some KCs were actually observed.
transition_matrix = np.load("P.npy")
sum_log_likelihoods = 0
num_transitions = 0
belief_state = np.zeros(NUM_PROFICIENCY_LEVEL ** NUM_KC)

for i in range(historical_test_data.shape[0]):
	current_action = historical_test_data.iloc[i]['Action_Types']
	if current_action == 'Prior Assessment Test':
		#Reset belief state at the start of episode
		new_initial_prof = ast.literal_eval(historical_test_data.iloc[i]['Cur_Proficiency'])
		state_index = encode_state(new_initial_prof)
		belief_state.fill(0)
		belief_state[state_index] = 1
	else:
		action_transition_matrix = transition_matrix[action_index[current_action]]
		#Note: The rows of action_transition_matrix represent the starting
		#state, and the columns represent the ending state!
		belief_state = np.matmul(belief_state, action_transition_matrix)

		#If assessment test, then update our log-likelihood average.
		#Also remove impossible states from belief state.
		if current_action[:2] == 'AT':
			revealed_proficiencies = ast.literal_eval(historical_test_data.iloc[i]['Cur_Proficiency'])
			#Convert from dictionary to list
			possible_states = [-1] * NUM_KC
			for kc in range(NUM_KC):
				if kc in revealed_proficiencies:
					possible_states[kc] = revealed_proficiencies[kc]

			likelihood = 0
			for state_index in range(NUM_PROFICIENCY_LEVEL ** NUM_KC):
				decoded_state = decode_state(state_index)
				if state_match(decoded_state, possible_states):
					likelihood += belief_state[state_index]
				else:
					belief_state[state_index] = 0 #This state is impossible
			sum_log_likelihoods += np.log(likelihood)
			num_transitions += 1

			#Normalize the belief state
			sum_beliefs = 0
			for state_index in range(NUM_PROFICIENCY_LEVEL ** NUM_KC):
				sum_beliefs += belief_state[state_index]
			for state_index in range(NUM_PROFICIENCY_LEVEL ** NUM_KC):
				belief_state[state_index] /= sum_beliefs


print(sum_log_likelihoods/num_transitions)