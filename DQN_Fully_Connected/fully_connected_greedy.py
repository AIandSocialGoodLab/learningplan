import random
import numpy as np
import pandas as pd
import ast
import torch
from torch import nn
from torch import optim
from torch.utils.data import Dataset 
from torch.utils.data import DataLoader 

#Read data
#This model does not read the costs: they are inferred as averages of historical data
#However, in testing, the actual (expected) costs are used.

#Read # of nodes
num_KP = -1
with open('num_nodes.txt') as num_nodes_file:
	num_KP = int(num_nodes_file.readline())

#Read Utilities
KP_utils = []
with open('utils.txt') as utils_file:
	for i in range(num_KP):
		current_util = int(utils_file.readline())
		KP_utils.append(current_util)

#Read DAG as adjacency list
prereq_df = pd.read_csv('prerequisites.csv')
prereq_adj_list = []
for i in range(num_KP):
	prereq_adj_list.append([])
for j in range(len(prereq_df.index)):
	current_row = prereq_df.iloc[j]
	#current_row.source is the prerequisite of current_row.target
	prereq_adj_list[current_row.source].append(current_row.target)

#Read generated data
student_records = pd.read_csv('generated_data.csv')
for j in range(len(student_records.index)):
	#Note: the lists in the csv file are read as
	#strings. ast.literal_eval() converts these strings
	#to int lists.
	student_records.at[j, 'Trace of Learning'] = ast.literal_eval(student_records.at[j, 'Trace of Learning'])
	student_records.at[j, 'Learning Time'] = ast.literal_eval(student_records.at[j, 'Learning Time'])

#Estimate time taken by KPs from generated data
KP_times = []
KP_counts = []
for i in range(num_KP):
	KP_times.append(0.0)
	KP_counts.append(0.0)
for j in range(len(student_records.index)):
	current_row = student_records.iloc[j]
	kps_learned = current_row['Trace of Learning']
	time_taken = current_row['Learning Time']
	for k in range(len(kps_learned)):
		encountered_kp = kps_learned[k]
		encountered_kp_time = time_taken[k]
		new_time_sum = KP_times[encountered_kp] * KP_counts[encountered_kp] + encountered_kp_time
		new_count = KP_counts[encountered_kp] + 1.0
		KP_counts[encountered_kp] = new_count
		KP_times[encountered_kp] = new_time_sum/new_count


#Define the replay buffer class
#Store each state transition
#Here, a state is a binary vector describing the KPs learned so far, and the remaining budget.
class DQNDataset(Dataset):
	#TODO: Determine the correct way to normalize the budget constraint.

	def __init__(self, student_records):
		self.transition_num = 0
		self.current_state = []
		self.next_state = []
		self.new_KP = []

		for j in range(len(student_records.index)):
			current_row = student_records.iloc[j]
			KPs_learned = current_row['Trace of Learning']
			time_taken = current_row['Learning Time']
			budget_constraint = sum(time_taken)

			current_state_vec = []
			for i in range(num_KP):
				current_state_vec.append(0.0)
			current_state_vec.append(budget_constraint)

			for k in range(len(KPs_learned)):
				additional_KP = KPs_learned[k]
				used_time = time_taken[k]

				self.current_state.append(current_state_vec)
				self.new_KP.append(additional_KP)
				self.transition_num += 1

				current_state_vec = current_state_vec.copy()
				current_state_vec[additional_KP] = 1
				current_state_vec[-1] -= used_time  ##Updating budget
				self.next_state.append(current_state_vec)
				current_state_vec = current_state_vec.copy()

		self.current_state = torch.tensor(self.current_state)
		self.next_state = torch.tensor(self.next_state)
		self.new_KP = torch.tensor(self.new_KP)

	def __getitem__(self, index):
		return self.current_state[index], self.next_state[index], self.new_KP[index]

	def __len__(self):
		return self.transition_num

"""
Define and train the Q-network.
Here, the inputs are:
- The binary vector describing the KPs learned so far.
- The budget constraint
The output is:
- A vector of length num_KP that gives the value of learning each new KP.
"""
def create_q_network():
	return nn.Sequential(nn.Linear(num_KP + 1, 50),
						 nn.ReLU(),
						 nn.Linear(50, 50),
						 nn.ReLU(),
						 nn.Linear(50, 50),
						 nn.ReLU(),
						 nn.Linear(50, 50),
						 nn.ReLU(),
						 nn.Linear(50, 50),
						 nn.ReLU(),
						 nn.Linear(50, 50),
						 nn.ReLU(),
						 nn.Linear(50, num_KP),
						 nn.ReLU())
Q_network = create_q_network()
#TODO: See the effect of an additional target network on performance

#Training
transition_list = DQNDataset(student_records)
replay_buffer = DataLoader(dataset=transition_list, batch_size=1, shuffle=True)
optimizer = optim.SGD(Q_network.parameters(), lr=0.01, momentum=0.4)
loss_fn = nn.MSELoss()

for epoch in range(5):
	print("Starting epoch %d" % epoch)
	for current_state, next_state, new_KP in replay_buffer:
		target = 0
		action_values = Q_network(next_state)
		for i in range(num_KP):
			#POTENTIAL BUG: action_values has shape [1, 25], rather than [25]
			target = max(target, action_values[0, i].item())
		target += KP_utils[new_KP]
		#At this point, target is a scalar, rather than a 
		#Pytorch tensor. It is not connected to Q_network. 
		#Therefore, backpropagation will not cause gradients 
		#to go through target.

		#Required in order to use loss_fn
		target = torch.tensor([target], dtype=torch.float32)

		prediction = Q_network(current_state)
		loss = loss_fn(prediction[0, new_KP], target)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

print("Training complete")

####TEST####
test_state = torch.zeros([num_KP + 1], dtype=torch.float32)
test_state[num_KP] = 300
ans = Q_network(test_state)
print(ans)
print(ans.shape)

"""

#Use trained NN to find optimal plan
def find_optimal_plan(budget):
	KP_sequence = []
	total_utility = 0.0
	current_state = torch.zeros([num_KP + 1], dtype=torch.float32)
	current_state[num_KP] = budget

	num_prereqs_left = []
	for i in range(num_KP):
		num_prereqs_left.append(0)
	for i in range(num_KP):
		for j in prereq_adj_list[i]:
			num_prereqs_left[j] += 1
	frontier = set()
	for i in range(num_KP):
		if num_prereqs_left[i] == 0:
			frontier.add(i)

	while budget > 0:
		action_values = Q_network(current_state)
		best_kp = -1
		max_val = 0

		for next_KP in frontier:
			if KP_times[next_KP] > budget:
				continue
			new_val = action_values[i].item()
			if new_val >= max_val:
				max_val = new_val
				best_kp = next_KP

		if best_kp == -1:
			break #No more KPs in the frontier can be used

		#Update plan and utility
		KP_sequence.append(best_kp)
		total_utility += KP_utils[best_kp]
		current_state[best_kp] = 1
		current_state[num_KP] -= KP_times[best_kp]

		#Update indegree counts and frontier
		for j in prereq_adj_list[i]:
			num_prereqs_left[j] -= 1
			if num_prereqs_left[j] == 0:
				frontier.add(j)
		frontier.remove(best_kp)

		#Update budget
		budget -= KP_times[best_kp]

	return total_utility, KP_sequence


#Performance of DQN is first compared to training examples (the random policy)
ratios = []
for j in range(len(student_records.index)):
	current_row = student_records.iloc[j]
	total_time = sum(current_row['Learning Time']) #this is the budget constraint

	dqn_utility, dqn_sequence = find_optimal_plan(total_time)

	#find utility obtained by simulated student
	student_utility = 0
	learning_trace = current_row['Trace of Learning']
	for kp in learning_trace:
		student_utility += KP_utils[kp]

	ratios.append(dqn_utility/student_utility)

print(ratios)

#TODO: Compare performance of DQN to ILP

"""

