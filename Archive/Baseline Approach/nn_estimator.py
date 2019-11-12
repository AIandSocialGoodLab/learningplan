import torch
from torch import nn
from torch import optim
import numpy as np

"""
TODO: Assign values to each of the KPs
"""
KP_values = []

"""
TODO: 
- Load data from CSV files into below arrays, according to the data format designed by Ziang.
- Load randomly generated graph into CSV file, and then into below prerequisite list
"""
KP_NUM = 25 #No. of knowledge points
N = 100 #No. of students in dataset
genders = []
ethnicities = []
ages = []
learning_traces = []
learning_times = []

prerequisite_lists = [] #For each KP, a list of prerequisites

"""
Neural network for estimating values
Inputs:
- Knowledge points learned (represented as a binary vector - dimension is # of KPs)
- Time taken to learn knowledge points (would it make sense to take the reciprocal? - dimension is # of KPs)
- Ethnicity (One-hot encoding with 5)
- Gender (One-hot encoding with 2)
- Time left (represented as an integer)
In all, there are KP_NUM + 1 + KP_NUM + 5 + 2 = 2 * KP_NUM + 8 input nodes.

The architecture of this fully-connected network is very tentative. Currently, there are
two hidden layers. Note that a ReLU is applied to the output layer to ensure that the result
is nonnegative.
"""
Neural_Net_Estimator = nn.Sequential(nn.Linear(2 * KP_NUM + 8, 60), 
					    			 nn.ReLU(), 
					    			 nn.Linear(60, 30), 
					    			 nn.ReLU(), 
					    			 nn.Linear(30, 10)
					    			 nn.ReLU(),
					    			 nn.Linear(10, 1),
					    			 nn.ReLU())


"""
Helper function returning one-hot encoding for s
"""
def get_gender_vector(s):
	if s == "Male":
		return [1, 0]
	elif s == "Female":
		return [0, 1]
	else:
		raise ValueError

def get_ethnicity_vector(s):
	if s == "Asian":
		return [1, 0, 0, 0, 0]
	elif s == "Arab":
		return [0, 1, 0, 0, 0]
	elif s == "Black or African American":
		return [0, 0, 1, 0, 0]
	elif s == "Hispanic or Latino":
		return [0, 0, 0, 1, 0]
	elif s == "White or Caucasian":
		return [0, 0, 0, 0, 1]
	else:
		raise ValueError


"""
Constructing training data for neural network, as follows:
- For each person, truncate learning traces, and time taken to acquire KPs,
at every possible prefix of the learning trace array. That will be the state
used to construct the input vector. The estimated value (the target) will be the sum of the
values of the KP that the student learns afterwards, and the estimate for time left
will also be the total time the student spends afterwards
"""
input_vectors = []
values = []
for student in range(N):
	for i in range(1 + len(learning_traces[student])):
		KPs_acquired = learning_traces[student][0:i]
		KP_times = learning_times[student][0:i]

		#Output
		achievable_value = 0
		for j in range(len(learning_traces[student])):
			achievable_value += KP_values[learning_traces[student][j]]
		values.append(achievable_value)

		#Input
		time_left = np.sum(learning_times[student][i:])

		KPs_learned = [] #Binary vector representing KPs learned
		KP_times = []
		for j in range(KP_NUM):
			KPs_learned.append(0)
			KP_times.append(-1)  #Time taken to learn KPs is -1 if KP not learned, and actual time otherwise.
		for j in range(i, len(learning_traces[student])):
			KPs_learned[learning_traces[student][j]] = 1
			KP_times[learning_traces[student][j]] = learning_times[student][j]

		gender_vec = get_gender_vector(genders[student])
		ethnicity_vec = get_ethnicity_vector(ethnicities[student])

		example_input = []
		example_input.extend(KPs_learned)
		example_input.extend(KP_times)
		example_input.extend(gender_vec)
		example_input.extend(ethnicity_vec)
		example_input.append(time_left)

		input_vectors.append(example_input)

input_vectors = torch.Tensor(input_vectors)
values = torch.Tensor(values)

"""
Train network on above training data using SGD + momentum, with an MSE loss function
- Hyperparameters still tentative
"""
dataset = torch.utils.data.TensorDataset(input_vectors, values)
loader = torch.utils.data.DataLoader(dataset, batch_size=25, shuffle=True)
optimizer = optim.SGD(Neural_Net_Estimator.parameters(), lr=0.1, momentum=0.3)
criterion = nn.MSELoss()

for epoch in range(25):
	for batch_input, batch_values in loader:
		optimizer.zero_grad()
		loss = criterion(batch_input, batch_values)
		loss.backward()
		optimizer.step()


#Estimate time taken to learn KPs from historical data
sum_times = []
kp_counts = []
time_estimates = []
for i in range(KP_NUM):
	sum_times.append(0)
	kp_counts.append(0)
	time_estimates.append(0)

for i in range(N):
	for j in range(len(learning_traces[i])):
		sum_times[learning_traces[i][j]] += learning_times[i][j]
		kp_counts[learning_traces[i][j]] += 1

for i in range(KP_NUM):
	if (kp_counts[i] > 0):
		time_estimates[i] = sum_times[i] / kp_counts[i]

"""
pick_action:
- kp_vector is a binary vector representing the KPs that have been learned
- learning_times is an ordinary array which contains the time taken to learn each KP (and -1 for KPs not learned)
- Ethnicity is one of 5 strings
- Gender is one of 2 strings

If no new KP can be learned in the remaining time, -1 is returned. Otherwise, the KP which results
in the highest value (as estimate by the neural network) is returned.
"""
def pick_action(kp_vector, learning_times, time_left, ethnicity, gender):

	####Construct new input vector, which will be modified for every possible
	####new state
	input_vec = []
	input_vec.extend(kp_vector)
	input_vec.extend(learning_times)
	input_vec.extend(get_gender_vector(gender))
	input_vec.extend(get_ethnicity_vector(ethnicity))
	input_vec.append(time_left)

	best_new_kp = -1
	best_value = -1
	for new_kp in range(KP_NUM):
		if kp_vector[new_kp] == 1 or time_estimates[new_kp] > time_left:
			continue
		prereqs_met = True
		for prev_kp in prerequisite_lists[new_kp]:
			if kp_vector[prev_kp] == 0:
				prereqs_met = False
				break
		if not prereqs_met:
			continue

		####Estimate value with new kp added
		input_vec[new_kp] = 1
		input_vec[-1] -= time_estimates[new_kp]
		input_vec[KP_NUM + new_kp] = time_estimates[new_kp]

		new_state_value = Neural_Net_Estimator(torch.Tensor(input_vec))[0]
		if best_new_kp == -1 or new_state_value > best_value:
			best_new_kp = new_kp
			best_value = new_state_value

		#Undo changes to input_vec
		input_vec[new_kp] = 0
		input_vec[-1] = time_left
		input_vec[KP_NUM + new_kp] = -1
	return best_new_kp

"""
TODO: Implement performance measures for pick_action and neural network
"""