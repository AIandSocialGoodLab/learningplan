import torch, csv, ast, copy
from torch import nn, distributions

NUM_KC = 5
NUM_PROFICIENCY_LEVEL = 5
device = "cpu"

#Represents an LSTM with hidden state of dimension
#(num_kc, num_proficiency_level)
#Represents probability distributions over each of
#the kc proficiency levels
#Uses 1-hot encoding of actions
class ProficiencyRNN(nn.Module):

	def __init__(self, num_kc, num_proficiency_level, num_actions, action_indices, action_revealed_proficiences):
		super().__init__()
		self.num_kc = num_kc
		self.num_proficiency_level = num_proficiency_level
		self.num_actions = num_actions
		self.action_indices = copy.deepcopy(action_indices)
		self.action_revealed_proficiences = copy.deepcopy(action_revealed_proficiences)
		self.lstm = nn.LSTM(num_actions, num_kc * num_proficiency_level)

	#Actions can be passed one at a time, as strings
	#hidden_state and cell_state are given as 2-D tensors representing
	#probability distributions
	def forward(self, hidden_state, cell_state, action):
		action_vec = torch.zeros(self.num_actions, device=device)
		action_vec[self.action_indices[action]] = 1
		action_vec = action_vec.view(1, 1, -1)
		hidden_state = hidden_state.view(1, 1, -1)
		cell_state = cell_state.view(1, 1, -1)

		#Since this lstm only has 1 layer, the output
		#is exactly the hidden state
		_, out = self.lstm(action_vec, (hidden_state, cell_state))
		new_hidden, new_cell = out
		new_hidden = new_hidden.view(self.num_kc, self.num_proficiency_level)
		new_cell = new_cell.view(self.num_kc, self.num_proficiency_level)
		return new_hidden, new_cell


#Given a matrix of dimension (num_kc, num_proficiency_level)
#that represents probability distributions over proficiency
#levels for each of the kc's, this function applies softmax
#to each row and then samples new proficiency levels
#Note 1: The argument is actually the matrix of logits
#Note 2: This will not be used during training, but rather, during
#testing and during generation of trajectories for training
#of RL algorithms
def sample_proficiency(prob_matrix):
	prob_dist = distributions.categorical.Categorical(logits=prob_matrix)
	new_proficiency_levels = prob_dist.sample()

	#Convert to one-hot encoding for each kc
	indices = torch.arange(prob_matrix.size()[0])
	one_hot_levels = torch.zeros_like(prob_matrix)
	one_hot_levels[indices, new_proficiency_levels] = 1
	return new_proficiency_levels, one_hot_levels

#Obtain Trajectories from Historical Data
historical_data = open("datasheet.csv", "r")
reader = csv.reader(historical_data)
trajectories = []
for line in reader:
	if line[1] == 'Student_ID':
		continue
	action_type = line[2]
	state_info = line[4]
	if state_info == '':
		state_info = {}
	else:
		state_info = ast.literal_eval(state_info)
		if isinstance(state_info, list):
			new_dict = {}
			for i in range(NUM_KC):
				new_dict[i] = state_info[i]
			state_info = new_dict
	if action_type == 'Prior Assessment Test':
		trajectories.append([])
	update = (action_type, state_info)
	trajectories[-1].append(update)
historical_data.close()

#Map actions to indices, and revealed proficiencies
action_index = {}
action_revealed_proficiences = {}
num_seen_actions = 0
for trajectory in trajectories:
	for i in range(len(trajectory)):
		action, result = trajectory[i]
		if action in action_index:
			continue
		action_index[action] = num_seen_actions
		num_seen_actions += 1
		action_revealed_proficiences[action] = []
		for kc in result.keys():
			action_revealed_proficiences[action].append(kc)

#Initialize RNN
transition_model = ProficiencyRNN(num_kc=NUM_KC, num_proficiency_level=NUM_PROFICIENCY_LEVEL, 
	num_actions=num_seen_actions, action_indices=action_index, 
	action_revealed_proficiences=action_revealed_proficiences)

######################################

"""
#Train RNN by passing actions one at a time
for epoch in range(500):
	for trajectory in trajectories:
		_, initial_prof = trajectory[0]

		#Convert initial_prof to matrix
		prob_matrix = torch.zeros(NUM_KC, NUM_PROFICIENCY_LEVEL)
		for i in range(NUM_KC):
			prob_matrix[i][initial_prof[i]] = 1

		#How to obtain initial logits?

		trajectory = trajectory[1:]
"""

