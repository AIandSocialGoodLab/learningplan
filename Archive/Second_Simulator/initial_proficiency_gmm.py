import numpy as np
import pandas as pd
import sklearn.mixture as mixture
import matplotlib.pyplot as plt
import ast, math, pickle, random

#Read number of KCs and proficiency levels
settings = open("settings.txt", "r")
NUM_KC = int(settings.readline())
NUM_PROFICIENCY_LEVEL = int(settings.readline())
settings.close()

#Input: Pandas dataframe, in the same data format as MDPdatasheet.csv
#Output: Numpy array, where each row is the initial proficiency at the
#beginning of some episode
def read_initial_proficiencies(transition_set):
	transition_set = transition_set[transition_set['Action_Types'] == 'Prior Assessment Test']
	prof_list = []
	for i in range(transition_set.shape[0]):
		prof_KC_str = transition_set.iloc[i]['Cur_Proficiency']
		prof_list.append(ast.literal_eval(prof_KC_str))
	return np.array(prof_list)

#Input: Training data (numpy array) for GMM, and number of components in mixture
#Output: GMM with full covariance matrices
def train_gmm_full(train_data, n_components):
	model = mixture.GaussianMixture(n_components=n_components, covariance_type='full')
	model.fit(train_data)
	return model

#Input: Training data (numpy array) for GMM, and number of components in mixture
#Output: GMM with diagonal covariance matrices
def train_gmm_diag(train_data, n_components):
	model = mixture.GaussianMixture(n_components=n_components, covariance_type='diag')
	model.fit(train_data)
	return model

#Output: GMM with at least min_components and at least max_components components, and either
#full or diagonal covariance matrix. Several models are trained with MLE (EM) on training data,
#and the one which maximizes likelihood on validation data is selected.
def select_gmm(train_data, validation_data, min_components, max_components):
	model_list = []
	for i in range(min_components, max_components + 1):
		model_list.append(train_gmm_full(train_data, i))
		model_list.append(train_gmm_diag(train_data, i))

	likelihood_list = []
	for model in model_list:
		likelihood_list.append(model.score(validation_data))

	print("Likelihoods of the model: ")
	print(likelihood_list)

	largest_likelihood_index = np.argmax(likelihood_list)
	return model_list[largest_likelihood_index]

#Split into training, validation and testing data
transition_set = pd.read_csv("MDPdatasheet.csv")
initial_prof_array = read_initial_proficiencies(transition_set)

test_train_split = open("test_train_split.txt")
train_percent = int(test_train_split.readline())
validation_percent = int(test_train_split.readline())
test_percent = int(test_train_split.readline())
test_train_split.close()

#Part 1: Train on historical data
num_examples = initial_prof_array.shape[0]
train_num = int(math.floor(num_examples * train_percent/100))
validation_num = int(math.floor(num_examples * (train_percent + validation_percent)/100))
train_data = initial_prof_array[:train_num]
validation_data = initial_prof_array[train_num:validation_num]
test_data = initial_prof_array[validation_num:]
initial_prof_GMM = select_gmm(train_data, validation_data, 1, 8)

print()
print("Description of Selected GMM:")
print(initial_prof_GMM)
print()

#Part 2: Compare the proficiencies created by the GMM to the original simulator
#The proficiencies created by the GMM are restricted to [0, NUM_PROFICIENCY_LEVEL - 1]
#and then rounded to the nearest integer.
def make_valid_prof(l):
	l = np.minimum(l, NUM_PROFICIENCY_LEVEL - 1)
	l = np.maximum(l, 0)
	return (l + 0.5).astype(int)

truth_samples = test_data
NUM_SAMPLES = truth_samples.shape[0]

gmm_samples = initial_prof_GMM.sample(NUM_SAMPLES)
gmm_samples = gmm_samples[0] #throw away the component labels
gmm_samples = make_valid_prof(gmm_samples)
print(gmm_samples)

#Collect the samples for each of the different KCs
kc_samples_gmm = np.zeros((NUM_KC, NUM_PROFICIENCY_LEVEL))
kc_samples_truth = np.zeros((NUM_KC, NUM_PROFICIENCY_LEVEL))

for i in range(NUM_SAMPLES):
	for kc in range(NUM_KC):
		gmm_i_kc = gmm_samples[i][kc]
		kc_samples_gmm[kc][gmm_i_kc] += 1

		truth_i_kc = truth_samples[i][kc]
		kc_samples_truth[kc][truth_i_kc] += 1

print()
print("=====================Comparison of Distributions=====================")
for kc in range(NUM_KC):
	print()
	print("========KC #%d" % kc)

	#Print the Distributions
	gmm_dist = kc_samples_gmm[kc] / NUM_SAMPLES
	truth_dist = kc_samples_truth[kc] / NUM_SAMPLES
	print("GMM Empirical Distribution: ")
	print(gmm_dist)
	print("Ground-Truth Empirical Distribution")
	print(truth_dist)

	#Calculate KL Divergence of GMM Dist. from Truth Dist. (Empirical)
	kl_div = np.log(gmm_dist/truth_dist)
	kl_div = gmm_dist * kl_div
	kl_div = np.sum(kl_div)
	print()
	print("KL Divergence: ")
	print(kl_div)
	print()

	#Plot the Distributions
	#The bars have width 1/3. The gap between the bars is 1/3
	truth_bar_x = np.linspace(1, NUM_PROFICIENCY_LEVEL, NUM_PROFICIENCY_LEVEL)
	gmm_bar_x = np.linspace(1 + 1.0/3, NUM_PROFICIENCY_LEVEL + 1.0/3, NUM_PROFICIENCY_LEVEL)

	plt.bar(x=truth_bar_x, height=truth_dist, width=1.0/3, align='edge', label='Ground Truth') #align edges to x coordinates
	plt.bar(x=gmm_bar_x, height=gmm_dist, width=1.0/3, align='edge', label='GMM')
	plt.legend()
	plt.xlabel("Distribution of KC # %d at beginning of course (first bar is truth, second is GMM)" % kc)

	plt.show()


#Save the model
out_file = open('gmm_pickled.txt', 'wb') #Source: https://stackoverflow.com/questions/12092527/python-write-bytes-to-file
gmm_str = pickle.dumps(initial_prof_GMM)
out_file.write(gmm_str)
out_file.close()