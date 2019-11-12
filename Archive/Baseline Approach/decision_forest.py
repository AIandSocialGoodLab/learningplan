# Random Forest Classification on Tensorflow
# Use some code from https://www.kaggle.com/salekali/random-forest-classification-with-tensorflow/data

# Import libraries

from __future__ import print_function

import numpy as np

import sklearn

import pandas as pd

import tensorflow as tf

from tensorflow.contrib.tensor_forest.python import tensor_forest

from tensorflow.python.ops import resources

M = 25 #number of knowledge points
MIN_TIME = 5
MAX_TIME = 100
ETHNICITY = ["Asian", "Arab", "Black or African American", "Hispanic or Latino", "White or Cucasian"]
GENDER = ["Male", "Female"]

# Import data

data = pd.read_csv('generated_data.csv')



#Extract feature and target np arrays (inputs for placeholders)

raw_demo = data.iloc[:, 1:-2].values

raw_trace =  data.iloc[:, -2].values
raw_time = data.iloc[:, -1].values



trace = []
time = []
for i in range(len(raw_trace)):
	cur_sequence = raw_trace[i].split(", ")
	cur_time = raw_time[i].split(", ")
	cur_sequence[0] = cur_sequence[0][1:]
	cur_sequence[-1] = cur_sequence[-1][:-1]
	cur_time[0] = cur_time[0][1:]
	cur_time[-1] = cur_time[-1][:-1]
	cur_sequence = list(map(int, cur_sequence))
	cur_time = list(map(int, cur_time))
	trace.append(cur_sequence)
	time.append(cur_time)
	

def gen_feature_seq(demographic_feat, cur_trace, cur_time):
	feature_seq = []
	demographic_feat[0] = ETHNICITY.index(demographic_feat[0])
	demographic_feat[1] = GENDER.index(demographic_feat[1])
	for i in range(len(cur_trace)):
		acquired_kps = [0]*M
		for j in range(i):
			acquired_kps[cur_trace[j]] = 1
		feature_seq.append((list(demographic_feat) + acquired_kps + [cur_trace[i]], cur_time[i]//20))
	return feature_seq


dataset = []
for i in range(len(raw_demo)):
	dataset += gen_feature_seq(raw_demo[i], trace[i], time[i])

input_x, input_y = zip(*dataset)

input_x = np.array(input_x)
input_y = np.array(input_y)

print(input_x)
print(input_y)

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(input_x, input_y, test_size = 0.25, random_state = 0)




# Parameters

num_steps = 500 # Total steps to train
num_classes = 6
num_features = M + 4
num_trees = 10
max_nodes = 1500 



# Input and Target placeholders 

X = tf.placeholder(tf.float32, shape=[None, num_features])
Y = tf.placeholder(tf.int64, shape=[None])



# Random Forest Parameters
hparams = tensor_forest.ForestHParams(num_classes=num_classes, num_features=num_features, num_trees=num_trees, max_nodes=max_nodes).fill()

# Build the Random Forest
forest_graph = tensor_forest.RandomForestGraphs(hparams)



# Get training graph and loss
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)



# Measure the accuracy
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



# Initialize the variables (i.e. assign their default value) and forest resources
init_vars = tf.group(tf.global_variables_initializer(), resources.initialize_resources(resources.shared_resources()))

sess = tf.Session()
sess.run(init_vars)



# Training

for i in range(1, num_steps+1):
    _, l = sess.run([train_op, loss_op], feed_dict={X: X_train, Y: y_train})
    if i % 50 == 0:
        acc = sess.run(accuracy_op, feed_dict={X: X_train, Y: y_train})
        print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))



# Test Model
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: X_test, Y: y_test}))
