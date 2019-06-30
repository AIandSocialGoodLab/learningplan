import random
import numpy as np
import pandas as pd
import torch
from torch import nn
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
	prereq_adj_list[current_row.target].append(current_row.source)

#Read generated data
student_records = pd.read_csv('generated_data.csv')

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


#Define a dataset for DQN, by storing the transitions





