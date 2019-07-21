import numpy as np
import pandas as pd

"""
There are two videos and two tests, as well as the final exam.
Videos and assessments may cover multiple KCs.
"""
num_KC = 5
num_videos = 2
num_tests = 3


class Student:

	#Input: initial_proficiency is a Numpy array of length num_KC
	def __init__(self, initial_proficiency):
		self.proficiency = initial_proficiency

	"""
	Input: 0 <= video_id < num_videos
	Effect: proficiencies are updated according to our specifications
	Returned: Time taken to watch the video
	"""
	def watch_video(self, video_id):
		if video_id == 1:
			prof_1_flip = np.random.randint(0, 2)
			time_1_flip = np.random.random()
			time_taken = 3 if time_1_flip < self.proficiency[0] else 5
			self.proficiency[0] = max(1, self.proficiency[0] + 0.25 * prof_1_flip)
			return time_taken
		elif video_id == 2:
			pass

	"""
	Input: 0 <= test_id < num_tests
	Effect: proficiencies are updated according to our specifications
	Returned: The amount of time the exam took, and a dictionary mapping KCs to new proficiencies.
	Note: If test_id = num_tests - 1 (meaning this exam is the final exam),
	then all of the KCs are keys for this dictionary.
	"""
	def take_assessment(self, test_id):
		if test_id == 1:
			pass
		elif test_id == 2:
			pass
		elif test_id == 3:
			pass

def generate_episode():
	pass

def create_data_set(num_episodes, filename):
	pass

