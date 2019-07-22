import numpy as np
import pandas as pd

"""
There are three videos and three tests (not including the final exam).
Videos and assessments may cover multiple KCs.
"""
num_KC = 5
num_videos = 3
num_tests = 4


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
		if video_id == 1: #Affects KC 0
			prof_0_flip = np.random.randint(0, 2)
			time_0_flip = np.random.random()
			time_taken = 3 if time_0_flip < self.proficiency[0] else 5
			self.proficiency[0] = max(1, self.proficiency[0] + 0.25 * prof_0_flip)
			return time_taken
		elif video_id == 2: #Affects KCs 0, 1 and 2
			original_prof_0 = self.proficiency[0]

			#Updates to proficiency in KC 0
			if original_prof_0 == 0:
				self.proficiency[0] = max(1, self.proficiency[0] + 0.25)
			else:
				prof_0_flip = np.random.random()
				if prof_0_flip <= 0.25:
					self.proficiency[0] = max(1, self.proficiency[0] + 0.25)
				else:
					self.proficiency[0] = max(1, self.proficiency[0] + 0.5)

			#Updates to proficiency in KC 1
			"""
			This will depend on both the old and new proficiencies in KC 0.
			Intuition: If the old proficiency in KC 0 was low, then the student
			will have to put more energy into mastering KC 0, in order to get anything
			out of the video's content in KC 1. If the student has improved in KC 0 due 
			to the video, however, then he/she will be likely to understand more of what
			the video shows on KC 1.
			"""
			avg_prof_0 = (original_prof_0 + self.proficiency[0])/2
			original_prof_1 = self.proficiency[1]
			if avg_prof_0 < 0.5:
				self.proficiency[1] += 0.25
			else:
				prof_1_flip = np.random.random()
				if prof_1_flip <= 0.75:
					self.proficiency[1] = max(1, self.proficiency[1] + 0.25)
				else:
					self.proficiency[1] = max(1, self.proficiency[1] + 0.5)

			#Updates to proficiency in KC 2
			#Depends on proficiencies in KCs 0 and 1
			#Depends more on KC 0 than KC 1
			avg_prof_1 = (original_prof_1 + self.proficiency[1])/2
			avg_prof_0_1 = 0.75 * avg_prof_0 + 0.25 * avg_prof_1
			original_prof_2 = self.proficiency[2]
			if avg_prof_0_1 < 0.5:
				self.proficiency[2] = max(1, self.proficiency[0] + 0.25)
			else:
				prof_2_flip = np.random.random()
				if prof_2_flip <= 0.75:
					self.proficiency[2] = max(1, self.proficiency[2] + 0.25)
				else:
					self.proficiency[2] = max(1, self.proficiency[2] + 0.5)

			#Time taken
			time_flip = np.random.random()
			time_taken = 2 if time_flip < original_prof_0 * original_prof_1 * original_prof_2 else 5
			return time_taken 

		elif video_id == 3: #Affects KCs 2, 3 and 4
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
		elif test_id == 4:
			pass

def generate_episode():
	pass

def create_data_set(num_episodes, filename):
	pass

