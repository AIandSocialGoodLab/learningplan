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
		self.proficiency = np.copy(initial_proficiency)

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
			self.proficiency[0] = min(1, self.proficiency[0] + 0.25 * prof_0_flip)
			return time_taken
		elif video_id == 2: #Affects KCs 0, 1 and 2
			original_prof_0 = self.proficiency[0]

			#Updates to proficiency in KC 0
			if original_prof_0 >= 0.5:
				prof_0_flip = np.random.random()
				if prof_0_flip <= 0.75:
					self.proficiency[0] = min(1, self.proficiency[0] + 0.5)
				else:
					self.proficiency[0] = min(1, self.proficiency[0] + 0.25)

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
			if avg_prof_0 <= 0.5:
				self.proficiency[1] = min(1, self.proficiency[1] + 0.25)
			else:
				prof_1_flip = np.random.random()
				if prof_1_flip <= 0.75:
					self.proficiency[1] = min(1, self.proficiency[1] + 0.5)
				else:
					self.proficiency[1] = min(1, self.proficiency[1] + 0.25)

			#Updates to proficiency in KC 2
			#Depends on proficiencies in KCs 0 and 1
			#Depends more on KC 0 than KC 1
			avg_prof_1 = (original_prof_1 + self.proficiency[1])/2
			avg_prof_0_1 = 0.75 * avg_prof_0 + 0.25 * avg_prof_1
			original_prof_2 = self.proficiency[2]
			if avg_prof_0_1 <= 0.5:
				self.proficiency[2] = min(1, self.proficiency[0] + 0.25)
			else:
				prof_2_flip = np.random.random()
				if prof_2_flip <= 0.75:
					self.proficiency[2] = min(1, self.proficiency[2] + 0.5)
				else:
					self.proficiency[2] = min(1, self.proficiency[2] + 0.25)

			#Time taken
			time_flip = np.random.random()
			time_taken = 2 if time_flip < original_prof_0 * original_prof_1 * original_prof_2 else 5
			return time_taken 

		elif video_id == 3: #Affects KCs 3 and 4, depends on KCs 1 and 2

			#Updates to Proficiency in KC 3 and 4
			original_prof_3 = self.proficiency[3]
			original_prof_4 = self.proficiency[4]

			#Probabilities that determine increases in proficiency for KCs 3 and 4
			#Intuitively, KCs 1 and 2 are equally important for learning KC 3, while
			#KC 2 is more important than KC 1 for learning KC 4
			update_prob_3 = 0.5 * self.proficiency[1] + 0.5 * self.proficiency[2]
			update_prob_4 = 0.25 * self.proficiency[1] + 0.75 * self.proficiency[2]

			prof_3_flip = np.random.random()
			if prof_3_flip <= update_prob_3:
				self.proficiency[3] = min(1, self.proficiency[3] + 0.5)
			else:
				self.proficiency[3] = min(1, self.proficiency[3] + 0.25)

			prof_4_flip = np.random.random()
			if prof_4_flip <= update_prob_4:
				self.proficiency[4] = min(1, self.proficiency[4] + 0.5)
			else:
				self.proficiency[4] = min(1, self.proficiency[4] + 0.25)

			#Time taken
			time_prob = 0.125 * self.proficiency[1] + 0.125 * self.proficiency[2] + 0.375 * self.proficiency[3] + 0.375 * self.proficiency[4]
			time_flip = np.random.random()
			time_taken = 4 if time_flip < time_prob else 8
			return time_taken

	"""
	Input: 0 <= test_id < num_tests
	Effect: proficiencies are updated according to our specifications
	Returned: The amount of time the exam took, and a dictionary mapping KCs to new proficiencies.
	Note: If test_id = num_tests - 1 (meaning this exam is the final exam),
	then all of the KCs are keys for this dictionary.
	"""
	def take_assessment(self, test_id):
		"""
		Currently, assessments are deterministic. Intuitively, students with
		higher proficiencies gain more from taking assessments, while
		students with lower proficiencies do not improve as much, and
		will have to watch videos to be ready.
		"""

		if test_id == 1: #Tests KCs 0, 1
			prof_dict = {}
			prof_dict[0] = self.proficiency[0]
			prof_dict[1] = self.proficiency[1]

			time_taken = 0
			if self.proficiency[0] <= 0.25 and self.proficiency[1] <= 0.25:
				time_taken = 20
			elif self.proficiency[1] <= 0.25:
				time_taken = 12
				self.proficiency[0] = 1
			elif self.proficiency[0] <= 0.25:
				time_taken = 16
				self.proficiency[1] = min(1, self.proficiency[1] + 0.25)
			else:
				time_taken = 8
				self.proficiency[0] = 1
				self.proficiency[1] = 1

			return time_taken, prof_dict


		elif test_id == 2: #Tests KCs 1, 2, 3
			prof_dict = {}
			prof_dict[1] = self.proficiency[1]
			prof_dict[2] = self.proficiency[2]
			prof_dict[3] = self.proficiency[3]

			time_taken = 0
			if self.proficiency[1] <= 0.5 or self.proficiency[2] <= 0.5:
				#No proficiency improvements
				#All questions require deep knowledge of KCs 1 and 2
				time_taken = 30
			else:
				self.proficiency[1] = 1
				self.proficiency[2] = 1
				if self.proficiency[3] == 0:
					time_taken = 20
				else:
					time_taken = 15
					if self.proficiency[3] <= 0.5:
						self.proficiency[3] = 0.5 #This test will not help too much in learning KC 3.

			return time_taken, prof_dict

		elif test_id == 3: #Tests KCs 2, 3, 4
			prof_dict = {}
			prof_dict[2] = self.proficiency[2]
			prof_dict[3] = self.proficiency[3]
			prof_dict[4] = self.proficiency[4]

			time_taken = 0
			if self.proficiency[2] < 1 or self.proficiency[3] <= 0.25:
				#Full knowledge of KC 2 required to obtain improvements from this test
				time_taken = 30
			else:
				self.proficiency[3] = 1
				self.proficiency[4] = min(1, self.proficiency[4] + 0.25)
				time_taken = 15

			return time_taken, prof_dict

		elif test_id == 4: #Tests all KCs
			prof_dict = {}
			num_low = 0 #Number of KCs where proficiency is at most 0.5
			for i in range(num_KC):
				prof_dict[i] = self.proficiency[i]
				if self.proficiency[i] <= 0.5:
					num_low += 1

			time_taken = 0
			if num_low >= 3:
				#No proficiency improvement
				time_taken = 45
			elif num_low == 2:
				time_taken = 35
			elif num_low == 1:
				time_taken = 30
			else:
				time_taken = 25

			if num_low <= 2:
				#Student masters KCs where proficiency is at least 0.75
				for i in range(num_KC):
					if self.proficiency[i] >= 0.75:
						self.proficiency[i] = 1

			return time_taken, prof_dict

def generate_episode(initial_proficiency):
	"""
	Right now, a uniformly random policy is being used to generate data,
	and this may not be realistic - students can select actions more
	wisely than this.
	"""

	test_student = Student(initial_proficiency)
	action_type = [] #0 if video, 1 if assessment
	action_id = []
	action_time = []
	action_results = [] #None if video, dictionary of proficiencies if assessment
	while True:
		student_passed = False
		picked_action = np.random.randint(1, 8)
		if picked_action <= 3:
			video_id = picked_action
			action_type.append(0)
			action_id.append(video_id)
			time_taken = test_student.watch_video(video_id)
			action_time.append(time_taken)
			action_results.append(None)
		else:
			test_id = picked_action - 3
			action_type.append(1)
			action_id.append(test_id)
			time_taken, new_profs = test_student.take_assessment(test_id)
			action_time.append(time_taken)
			action_results.append(new_profs)

			#Terminal state is where all proficiencies are 1
			if test_id == 4:
				student_passed = True
				for i in range(num_KC):
					if new_profs[i] < 1:
						student_passed = False
		if student_passed:
			break

	return action_type, action_id, action_time, action_results

def create_data_set(num_episodes, filename):
	df = pd.DataFrame(columns=["Action_Types", "Action_IDs", "Action_Times", "Action_Results"])
	print(-1)
	for i in range(num_episodes):
		action_type, action_id, action_time, action_results = generate_episode(np.zeros(num_KC))
		new_row_dict = {}
		new_row_dict["Action_Types"] = action_type
		new_row_dict["Action_IDs"] = action_id
		new_row_dict["Action_Times"] = action_time
		new_row_dict["Action_Results"] = action_results
		df = df.append(other=new_row_dict, ignore_index=True)
		print(i)
	df.to_csv(filename)

create_data_set(10, "student_histories.csv")