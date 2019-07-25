
import getopt, sys, math, random, json, csv
import pandas as pd

STUDENTS = 10
OUTPUT_PATH = "./dataset/learning_history.csv"
NUM_KC = 5
NUM_PROFICIENCY_LEVEL = 5
ACTION_PATH =  "./action.txt"#last action is "assessment test"
AVERAGE_PROFICIENCY_INCREMENT = 1.5
ACCURACY = 0.001

args = sys.argv[1:]
options = ["students=", "output=", "kc=", "proficiency=","actions="] 
try:
	arguments, _ = getopt.getopt(args, [], options)
except getopt.error as err:
	# output error, and return with an error code
	print (str(err))
	sys.exit(2)

for currentArgument, currentValue in arguments:
	if currentArgument == "--students":
		STUDENTS = int(currentValue)
	elif currentArgument == "--output":
		OUTPUT_PATH = str(currentValue)
	elif currentArgument == "--kc":
		NUM_KC = str(currentValue)
	elif currentArgument == "--proficiency":
		NUM_PROFICIENCY_LEVEL = str(currentValue)
	elif currentArgument == "--actions":
		ACTION_PATH = str(currentValue)


with open(ACTION_PATH) as json_file:
	all_actions = json.load(json_file)
	action_set = []
	for action in all_actions:
		action_set.append(action)
	num_actions = len(action_set)


def gen_random_output_index(l):
	if 1 - sum(l) > ACCURACY:
		print("Invalid Input for gen_random_output_index!")
		return
	else:
		p = random.random()
		index = 0
		while p > 0:
			p -=l[index]
			index += 1
		return index - 1


def generate_proficiency(l):
	p_level = []
	for i in range(NUM_KC):
		p_level.append(gen_random_output_index(l))
	return p_level


df = pd.DataFrame(columns=["Student_ID", "Action_Types", "Action_Time", "Cur_Proficiency"])
def build_row(ID, action_type, time, proficiency):
	new_row = {}
	new_row["Student_ID"] = ID 
	new_row["Action_Types"] = action_type
	new_row["Action_Time"] = time 
	new_row["Cur_Proficiency"] = proficiency
	return new_row

def state_match(s1, s2):
	for i in range(len(s1)):
		if s1[i] != s2[i] and s1[i]!= -1:
			return False 
	return True

def update_state(cur_proficiency, end_state):
	res = []
	for i in range(len(cur_proficiency)):
		if end_state[i]!= -1:
			res.append(end_state[i])
		else:
			res.append(cur_proficiency[i])
	return res

for i in range(STUDENTS):
	id_length = math.floor(math.log(STUDENTS, 10))+1
	cur_id = str(i)
	cur_id = "0"*int(id_length - len(cur_id)) + cur_id
	cur_proficiency = generate_proficiency([0.4,0.45,0.1,0.04,0.01])
	df = df.append(other = build_row(cur_id, "Prior Assessment Test", 90, cur_proficiency), ignore_index = True)
	while True:
		terminate_probability = (sum(cur_proficiency) - 10.0)/20.0
		if random.random() < terminate_probability or sum(cur_proficiency) == 20:
			break
		not_found_action = True
		step = 0
		while not_found_action and step<50:
			step += 1
			action_type = action_set[random.randint(0,len(action_set)-1)]
			time = 1 + int(60*(random.random())**1.5)
			for state in all_actions[action_type]:
				cur_state = [int(i) for i in state.strip('[]').split(',')]
				if state_match(cur_state, cur_proficiency):
					not_found_action = False
					tok = random.random()
					index = 0
					while tok > 0:
						tok -= all_actions[action_type][state][index]["p"]
						index += 1
					end_state = all_actions[action_type][state][index-1]["end_state"]
					cur_proficiency = update_state(cur_proficiency, end_state) 
					break
		if step < 50:
			if action_type[:2] == "AT":
				df = df.append(other = build_row(cur_id, action_type, time, cur_proficiency), ignore_index = True)
			else:
				df = df.append(other = build_row(cur_id, action_type, time, None), ignore_index = True)
		else:
			break
	df = df.append(other = build_row(cur_id, "Final Exam", 90, cur_proficiency), ignore_index = True)
df.to_csv(OUTPUT_PATH)


