import json, random, copy, sys, getopt

NUM_KC = 5
NUM_PROFICIENCY_LEVEL = 5
ACTION_PATH =  "./action.txt"#last action is "assessment test"



args = sys.argv[1:]
options = ["kc=", "proficiency=","actions="] 
try:
	arguments, _ = getopt.getopt(args, [], options)
except getopt.error as err:
	# output error, and return with an error code
	print (str(err))
	sys.exit(2)

for currentArgument, currentValue in arguments:
	if currentArgument == "--kc":
		NUM_KC = str(currentValue)
	elif currentArgument == "--proficiency":
		NUM_PROFICIENCY_LEVEL = str(currentValue)
	elif currentArgument == "--actions":
		ACTION_PATH = str(currentValue)

accuracy = 0.001
ACTIONS = {}


def add_action(name, start_states, end_states, transit_probability):
	if len(start_states)!= len(end_states) or len(start_states)!=len(transit_probability):
		print("transition pairs do not match!")
	else:
		for i in range(len(end_states)):
			if len(end_states[i]) != len(transit_probability[i]):
				print("transition paris do not match 2 !")
				return
			if 1 - sum(transit_probability[i]) > accuracy:
				print("invalid transition probability.")
				return
		ACTIONS[name]={}
		for i in range(len(start_states)):
			ACTIONS[name][str(start_states[i])] = []
			for j in range(len(end_states[i])):
				ACTIONS[name][str(start_states[i])].append({"end_state":end_states[i][j], "p":transit_probability[i][j]})


def gen_sum_set(s,l):
	if s==0:
		return [[0]*l]
	if l==0:
		return None
	res = []
	for i in range(s+1):
		if gen_sum_set(s-i,l-1) != None:
			def addi(t):
				return [i]+t
			res += map(addi, gen_sum_set(s-i,l-1))
	return res 


def valid(state):
	for p_level in state:
		if p_level < 0 or p_level >= NUM_PROFICIENCY_LEVEL:
			return False
	return True
def gen_random_transition_probability(state, max_proficiency_increment, alpha):
	unrelated_index = []
	related_plevels = []
	for i in range(len(state)):
		if state[i] == -1:
			unrelated_index.append(i)
		else:
			related_plevels.append(state[i])
	final_states = []
	for i in range(max_proficiency_increment):
		increments = gen_sum_set(i+1, len(related_plevels))
		for increment in increments:
			final_state = map(lambda x,y: x+y, related_plevels, increment)
			tokens = random.randint(5,10)*(alpha)**i
			final_states.append((final_state,tokens))
	end_states = []
	all_tokens = []
	for final_state, tokens in final_states:
		if valid(final_state):
			end_states.append(final_state)
			all_tokens.append(tokens)


	def put_back_unchanged_entries(state):
		res = []
		for i in range(NUM_KC):
			if i in unrelated_index:
				res.append(-1)
			else:
				res.append(state[0])
				state=state[1:]
		return res
	end_states = map(put_back_unchanged_entries, end_states)
	sum_tokens = sum(all_tokens)
	probability = map(lambda x: x/sum_tokens, all_tokens)
	return end_states, probability


def gen_random_input_states(related_entries):
	res = []
	if len(related_entries) == 1:
		for i in range(NUM_PROFICIENCY_LEVEL):
			res.append([-1]*related_entries[0]+[i]+[-1]*(NUM_KC - 1 - related_entries[0]))
		return res 
	else:
		entry = related_entries[0]
		rest_entries = gen_random_input_states(related_entries[1:])
		for i in range(NUM_PROFICIENCY_LEVEL):
			for p_levels in rest_entries:
				p_levels[entry] = i
			cur_part = copy.deepcopy(rest_entries)
			res += cur_part

		return res

watch_video_states = gen_random_input_states([0,1,3])
read_textbook_states = gen_random_input_states([2,4])
take_assessment_test_states = gen_random_input_states([2,3])

watch_video_end_states = []
watch_video_end_p = []
for state in watch_video_states:
	end_state, p = gen_random_transition_probability(state, 3, 0.6)
	if p!= []:
		watch_video_end_states.append(end_state)
		watch_video_end_p.append(p)


add_action("watch video", watch_video_states[:-1],watch_video_end_states,watch_video_end_p)


read_textbook_end_states = []
read_textbook_end_p = []
for state in read_textbook_states:
	end_state, p = gen_random_transition_probability(state, 2, 0.45)
	if p!= []:
		read_textbook_end_states.append(end_state)
		read_textbook_end_p.append(p)

add_action("read textbook", read_textbook_states[:-1], read_textbook_end_states, read_textbook_end_p)

take_assessment_test_end_states = []
take_assessment_test_end_p = []
for state in take_assessment_test_states:
	end_state, p = gen_random_transition_probability(state, 2, 0.3)
	if p != []:
		take_assessment_test_end_p.append(p)
		take_assessment_test_end_states.append(end_state)

add_action("take assessment test", take_assessment_test_states[:-1], take_assessment_test_end_states, take_assessment_test_end_p)



with open(ACTION_PATH, 'w') as outfile:
    json.dump(ACTIONS, outfile)

