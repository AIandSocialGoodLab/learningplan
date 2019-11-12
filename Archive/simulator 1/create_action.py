import json, random, copy, sys, getopt
from simulator import NUM_KC, NUM_PROFICIENCY_LEVEL, ACTION_PATH
from simulator import gen_random_output_index, ACCURACY, NUM_ACTIONS, NUM_ASSESSMENT_TESTS

ACTIONS = {"reveal_plevels":{}, "actions":{},"related_entries":{}}


def add_action(name, related_kc, related_entries, start_states, end_states, transit_probability):
	if len(start_states)!= len(end_states) or len(start_states)!=len(transit_probability):
		print("transition pairs do not match!")
	else:
		for i in range(len(end_states)):
			if len(end_states[i]) != len(transit_probability[i]):
				print("transition paris do not match 2 !")
				return
			if 1 - sum(transit_probability[i]) > ACCURACY:
				print("invalid transition probability.")
				return
		ACTIONS["actions"][name]={}
		ACTIONS["reveal_plevels"][name] = related_entries 
		ACTIONS["related_entries"][name] = related_kc
		for i in range(len(start_states)):
			ACTIONS["actions"][name][str(start_states[i])] = []
			for j in range(len(end_states[i])):
				ACTIONS["actions"][name][str(start_states[i])].append(
					{"end_state":end_states[i][j], "p":transit_probability[i][j]})


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
def gen_random_transition_probability(state, max_proficiency_increment, alpha = 0.6):
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


def gen_random_input_states(related_entries, minimum_required_plevel, alpha = 0.8):
	res = []
	if len(related_entries) == 0:
		return [[-1]*(NUM_KC)]
	else:
		main_entry = random.randint(0,len(related_entries)-1)
		entry = related_entries[main_entry]
		for i in range(minimum_required_plevel[main_entry], NUM_PROFICIENCY_LEVEL):
			new_related_entries = []
			new_minimum_required_plevel = []
			for j in range(len(related_entries)):
				if random.random() < alpha and j != main_entry:
					new_related_entries.append(related_entries[j])
					new_minimum_required_plevel.append(minimum_required_plevel[j])
			rest_entries = gen_random_input_states(new_related_entries, new_minimum_required_plevel)
			for p_levels in rest_entries:
				p_levels[entry] = i
			cur_part = copy.deepcopy(rest_entries)
			res += cur_part

		return res




for i in range(NUM_ACTIONS+NUM_ASSESSMENT_TESTS):
	action_id = str(i)
	if i >= NUM_ACTIONS:
		action_id = "AT" + action_id
	related_entries = []
	minimum_required_plevel = []
	include_p = 0.5
	for i in range(NUM_KC):
		if random.random()>0.3:
			related_entries.append(i)
		minimum_required_plevel.append(gen_random_output_index([0.6,0.3,0.1]))
	if related_entries == []:
		related_entries.append(random.randint(0,NUM_KC-1))
	start_states = gen_random_input_states(related_entries, minimum_required_plevel)
	valid_start_states =[] 
	end_states = []
	end_p = []
	for state in start_states:
		max_proficiency_increment = gen_random_output_index([0.1,0.6,0.3])
		end_state, p = gen_random_transition_probability(state, max_proficiency_increment)
		if p!= []:
			valid_start_states.append(state)
			end_states.append(end_state)
			end_p.append(p)
	related_kc = related_entries
	if action_id[:2] != "AT":
		related_entries = []
	add_action(action_id,related_kc, related_entries, valid_start_states,end_states, end_p)




with open(ACTION_PATH, 'w') as outfile:
    json.dump(ACTIONS, outfile)

