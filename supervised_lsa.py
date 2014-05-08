# COS 401 Final Project
# Kiran Vodrahalli
# Evelyn Ding
# Albert Lee
# handle_data.py: 
# does data processing:
## - transform corpus into multiple words (i.e. bank1, bank2, etc.)
## - build matrix for LSA 
## - modified Lesk algorithm for calculating multiple-sense score (MSS)
## - filter words that should be multiple based on MSS and TF-IDF or Wordnet
## - actually apply LSA (using SVD/ PCA)

import numpy as np 
from numpy.linalg import svd 
from numpy.linalg import inv
from numpy.linalg import norm 
from nltk.corpus import brown  


punctuation_list = [",", ".", ";", "\"", "'", "!", "?", ":"]

# makes words in nice format
def nice_word(w):
	return ''.join(c for c in w.lower() if c not in punctuation_list)
# from swarthmore paper: 

# The training data contains documents that each correspond to a specific semantic sense.
# This data is then parsed to form a large term-document matrix which is then decomposed
# into U, D, V^T matrices. 

# The test data is parsed into individual vectors for each document,
# containing the counts for each term in the document. Within this document, there exists a
# single test word to be disambiguated. 

# The test vectors will be compared after being folded into semantic space by multiplying 
# by U* and S^-1 * (the dimension-reduced versions)


## TWO THINGS WE CAN DO

## 1) row vectors are specific word-senses, column vectors are distance-scores to non-ambiguous words
##    -- meaning, for each row: we have a word, w. we want to make the columns -- nah


## 2) the columns are each a document about a certain topic. we have the rows as words (not sense-specific)
##    then, we look at the number of occurrences for a given word in each document topic and write it down. 
##    we take the logs of these values and average by row. then we do SVD. resulting analysis is based around
##    the column vectors (the topics.) basically, we reduce dimension appropriately after SVD. this is then our model.
##    now what we do is we are given a paragraph or something, and we project the term-document matrix that this relates to
##    into the space of our model (i.e. we want to figure out what topic it is -- BECAUSE TOPICS ARE A ONE-TO-ONE MAP TO 
##    WORD SENSE -- OUR INPUT DATA NEEDS TO BE TOPIC -BASED HERE, WE CAN USE BROWN CORPUS FOR DIFFERENT TOPICS). 
##    then we can use cosine similarity to determine which topic column it is most similar to. So we have our topic column
##    that we are most similar to now. good. 

##    now what we have to do is use the training data appropriately: so we have a bunch of senses and example 
##    paragraphs where it is labeled what sense the word we want to disambiguate has. we get a probability measure
##    for mappings between topic column vectors and between word senses for a word. WE MAY HAVE THE OCCASION THAT 
##    THE NUMBER OF WORD SENSES IS LESS THAN THE NUMBER OF TOPIC VECTORS. this is something to watch out for. (i.e.
##    what if the topic column vector that a word usage matches to is none of the columns for which a sense is recognized)
##    
##    SO: how do we get the word sense from the training data (tagged word-sense associated with paragraph). 
##    we figure out what topic the paragraph is supposed to be ==> T1, T2, etc. <- one of these
##    we have a word-sense vector: word_sense1: T1:3, T2:0, T3:4, T4:1000
## 								   word_sense2: T1:89, T2: 38, T3:39, T4:6
##								   word_sense3: T1:8, T2: 398, T4:93, T4:82
##								   etc. 

##
##	  If our input test data is just a paragraph where we are given the word we have to disambiguate, then the option we 
##    have is to see what the topic is for the paragraph (based on projecting as discussed before). When we get our topic,
##    we sum the columns over that topic and average to see which word_sense is most likely. Suppose we got T3 as our
##    topic for a paragraph containing ambiguous word "word". then we would see 93/(4 + 39 + 93) is the probability that
##    the word sense is word_sense3 with that probability, and that is what we would guess the word_sense is. 

##    IF however our input test data is a SET OF PARAGRAPHS EACH ON ONE OF THE TOPICS Ti where we know usage is the same
##    (just unknown)
##    (we want to cover all possible topics), THEN WHAT WE COULD DO is build a WORD_SENSE VECTOR for our unknown 
##    word sense and USE COSINE SIMILARITY ON THE WORD_SENSE VECTORS to determine which word_sense is most likely. 
##    the reason this is valid is because a given word sense would have differing frequencies on different topics--
##    it is possible that "bank" in the sense of river bank is just used VERY RARELY in financial texts. of course
##    it is allowed that there are 0 occurrences. 
##    weaknesses here: if all other topics are 0 counts (i.e. T1:0, T2:0, T3:48, T4:0), then this is exactly the
##    same as the previously discussed approach. 
##    another weakness is that we might not have the knowledge that a word is used in a specific sense in T1, T2, etc. 
##    also we might need a lot of data to do this.


##    NOTE: this approach is fundamentally topic-based. Also there are some words where there are more than 15 senses.
##          this doesn't matter because we calculate topic array for each word_sense. (word_sense rows, topic columns)

##    This is a more hardcore "contextual-overlap" approach 

def topic_modeler():
	words = dict()
	category_map = dict()
	cat_num = 1
	for category in brown.categories(): 
		category_map[category] = cat_num
		for w in brown.words(categories=category):
			w = nice_word(w)
			if w not in words:
				words[w] = []
				for i in range(1, cat_num):
					words[w].append(0)
				words[w].append(1) # at index cat_num - 1
			else: 
				curr_index = len(words[w]) - 1
				if curr_index < cat_num -1: 
					while curr_index < cat_num -2: 
						words[w].append(0)
						curr_index += 1
					assert (len(words[w]) == cat_num -1)
					words[w].append(1)
				else:
					words[w][cat_num - 1] += 1
		cat_num += 1
	cat_num = len(brown.categories())
	list_form = words.values()
	for l in list_form:
		while len(l) < cat_num:
			l.append(0)
		assert len(l) == cat_num
	for l in list_form: 
		# normalize
		summed = sum(l) + 0.0
		for i in range(0, len(l)):
			l[i] = l[i]/summed
	# word_dict is so that we have a map between row # and word
	word_dict = dict()
	i = 1
	for w in words.keys():
		word_dict[w] = i
		i += 1

	return word_dict, np.matrix(list_form), category_map

# M = matrix  
# http://stackoverflow.com/questions/1730600/principal-component-analysis-in-python 
# http://alias-i.com/lingpipe/demos/tutorial/svd/read-me.html
# for more reference, read Section 8.2 in: 
##   http://www.ling.ohio-state.edu/~kbaker/pubs/Singular_Value_Decomposition_Tutorial.pdf
# Do SVD on M, do dimension reduction 
# dimension reduction based on 
## -- Guttman-Kaiser criterion: remove singular values with value < 1
## -- sum squares is 85% of total square summed singular values
## -- COULD ALSO USE INSTEAD AN ENTROPY BASED METHOD
## -- COULD ALSO USED FROBENIUS NORM BASED METHOD TO FIND STRUCTURE
## -- see https://www.mpi-inf.mpg.de/departments/d5/teaching/ss13/dmm/slides/03-svd-handout.pdf
# return resulting matrices 
def pca(M):
	U, s, Vt = svd(M, full_matrices=False)
	V = Vt.T
	D = np.diag(s)
	# we want to chop of irrelevant part of s
	s_s = sorted(s)
	s_len = len(s)
	chop_num1 = 0
	for i in range(0, s_len):
		if s_s[i] < 1: 
			chop_num1 += 1
	# now chop
	for i in range(0, chop_num1):
		s = np.delete(s, (s_len - (i + 1)))
	s_len = len(s)
	# square sum must be 90% of total square summed singular values
	total_sq_sum = sum(v*v for v in s)
	curr_sq_sum = 0
	chop_num2 = 0
	for i in range(0, s_len):
		curr_sq_sum += s[i]*s[i]
		if curr_sq_sum > 0.85*total_sq_sum:
			chop_num2 += 1
	chop_num = chop_num1 + chop_num2
	# now chop
	D = D[:(s_len - chop_num), :(s_len - chop_num)]
	
	# now take care of deleting from U and V appropriately 
	# delete chop_num rows from the bottoms of U and V 
	U = U[:, :(s_len - chop_num)]
	V = V[:, :(s_len - chop_num)]

	reconstructed = np.dot(U, np.dot(D, V.T))
	return U, D, V, reconstructed

# actual values for use
# calculate once
wd, term_docM, cm = topic_modeler()
U_, D_, V_, recon = pca(term_docM)

# projects the document into the space defined by the pca
# document is given as a string
def project(doc_str):
	# need to build the column vector as though
	# doc is a new column component in the term-document matrix 
	doc = doc_str.split(" ")
	doc_vec = np.zeros(len(U_)).T
	num_missing_words = 0
	for w_ in doc: 
		w = nice_word(w_)
		if w in wd.keys(): # make sure word is in the set of known words
			doc_vec[wd[w] -1] += 1
		else: 
			# we have a word in the document not in the topic words etc. 
			# solution: we just ignore it. 
			# we can later implement something else to update this knowledge (increasing num of columns)
			num_missing_words += 1

	# print num_missing_words

	# then, project this document into the appropriate space for comparison
	# by multiplying the column vector by U_ * D_^(-1)
	# i.e.: new_doc' = new_doc * U_ * D_^(-1)
	new_doc_vec = np.dot(doc_vec, np.dot(U_, inv(D_)))
	return new_doc_vec

# use cosine similarity to check doc_vec (a document column vector)
# against all the other topic vectors in the reduced space
# return the topic which is closest it
def most_sim_topic(doc_vec):
	doc_vec = np.array(doc_vec)
	cosines = map(lambda v: np.vdot(np.array(v), doc_vec)/(norm(np.array(v)) * norm(doc_vec)) , V_)
	max_v = -2 # out of range of cosine
	max_index = 0
	for i in range(0, len(cosines)):
		if cosines[i] > max_v:
			max_index = i
			#print brown.categories()[max_index]
			max_v = cosines[i]
			#print max_v
	# print max_v
	# print brown.categories()[max_index]
	return max_index

# train_data:
# list of tuples of string, word, classification
# i.e: 
##
##   ("They went the store and then ....", "store", /wordnet_sense_id/)

## NOTE: from basic tests, it seems that having the brown corpus as the columns
##       is not the greatest, what we may want to do is hand-generate better 
##       corpuses for the specific words involved and use these as our columns


## probably should re-write this so it is more clear
def train_model(train_data):
	word_sense_dict = dict()
	for (doc_str, w, sense) in train_data:
		topic_id = most_sim_topic(project(doc_str)) # in array notation
		if w not in word_sense_dict:
			word_sense_dict[w] = dict()
			word_sense_dict[w][sense] = []
			i = 0
			while i <= topic_id - 1:
				word_sense_dict[w][sense].append(0)
				i += 1
			word_sense_dict[w][sense].append(1)
		else:
			if sense not in word_sense_dict[w]:
				word_sense_dict[w][sense] = []
				i = 0
				while i <= topic_id - 1:
					word_sense_dict[w][sense].append(0)
					i += 1
				word_sense_dict[w][sense].append(1)
			else: 
				w_sense_len = len(word_sense_dict[w][sense])
				if topic_id <= w_sense_len -1:
					word_sense_dict[w][sense][topic_id] += 1
				else:
					for i in range(0, topic_id - w_sense_len):
						word_sense_dict[w][sense].append(0)
					word_sense_dict[w][sense].append(1)
	# normalize
	# do we want to normalize only with respect to columns? 
	# no, we want to normalize along rows AND along columns
	for w in word_sense_dict.keys():
		for sense in word_sense_dict[w].keys():
			# note if it's not full length (i.e. all 9 columns), the rest will just be zero
			summed = sum(word_sense_dict[w][sense]) + 0.0
			for i in range(0, len(word_sense_dict[w][sense])):
				word_sense_dict[w][sense][i] /= summed
	return word_sense_dict

# some test train_data
td1 = [("i like chasing rivers and running alongside banks", "bank", "wn_bank_1"), ("i like going to the bank and getting money", "bank", "wn_bank_2"), ("i am scared of bats in caves", "bat", "wn_bat_1"), ("i play baseball with my bat", "bat", "wn_bat_2")]

td2 = [("i like chasing rivers and running alongside banks", "bank", "wn_bank_1"), ("the river bank is an interesting place because of the animals and plants", "bank", "wn_bank_1"), ("i like going to the bank and getting money", "bank", "wn_bank_2"), ("i am scared of bats in caves", "bat", "wn_bat_1"), ("i play baseball with my bat", "bat", "wn_bat_2")]


# takes labled data from somewhere 
# and turns it into format that train_model uses
# (i.e. a list of (doc_str, w, sense))
# ideally, doc_str contains a bunch of sentences spaced NORMALLY. 
# WE WANT JUST INDIVIDUAL WORDS (punctuation allowed to be tagged on)
# when we .split(" ") the sentence. 
def make_training_data():
	print "not implemented\n"
	return td2 # for now

trained_model = train_model(make_training_data())

# given an ambiguous word and a context
# which it is in, return the word sense from wordnet
# word is just a string
# context is a string (i.e. a doc_str)
def guess_word_sense(word, context):
	word = nice_word(word)
	topic_id = most_sim_topic(project(context))
	likelihoods = []
	for sense in trained_model[word].keys():
		curr_word_sense = trained_model[word][sense]
		if topic_id > len(curr_word_sense) -1:
			# print brown.categories()[len(curr_word_sense) - 1]
			# print brown.categories()[topic_id]
			likelihoods.append((sense, 0 + 0.0))
		else:
			likelihoods.append((sense, curr_word_sense[topic_id]))
	summed = 0.0
	for i in range(0, len(likelihoods)):
		summed += likelihoods[i][1]
	error_sense = "Sorry, the data indicates that all senses have probability 0. Sadface."
	if summed == 0.0:
		return error_sense #, 0.0, likelihoods #(the latter two are for testing)
	likelihoods = map(lambda (ws, p): (ws, p/summed), likelihoods)
	max_prob = 0.0
	max_sense = error_sense
	for (sense, probability) in likelihoods:
		if probability > max_prob:
			max_prob = probability
			max_sense = sense
	return max_sense #, max_prob, likelihoods #(the latter two are for testing)



# OLD STUFF # 

'''
# transforms the corpus into multiple-word representation (MWR)
# c is a .txt file containing corpus
# MIGHT NOT USE THIS AT ALL ACTUALLY! 
def transform_corpus(c):
	transformed = []
	non_word_tokens = set([".", ",", ";", "'", "\"", "~", "`", "+", "-"])
	word_dict = dict()
	with open(c, 'r') as f_in:
		for line in f_in:
			ll = line.split(" ")
			trans_line = []
			for w in ll: 
				if w not in non_word_tokens:
					if w not in word_dict:
						word_dict[w] = 1
					else:
						word_dict[w] += 1
					trans_line.append(w + str(word_dict[w]))
			ftrans_line = " ".join(trans_line)
			transformed.append(ftrans_line)
	return " ".join(transformed)
				

# so basically, every "context column" should be a paragraph
# about a specific version of the topic? 

# context_list is a list of separated 
# contexts from a document, where each context is a string
# so you need to do the preprocessing work beforehand,
# and put it into context_list. 
# RETURN the LSA MATRIX and WORD-TO-ROW# DICTIONARY AS A TUPLE.
def lsa_matrix_dict(context_list):
	cols = len(context_list)
	# each word is a row
	# each word gets a list
	# then we'll stick this into a numpy matrix.
	words = dict()
	context_num = 1
	for context in context_list: 
		c = context.split(" ")
		for w in c: 
			if w not in words:
				words[w] = []
				for i in range(1, context_num):
					words[w].append(0)
				words[w].append(1)
			else: 
				if len(words[w]) == context_num - 1:
					words[w].append(1)
				else:
					words[w][context_num -1] += 1
		context_num += 1
	list_form = words.values()
	# word_dict is so that we have a map between row # and word
	word_dict = dict()
	i = 1
	for w in words.keys():
		word_dict[w] = i
		i += 1
	for l in list_form:
		while len(l) < context_num -1:
			l.append(0)
	return np.matrix(list_form), word_dict


# perform all of LSA 
# use lsa_matrix_dict to get the matrix
# then apply SVD on the matrix to make the model
# recommender system comes later when we get a new input phrase
# or not, we just get a new input phrase now? but we want to store this model i think
# and then increment it

# THE THING WE CARE ABOUT REDUCING IS V! WE REDUCE ITS DIMENSION, THIS IS THE MEANING MATRIX
# EACH COLUMN IS A DOCUMENT WITH A PARTICULAR SENSE ATTACHED TO IT 
# WE FOLD INTO SEMANTIC SPACE BY MULTIPLYING BY U* AND D^-1. 
# THIS IS WHAT WE DO COSINE SIMILARITY ON 

# THEN, THIS IS WHAT WE DO: FOR EACH POSSIBLE SENSE OF THE WORD (DERIVED FROM TRAINING), 
# WE FOLD THE REPRESENTATION INTO THIS SEMANTIC SPACE (AS DESCRIBED ABOVE). 
# WE FOLD THE REPRESENTATION OF THE WORD WE'RE TRYING TO DISAMBIGUATE INTO SEMANTIC SPACE
# TOO, AND COMPARE WITH ALL THE OTHER SENSE-SEMANTIC VECTORS TO SEE WHICH ONE IS CLOSEST IN MEANING.

# this is key: 
# *********** TRAINING DATA MUST CONTAIN DOCUMENTS THAT CORRESPOND TO CERTAIN SEMANTIC SENSES. ************


# 
def LSA(context_list):
	M, wd = lsa_matrix_dict(context_list)
	U, s, V = svd(M)
	# we want to do some sort of dimension reduction here
	# chop off the smaller eigenvalues and put the matrix back together
	# then to check what word sense project the 
	D = np.diag(s)
	print "U:"
	print U
	print "D:"
	print D
	print "V:"
	print V







# cs = context sentence
# ambig = ambiguous word
# returns the row vector which is most similar
# to the input using cosine similarity ? 

#def lsa_wsd(cs, ambig):
'''


