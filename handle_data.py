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
from nltk.corpus import brown  


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

def topic_modeler(context_list):
	words = dict()
	category_map = dict()
	cat_num = 1
	for category in brown.categories(): 
		category_map[category] = cat_num
		for w in brown.words(categories=category):
			w = w.lower() #lowercase everything
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
# Do SVD on M, do dimension reduction to dim, 
# return resulting matrices 
def pca(M, dim):
	U, s, Vt = svd(M, full_matrices=False)
	V = Vt.T
	# we want to chop of irrelevant part of s
	print len(U)
	print s
	print V
	D = np.diag(s)






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


