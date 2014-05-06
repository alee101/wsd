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

def lsa_wsd(cs, ambig)

