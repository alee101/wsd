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

# transforms the corpus into multiple-word representation (MWR)
# c is a .txt file containing corpus
def transform_corpus(c):
	transformed = []
	non_word_tokens = set([".", ",", ";", "'", """, "~", "`", "+", "-"])
	with open(c, 'r') as f_in:
		for line in f_in:
			# need hashtable and shit
