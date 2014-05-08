import json
import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm 
from supervised_lsa import nice_word

try:
    U = np.matrix(json.load(open('U_')))
    D = np.matrix(json.load(open('D_')))
    V = np.matrix(json.load(open('V_')))
    wd = json.load(open('wd'))
except:
    print "please first run supervised_lsa.py to create the U_, D_, V_ and wd files."

# projects the document into the space defined by the pca
# document is given as a string
def project(doc_str):
    # need to build the column vector as though
    # doc is a new column component in the term-document matrix 
    doc = doc_str.split(" ")
    doc_vec = np.zeros(len(U)).T
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
    # by multiplying the column vector by U * D^(-1)
    # i.e.: new_doc' = new_doc * U * D^(-1)
    new_doc_vec = np.dot(doc_vec, np.dot(U, inv(D)))
    return new_doc_vec

# use cosine similarity to check doc_vec (a document column vector)
# against all the other topic vectors in the reduced space
# return the eigentopic which is closest it
def most_sim_topic(doc_vec):
    doc_vec = np.array(doc_vec)
    cosines = map(lambda v: np.vdot(np.array(v), doc_vec)/(norm(np.array(v)) * norm(doc_vec)) , V)
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
def train_model_old(train_data):
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
    # yes, because otherwise the presence of a 1.0 will dominate along columns.

    
    #for w in word_sense_dict.keys():
    #    for sense in word_sense_dict[w].keys():
    #        # note if it's not full length (i.e. all 9 columns), the rest will just be zero
    #        summed = sum(word_sense_dict[w][sense]) + 0.0
    #        for i in range(0, len(word_sense_dict[w][sense])):
    #            word_sense_dict[w][sense][i] /= summed
    
    return word_sense_dict


# given an ambiguous word and a context
# which it is in, return the word sense from wordnet
# word is just a string
# context is a string (i.e. a doc_str)
def guess_word_sense_old(trained_model, word, context):
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

# analagous to most_sim_topic:
# instaed of returning a single topic, we return a vector 
# of probabilities that it is each topic
# -1 is really bad, 1 is really close to a given topic eigenvector
# we leave negative signs in here because we want to represent opposed-ness of the semantic vectors
def prob_topic_vec(doc_vec):
    doc_vec = np.array(doc_vec)
    cosines = map(lambda v: np.vdot(np.array(v), doc_vec)/(norm(np.array(v)) * norm(doc_vec)) , V)
    return np.array(cosines)



# uses topic eigenvectors for each word-sense instead of counts for each topic eigenvector
# basically average over examples to get general topic eigenvector for each word-sense
# then when you're testing, you do a cosine comparison with the avg. topic eigenvector for each sense
def train_model(train_data):
    word_sense_dict = dict()
    for (doc_str, w, sense) in train_data:
        topic_vec = prob_topic_vec(project(doc_str))
        if w not in word_sense_dict:
            word_sense_dict[w] = dict()
        if sense not in word_sense_dict[w]:
            word_sense_dict[w][sense] = (topic_vec, 1)
        else:
            word_sense_dict[w][sense] = (np.add(topic_vec, word_sense_dict[w][sense][0]), word_sense_dict[w][sense][1] + 1)
    # average
    for w in word_sense_dict.keys():
        for sense in word_sense_dict[w].keys():
            word_sense_dict[w][sense] = word_sense_dict[w][sense][0]/ (word_sense_dict[w][sense][1] + 0.0)
    return word_sense_dict

# some test train_data
td1 = [("i like chasing rivers and running alongside banks", "bank", "wn_bank_1"), ("i like going to the bank and getting money", "bank", "wn_bank_2"), ("i am scared of bats in caves", "bat", "wn_bat_1"), ("i play baseball with my bat", "bat", "wn_bat_2")]

td2 = [("i like chasing rivers and running alongside banks", "bank", "wn_bank_1"), ("animals are cute and so is running alongside banks", "bank", "wn_bank_1"), ("the river bank is an interesting place because of the animals and plants", "bank", "wn_bank_1"), ("i like going to the bank and getting money", "bank", "wn_bank_2"), ("i am scared of bats in caves", "bat", "wn_bat_1"), ("i play baseball with my bat", "bat", "wn_bat_2")]


# takes labled data from somewhere 
# and turns it into format that train_model uses
# (i.e. a list of (doc_str, w, sense))
# ideally, doc_str contains a bunch of sentences spaced NORMALLY. 
# WE WANT JUST INDIVIDUAL WORDS (punctuation allowed to be tagged on)
# when we .split(" ") the sentence. 
def make_training_data(training_dict, word):
    training_data = []
    #for word in training_dict:
    for sense_key in training_dict[word]:
        for instance in training_dict[word][sense_key]:
            paragraph = instance.paragraph_context()
            training_data.append((paragraph, word, sense_key))
    return training_data


# for given word and context, calculate topic vector for context. 
# then for given word, compare topic vector to each sense with cosine similarity 
# return the sense that has best cosine similarity. 
def guess_word_sense(trained_model, word, context):
    word = nice_word(word)
    topic_vec = prob_topic_vec(project(context))
    max_sense = "nullsense"
    max_cosine = -2 # outside min domain of cosine
    def cos_sim(v1, v2):
        return np.vdot(v1, v2)/(norm(v1) * norm(v2))
    for sense in trained_model[word].keys():
        sense_vec = trained_model[word][sense]
        curr_cos_sim = cos_sim(topic_vec, sense_vec)
        # print sense
        # print sense_vec
        # print curr_cos_sim
        if curr_cos_sim > max_cosine:
            max_sense = sense
            max_cosine = curr_cos_sim
    return max_sense

