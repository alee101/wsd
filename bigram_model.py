from collections import defaultdict
import json
import heapq

from nltk.corpus import wordnet as wn
import nltk

# Return the predicted WordNet sense key of the given word using the given bigram model and bigram
# if the frequency count and probability thresholds are met. Return None otherwise.
def predict_sensekey(bigram_model, word, bigram, count_threshold=5, prob_threshold=0.8):
    if bigram in bigram_model[word]:
        sensekey_dict = bigram_model[word][bigram]
        if (sensekey_dict['count'] > count_threshold and 
            heapq.nlargest(3, sensekey_dict.itervalues())[1] > prob_threshold):
            #return bigram_model[word][bigram]
            return heapq.nlargest(3, sensekey_dict, key=sensekey_dict.__getitem__)[1]
    return None

# Return forward and backward bigram test data, returning a cached copy if it exists. 
# Set clear_cache to True if test_dict is built on a different dataset since last use.
def pos_bigram_test_data(test_dict, clear_cache=False):
    if clear_cache:
        (forward_bigrams_test, back_bigrams_test) = build_pos_test_bigrams(test_dict)
    else:
        try:
            (forward_bigrams_test, back_bigrams_test) = (json.load(open('forward_bigrams_test')), json.load(open('back_bigrams_test')))
        except:
            (forward_bigrams_test, back_bigrams_test) = build_pos_test_bigrams(test_dict)

    return (forward_bigrams_test, back_bigrams_test)

# Return forward and backward bigram test data based on test_dict. Each bigram data model returned
# is a dictionary where the key is the word being disambiguated and the value another dictionary
# where the key is the POS bigram and the value the unique instance id.
def build_pos_test_bigrams(test_dict):
    forward_bigrams_test = defaultdict(lambda: defaultdict(str))
    back_bigrams_test = defaultdict(lambda: defaultdict(str))

    for word in test_dict:
        for instance in test_dict[word]:
            (pre, w, post) = instance.sentence_context_list()
            pre_count = pre.count(w)
            match_count = 0
            sentence = instance.sentence_context()
            tags = get_pos(sentence)
            for i, tag in enumerate(tags):
                if tag[0].find(w) > -1:
                    if match_count == pre_count:
                        # keys need to be str(tuple) for lookup from json serialization
                        forward_bigram = str(tuple([tag[1] for tag in tags[i-1:i+1]]))
                        back_bigram = str(tuple([tag[1] for tag in tags[i:i+2]]))
                        if back_bigram:
                            back_bigrams_test[word][back_bigram] = instance.iid
                        if forward_bigram:
                            forward_bigrams_test[word][forward_bigram] = instance.iid
                        break
                    elif match_count < pre_count:
                        match_count += 1
    # Cache bigram test data
    json.dump(forward_bigrams_test, open('forward_bigrams_test', 'w'))
    json.dump(back_bigrams_test, open('back_bigrams_test', 'w'))

    return (forward_bigrams_test, back_bigrams_test) 

# Return forward and backward bigram models
def build_bigram_models(training_dict):
    (forward_bigrams_data, back_bigrams_data) = pos_bigram_data(training_dict)
    (forward_bigrams, back_bigrams) = (bigram_model(forward_bigrams_data), bigram_model(back_bigrams_data))

    return (forward_bigrams, back_bigrams)

# Return forward and backward bigram data, returning a cached copy if it exists. 
# Set clear_cache to True if training_dict is built on a different dataset since last use.
def pos_bigram_data(training_dict, clear_cache=False):
    if clear_cache:
        (forward_bigrams_data, back_bigrams_data) = build_pos_bigrams(training_dict)
    else:
        try:
            (forward_bigrams_data, back_bigrams_data) = (json.load(open('forwardbigrams')), json.load(open('backbigrams')))
        except:
            (forward_bigrams_data, back_bigrams_data) = build_pos_bigrams(training_dict)  

    return (forward_bigrams_data, back_bigrams_data)

# Return a bigram model for the given bigram data
def bigram_model(bigram_data):
    for word in bigram_data:
        for pos in bigram_data[word]:
            count = sum(bigram_data[word][pos].itervalues())
            bigram_data[word][pos] = {k: v*1.0/count for (k, v) in bigram_data[word][pos].iteritems()}
            bigram_data[word][pos]['count'] = count

    return bigram_data

# Return forward and backward bigram data based on training_dict.
def build_pos_bigrams(training_dict):
    forward_bigrams_data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    back_bigrams_data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for word in training_dict:
        for sense_key in training_dict[word]:
            for instance in training_dict[word][sense_key]:
                (pre, w, post) = instance.sentence_context_list()
                pre_count = pre.count(w)
                match_count = 0
                sentence = instance.sentence_context()
                tags = get_pos(sentence)
                for i, tag in enumerate(tags):
                    if tag[0].find(w) > -1:
                        if match_count == pre_count:
                            # str(tuple) for json serialization
                            forward_bigram = str(tuple([tag[1] for tag in tags[i-1:i+1]]))
                            back_bigram = str(tuple([tag[1] for tag in tags[i:i+2]]))
                            # if sense_key != 'P' and sense_key != 'U':
                            if back_bigram:
                                back_bigrams_data[word][back_bigram][sense_key] += 1
                            if forward_bigram:
                                forward_bigrams_data[word][forward_bigram][sense_key] += 1
                            break
                        elif match_count < pre_count:
                            match_count += 1

    # Cache bigram data
    # json.dump(forward_bigrams_data, open('forwardbigrams', 'w'))
    # json.dump(back_bigrams_data, open('backbigrams', 'w'))
    json.dump(forward_bigrams_data, open('forward_bigrams', 'w'))
    json.dump(back_bigrams_data, open('back_bigrams', 'w'))

    return (forward_bigrams_data, back_bigrams_data)

# Return the synset of a word given its sense key
def get_synset(sense_key):
    synset = None
    try:
        synset = wn.lemma_from_key(sense_key).synset
    except:
        print 'Could not find synset for sense key ' + sense_key
    return synset

# Return part of speech tags for given sentence
def get_pos(sent):
    tokens = nltk.word_tokenize(sent)
    tag_tuples = nltk.pos_tag(tokens)
    return tag_tuples
