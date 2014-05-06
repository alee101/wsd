import xml.etree.ElementTree as ET
from collections import defaultdict
import re
from nltk.corpus import wordnet as wn
import nltk

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# A training instance contains the instance id and the corresponding context
# in which it appeared, where the context is a three element list containing
# the text preceding the word to be disambiguated, the word, and the text 
# following the word.
class TrainingInstance:
    def __init__(self, instanceid, context):
        self.iid = instanceid
        self.context = context
    def sentence_context(self):
        # Return the sentence in which the word being disambiguated appears
        sentence = [self.context[0].split('.')[-1], self.context[1], self.context[2]]
        # Strip tags, newlines, leading and trailing whitespace
        sentence = map(lambda s: re.sub(r'\[.*?\]|\n', '', s), sentence)
        return map(lambda s: s.strip(' '), sentence)


# Return a dictionary whose key is a word being disambiguated and value is 
# another dictionary whose key is a sense key and value is a list of TrainingInstances.
def parse_data(f_data):
    wsds = defaultdict(lambda: defaultdict(list))

    # Ignore unknown XML entities
    parser = ET.XMLParser()
    parser.parser.UseForeignDTD(True)
    parser.entity = defaultdict(str)

    tree = ET.parse(f_data, parser=parser)
    root = tree.getroot()
    for word in root.findall('lexelt'):
        instances = word.findall('instance')
        for instance in instances:
            answers = instance.findall('answer')
            senseids = [answer.attrib['senseid'] for answer in answers]
            instanceid = [answer.attrib['instance'] for answer in answers][0]
            word = instanceid.split('.')[0]
            context = list(instance.find('context').itertext())
            for senseid in senseids:
                wsds[word][senseid].append(TrainingInstance(instanceid, context))
    return wsds

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

wsds = parse_data('data/eng-lex-sample.training.xml')
for word in wsds:
    for sense_key in wsds[word]:
        synset = get_synset(sense_key)
