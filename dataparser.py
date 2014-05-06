import xml.etree.ElementTree as ET
from collections import defaultdict
import re

# A data instance contains the instance id and the corresponding context
# in which it appeared, where the context is a three element list containing
# the text preceding the word to be disambiguated, the word, and the text 
# following the word.
class DataInstance:
    def __init__(self, instanceid, context):
        self.iid = instanceid
        self.context = context
    def sentence_context_list(self):
        # Return the sentence (as a three element list) in which the word being disambiguated appears
        sentence = [self.context[0].split('.')[-1], self.context[1], self.context[2]]
        # Strip tags, newlines, leading and trailing whitespace
        sentence = map(lambda s: re.sub(r'\[.*?\]|\n', '', s), sentence)
        return map(lambda s: s.strip(' '), sentence)
    def sentence_context(self):
        # Return the sentence in which the word being disambiguated appears
        return ' '.join(' '.join(self.sentence_context_list()).split())

# XML Parser
class DataParser:
    def __init__(self):
        # Ignores unknown XML entities
        parser = ET.XMLParser()
        parser.parser.UseForeignDTD(True)
        parser.entity = defaultdict(str)
        self.parser = parser

# Return a dictionary whose key is a word being disambiguated and value is 
# another dictionary whose key is a sense key and value is a list of DataInstances.
def parse_training_data(f_data):
    training_dict = defaultdict(lambda: defaultdict(list))

    tree = ET.parse(f_data, parser=DataParser().parser)
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
                training_dict[word][senseid].append(DataInstance(instanceid, context))

    return training_dict

# Return a dictionary whose key is a word being disambiguated and value is a list of DataInstances.
def parse_test_data(f_data):
    test_dict = defaultdict(list)

    tree = ET.parse(f_data, parser=DataParser().parser)
    #tree = ET.parse('sample-test.xml', parser=parser)
    root = tree.getroot()
    for word in root.findall('lexelt'):
        print word.attrib
        instances = word.findall('instance')
        for instance in instances:
            instanceid = instance.attrib['id']
            word = instanceid.split('.')[0]
            print word
            context = list(instance.find('context').itertext())
            test_dict[word].append(DataInstance(instanceid, context))

    return test_dict
    