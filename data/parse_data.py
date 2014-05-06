import xml.etree.ElementTree as ET
from collections import defaultdict
import re

class TrainingInstance:
    def __init__(self, instance_id, context):
        self.iid = instance_id
        self.context = context
    def sentence_context(self):
        sentence = [self.context[0].split('.')[-1], self.context[1], self.context[2]]
        sentence = map(lambda s: re.sub(r'\[.*?\]|\n', '', s), sentence)
        # sentence = map(lambda s: re.sub(r'\[.*?\]', '', s), sentence)
        # return map(lambda s: s.strip(), sentence)
        return map(lambda s: s.strip(' '), sentence)

wsds = defaultdict(lambda: defaultdict(list))

# Ignore unknown XML entities
parser = ET.XMLParser()
parser.parser.UseForeignDTD(True)
parser.entity = defaultdict(str)

tree = ET.parse('eng-lex-sample.training.xml', parser=parser)
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

for word in wsds:
    print '\n***WORD: ', word, '***'
    for senseid in wsds[word]:
        print '\n**SENSEID: ', senseid, '**'
        for instance in wsds[word][senseid]:
            print instance.iid
            print instance.sentence_context()
