from collections import defaultdict

answers = defaultdict(lambda: defaultdict(list))
with open('key.sorted') as f_results:
    for line in f_results:
        fields = line.strip().split()
        answers[fields[0]][fields[1]] = fields[2:]

with open('out') as f_predictions:
    for line in f_predictions:
        fields = line.strip().split()
        (word, iid, sensekey) = fields[:3]
        metadata = fields[3:]
        if sensekey in answers[word][iid]:
            print 'Yes', metadata, sensekey
        else:
            print 'No', metadata, sensekey, 'vs.', answers[word][iid]

