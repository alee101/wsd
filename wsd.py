import dataparser
import bigram_model

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

training_dict = dataparser.parse_training_data('data/eng-lex-sample.training.xml')
(forward_bigrams, back_bigrams) = bigram_model.build_bigram_models(training_dict)
test_dict = dataparser.parse_test_data('data/eng-lex-sample.evaluation.xml')

# import pprint
# pprint.pprint(dict(back_bigrams))
# print('------------------------------------------------------------')
# pprint.pprint(dict(forward_bigrams))
