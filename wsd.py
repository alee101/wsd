import dataparser
import bigram_model

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

training_dict = dataparser.parse_training_data('data/eng-lex-sample.training.xml')
(forward_bigrams, back_bigrams) = bigram_model.build_bigram_models(training_dict)
test_dict = dataparser.parse_test_data('data/eng-lex-sample.evaluation.xml')

(forward_bigrams_test, back_bigrams_test) = bigram_model.pos_bigram_test_data(test_dict)
for word in back_bigrams_test:
    for bigram in back_bigrams_test[word]:
        prediction = bigram_model.predict_sensekey(back_bigrams, word, bigram)
        if prediction:
            print word, back_bigrams_test[word][bigram], prediction

# import pprint
# pprint.pprint(dict(back_bigrams))
# print('------------------------------------------------------------')
# pprint.pprint(dict(forward_bigrams))
