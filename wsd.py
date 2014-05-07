import dataparser
import bigram_model

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

training_dict = dataparser.parse_training_data('data/eng-lex-sample.training.xml')
(forward_bigrams, back_bigrams) = bigram_model.build_bigram_models(training_dict)
test_dict = dataparser.parse_test_data('data/eng-lex-sample.evaluation.xml')

(forward_bigrams_test, back_bigrams_test) = bigram_model.pos_bigram_test_data(test_dict)
predicted = []
for word in back_bigrams_test:
    for bigram in back_bigrams_test[word]:
        back_prediction = bigram_model.predict_sensekey(back_bigrams, word, bigram)
        if back_prediction:
            predicted.append(back_bigrams_test[word][bigram])
            #print word, back_bigrams_test[word][bigram], back_prediction[0], back_prediction[1]
            print word, back_bigrams_test[word][bigram], back_prediction[0], 'back'

for word in forward_bigrams_test:
    for bigram in forward_bigrams_test[word]:
        forward_prediction = bigram_model.predict_sensekey(forward_bigrams, word, bigram)
        if forward_prediction and forward_bigrams_test[word][bigram] not in predicted:
            predicted.append(forward_bigrams_test[word][bigram])
            # print word, forward_bigrams_test[word][bigram], forward_prediction[0], forward_prediction[1]       
            print word, forward_bigrams_test[word][bigram], forward_prediction[0], 'forward'
