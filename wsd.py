import dataparser
import bigram_model
import wsd_lsa as wl

# from collections import defaultdict
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
        instance_id = back_bigrams_test[word][bigram]
        if back_prediction:
            predicted.append(instance_id)
            print word, instance_id, back_prediction[0], 'back'

for word in forward_bigrams_test:
    for bigram in forward_bigrams_test[word]:
        forward_prediction = bigram_model.predict_sensekey(forward_bigrams, word, bigram)
        instance_id = forward_bigrams_test[word][bigram]
        if forward_prediction and instance_id not in predicted:
            predicted.append(instance_id)
            print word, instance_id, forward_prediction[0], 'forward'

for word in training_dict:
    if word not in predicted:
        lsa_training_data = wl.make_training_data(training_dict, word)
        trained_model = wl.train_model_old(lsa_training_data)
        for instance in test_dict[word]:
        	prediction = wl.guess_word_sense_old(trained_model, word, instance.paragraph_context())
        	print word, instance.iid, prediction, 'lsa'  
