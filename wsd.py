import dataparser
import bigram_model
import supervised_lsa

# from collections import defaultdict
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

training_dict = dataparser.parse_training_data('data/eng-lex-sample.training.xml')
(forward_bigrams, back_bigrams) = bigram_model.build_bigram_models(training_dict)
test_dict = dataparser.parse_test_data('data/eng-lex-sample.evaluation.xml')
(forward_bigrams_test, back_bigrams_test) = bigram_model.pos_bigram_test_data(test_dict)

# Construct a copy for easier removal of predictions
# test_instances = defaultdict(lambda: defaultdict(list))
# for word in test_dict:
#     for instance in test_dict[word]:
#         test_instances[word][instance.iid] = instance

# predicted = []
# for word in back_bigrams_test:
#     for bigram in back_bigrams_test[word]:
#         back_prediction = bigram_model.predict_sensekey(back_bigrams, word, bigram)
#         instance_id = back_bigrams_test[word][bigram]
#         if back_prediction:
#             predicted.append(instance_id)
#             #del test_instances[word][instance_id]
#             #print word, back_bigrams_test[word][bigram], back_prediction[0], back_prediction[1]
#             print word, instance_id, back_prediction[0], 'back'

# for word in forward_bigrams_test:
#     for bigram in forward_bigrams_test[word]:
#         forward_prediction = bigram_model.predict_sensekey(forward_bigrams, word, bigram)
#         instance_id = forward_bigrams_test[word][bigram]
#         if forward_prediction and instance_id not in predicted:
#             predicted.append(instance_id)
#             #del test_instances[word][instance_id]
#             # print word, forward_bigrams_test[word][bigram], forward_prediction[0], forward_prediction[1]       
#             print word, instance_id, forward_prediction[0], 'forward'

#for word in training_dict:
word = 'fine'
lsa_training_data = supervised_lsa.make_training_data(training_dict, word)
# print lsa_training_data
trained_model = supervised_lsa.train_model(lsa_training_data)
print trained_model
#trained_model = supervised_lsa.train_model(supervised_lsa.make_training_data(training_dict))
#print trained_model
# count = 0
# for word in test_dict:
#     for instance in test_dict[word]:
#         if count < 10:
#             prediction = supervised_lsa.guess_word_sense(trained_model, word, instance.paragraph_context())
#             print word, instance_id, prediction, 'lsa'
#         else:
#             break
for instance in test_dict[word]:
    prediction = supervised_lsa.guess_word_sense(trained_model, word, instance.paragraph_context())
    print word, instance.iid, prediction, 'lsa'  