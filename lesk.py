import string

from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from itertools import chain

import nltk
nltk.data.path.append('./nltk_data/')

porter = PorterStemmer()

'''
#TODO: various tokenizers.
from nltk import word_tokenize
def tokenize(sentence, option="split"):
  if option == "split": # Simply splits by whitespaces.
    return sentence.split()
  if option == "word_tokenize": # Uses NLTK word_tokenize().
    return word_tokenize(sentence)
'''

'''
#TODO: various stem / lemmatizers.
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
def stem(word, option="wnlemma")
  if option == "wnlemma":
    return wnl.lemmatize(word)
  if option == "porter":
    return porter.stem(word)
'''

def compare_overlaps_greedy(context, synsets_signatures):
  """
  Calculate overlaps between the context sentence and the synset_signature
  and returns the synset with the highest overlap.
  
  Note: Greedy algorithm only keeps the best sense, see http://goo.gl/OWSfOZ
  Keeping greedy algorithm for documentary sake because original_lesk is greedy.
  """
  max_overlaps = 0; lesk_sense = None
  for ss in synsets_signatures:
    overlaps = set(synsets_signatures[ss]).intersection(context)
    if len(overlaps) > max_overlaps:
      lesk_sense = ss
      max_overlaps = len(overlaps)    
  return lesk_sense

def compare_overlaps(context, synsets_signatures, \
                     nbest=False, keepscore=False, normalizescore=False):
  """ 
  Calculates overlaps between the context sentence and the synset_signture
  and returns a ranked list of synsets from highest overlap to lowest.
  """
  overlaplen_synsets = [] # a tuple of (len(overlap), synset).
  for ss in synsets_signatures:
    overlaps = set(synsets_signatures[ss]).intersection(context)
    overlaplen_synsets.append((len(overlaps), ss))

  # Rank synsets from highest to lowest overlap.
  ranked_synsets = sorted(overlaplen_synsets, reverse=True)
  
  # Normalize scores such that it's between 0 to 1. 
  if normalizescore:
    total = float(sum(i[0] for i in ranked_synsets))
    ranked_synsets = [(i/total,j) for i,j in ranked_synsets]
    
  if not keepscore: # Returns a list of ranked synsets without scores
    ranked_synsets = [i[1] for i in sorted(overlaplen_synsets, reverse=True)]
    
  if nbest: # Returns a ranked list of synsets.
    return ranked_synsets
  else: # Returns only the best sense.
    return ranked_synsets[0]

def original_lesk(context_sentence, ambiguous_word, dictionary=None):
  """
  This function is the implementation of the original Lesk algorithm (1986).
  It requires a dictionary which contains the definition of the different
  sense of each word. See http://goo.gl/8TB15w
  """
  if not dictionary:
    dictionary = {ss:ss.definition.split() for ss in wn.synsets(ambiguous_word)}
  best_sense = compare_overlaps_greedy(context_sentence.split(), dictionary)
  return best_sense    

def simple_signature(ambiguous_word, pos=None, stem=True, \
                     hyperhypo=True, stop=True):
  """ 
  Returns a synsets_signatures dictionary that includes signature words of a 
  sense from its:
  (i)   definition
  (ii)  example sentences
  (iii) hypernyms and hyponyms
  """
  synsets_signatures = {}
  for ss in wn.synsets(ambiguous_word):
    # If POS is specified.
    if pos and str(ss.pos()) is not pos:
      continue
    signature = []
    # Includes definition.
    signature+= ss.definition.split()
    # Includes examples
    # signature+= list(chain(*[i.split() for i in ss.examples()]))
    # Includes lemma_names.
    signature = signature + ss.lemma_names
    # Optional: includes lemma_names of hypernyms and hyponyms.
    # if hyperhypo == True:
      # signature+= list(chain(*[i.lemma_names for i in ss.hypernyms()+ss.hyponyms()]))
    # Optional: removes stopwords.
    if stop == True:
      signature = [i for i in signature if i not in stopwords.words('english')]    
    # Matching exact words causes sparsity, so optional matching for stems.
    if stem == True: 
      signature = [porter.stem(i) for i in signature]
    synsets_signatures[ss] = signature
  return synsets_signatures

def simple_lesk(context_sentence, ambiguous_word, \
                pos=None, stem=True, hyperhypo=True, \
                nbest=False, keepscore=False, normalizescore=False):
  """
  Simple Lesk is somewhere in between using more than the 
  original Lesk algorithm (1986) and using less signature 
  words than adapted Lesk (Banerjee and Pederson, 2002)
  """
  # Get the signatures for each synset.
  ss_sign = simple_signature(ambiguous_word, pos, stem, hyperhypo)
  # Disambiguate the sense in context.
  context_sentence = [porter.stem(i) for i in context_sentence.split()]
  best_sense = compare_overlaps(context_sentence, ss_sign, \
                                nbest=nbest, keepscore=keepscore, \
                                normalizescore=normalizescore)  
  print best_sense.definition;
  return best_sense

def adapted_lesk(context_sentence, ambiguous_word, \
                pos=None, stem=True, hyperhypo=True, stop=True, \
                nbest=False, keepscore=False, normalizescore=False):

  # Get the signatures for each synset.
  ss_sign = simple_signature(ambiguous_word, pos, stem, hyperhypo)
  for ss in ss_sign:
    related_senses = list(set(ss.member_holonyms() + ss.member_meronyms() + 
                             ss.part_meronyms() + ss.part_holonyms() + 
                             ss.similar_tos() + ss.substance_holonyms() + 
                             ss.substance_meronyms()))
    
    signature = list([j for j in chain(*[i.lemma_names for i in \
                      related_senses]) if j not in stopwords.words('english')])    
    # Matching exact words causes sparsity, so optional matching for stems.
    if stem == True: 
      signature = [porter.stem(i) for i in signature]
    ss_sign[ss]+=signature
  
  # Disambiguate the sense in context.
  context_sentence = [porter.stem(i) for i in context_sentence.split()]
  best_sense = compare_overlaps(context_sentence, ss_sign, \
                                nbest=nbest, keepscore=keepscore, \
                                normalizescore=normalizescore)
  return best_sense

def cosine_lesk(context_sentence, ambiguous_word, stem=False, stop=True, \
                nbest=False):
  """ 
  In line with vector space models, we can use cosine to calculate overlaps
  instead of using raw overlap counts. Essentially, the idea of using 
  signatures (aka 'sense paraphrases') is lesk-like.
  """
  from nltk.tokenize import word_tokenize
  from cosine import cosine_similarity as cos_sim
  synsets_signatures = simple_signature(ambiguous_word, stem=stem, stop=stop)
  
  scores = []
  for ss, signature in synsets_signatures.items():
    # Lowercase and replace "_" with spaces.
    signature = " ".join(map(str, signature)).lower().replace("_", " ")
    # Removes punctuation.
    signature = [i for i in word_tokenize(signature) \
                 if i not in string.punctuation]
    # Optional: remove stopwords.
    if stop:
      signature = [i for i in signature if i not in stopwords.words('english')]
    # Optional: stem the tokens.
    if stem:
      signature = [porter.stem(i) for i in signature]
    scores.append((cos_sim(context_sentence, " ".join(signature)), ss))

  if not nbest:
    return sorted(scores, reverse=True)[0][1]
  else:
    return [(j,i) for i,j in sorted(scores, reverse=True)]