"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""

import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5   # exact setting seems to have little or no effect


def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0) # {init tag: #}
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag: {word: # }}
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0)) # {tag0:{tag1: # }}
    
    # list of sentences
    unique_tags = {}
    unique_tag_sequence = {}
    init_prob['START'] = 1
    
    for sentence in sentences:
        previous_word, previous_tag = sentence[0]
        for word, tag in sentence[1:]:
            if tag not in unique_tags:
                unique_tags[tag] = {}
            if word in unique_tags[tag]:
                unique_tags[tag][word] += 1
            else:
                unique_tags[tag][word] = 1

            if previous_tag not in unique_tag_sequence:
                unique_tag_sequence[previous_tag] = {}
            if tag in unique_tag_sequence[previous_tag]:
                unique_tag_sequence[previous_tag][tag] += 1
            else:
                unique_tag_sequence[previous_tag][tag] = 1

            previous_word = word
            previous_tag = tag


    for prev_tag in unique_tag_sequence:
        for curr_tag in unique_tag_sequence[prev_tag]:
            trans_prob[prev_tag][curr_tag] = (unique_tag_sequence[prev_tag][curr_tag] + epsilon_for_pt) / (sum(unique_tag_sequence[prev_tag].values()) + epsilon_for_pt*(len(unique_tag_sequence[prev_tag]) + 1))


    for e_tag in unique_tags:
        for e_word in unique_tags[e_tag]:
            emit_prob[e_tag][e_word] = (unique_tags[e_tag][e_word] + emit_epsilon )/ (sum(unique_tags[e_tag].values()) + emit_epsilon* (len(unique_tags[e_tag]) + 1))
        emit_prob[e_tag]['NEWWORD'] = emit_epsilon / (sum(unique_tags[e_tag].values()) + emit_epsilon* (len(unique_tags[e_tag]) + 1))

    
    # TODO: (I)
    # Input the training set, output the formatted probabilities according to data statistics.
    
    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    :param i: The i'th column of the lattice/MDP (0-indexing)
    :param word: The i'th observed word
    :param prev_prob: A dictionary of tags to probs representing the max probability of getting to each tag at in the
    previous column of the lattice
    :param prev_predict_tag_seq: A dictionary representing the predicted tag sequences leading up to the previous column
    of the lattice for each tag in the previous column
    :param emit_prob: Emission probabilities
    :param trans_prob: Transition probabilities
    :return: Current best log probs leading to the i'th column for each tag, and the respective predicted tag sequences
    """
    log_prob = {} # This should store the log_prob for all the tags at current column (i)
    predict_tag_seq = {} # This should store the tag sequence to reach each tag at column (i)
    prev_sequence_tags = []

    # TODO: (II)
    # implement one step of trellis computation at column (i)
    # You should pay attention to the i=0 special case.

    if i != 0:
        for i_tag in emit_prob:
            max_prob = -99999999999999 # really large negative number
            optimal_prev_tag = 'START'
            
            for last_i_tag in prev_prob:
                if i_tag in trans_prob[last_i_tag]:
                    transition = trans_prob[last_i_tag][i_tag] # prev_prob[last_i_tag] + 
                else:
                    transition =  emit_epsilon # prev_prob[last_i_tag] +
                
                if word in emit_prob[i_tag]:
                    emmision = emit_prob[i_tag][word]
                else: 
                    emmision = emit_prob[i_tag]['NEWWORD']

                i_prob = prev_prob[last_i_tag] + log(transition) + log(emmision)

                if max_prob < i_prob:
                    optimal_prev_tag = last_i_tag
                    max_prob = i_prob

            log_prob[i_tag] = max_prob
            predict_tag_seq[i_tag] = prev_predict_tag_seq[optimal_prev_tag] + [optimal_prev_tag]

    else:
        for tag in emit_prob:
            if tag == 'START':
                log_prob[tag] = 1
            else:
                log_prob[tag] = log(epsilon_for_pt)
            predict_tag_seq[tag] = []
            predict_tag_seq[tag].append(tag)

    # for logs in log_prob:
    #     print(logs)
    
        
    return log_prob, predict_tag_seq

def viterbi_1(train, test, get_probs=training):
    '''
    input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
            test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    init_prob, emit_prob, trans_prob = get_probs(train)
    
    predicts = []
    
    for sen in range(len(test)):
        sentence=test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        # init log prob
        for t in emit_prob:
            if t in init_prob:
                log_prob[t] = log(init_prob[t])
            else:
                log_prob[t] = log(epsilon_for_pt)
            predict_tag_seq[t] = []

        # forward steps to calculate log probs for sentence
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob,trans_prob)
            
        # TODO:(III) 
        # according to the storage of probabilities and sequences, get the final prediction.
        last_tag = max(log_prob, key = log_prob.get)
        correct_tags = []
        correct_tags = predict_tag_seq[last_tag] 
        correct_tags.append(last_tag)
        predicts.append(list(zip(sentence, correct_tags[1:])))


        
    return predicts