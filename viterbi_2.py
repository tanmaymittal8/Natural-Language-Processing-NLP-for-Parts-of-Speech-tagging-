import math
from collections import defaultdict, Counter
from math import log

# Note: remember to use these two elements when you find a probability is 0 in the training data.
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5 

def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    :param sentences:
    :return: intitial tag probs, emission words given tag probs, transition of tags to tags probs
    """
    init_prob = defaultdict(lambda: 0)
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0))
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0))
    
    # 1. Count word frequencies to identify Hapax Legomena
    word_counts = defaultdict(int)
    for sentence in sentences:
        for word, tag in sentence[1:-1]: # Skip START and END
            word_counts[word] += 1
            
    # 2. Track Hapax tags
    hapax_tag_counts = defaultdict(int)
    total_hapax = 0
    
    # 3. Build counts for transitions and emissions
    unique_tags = {} # {tag: {word: count}}
    unique_tag_sequence = {} # {prev_tag: {curr_tag: count}}
    
    init_prob['START'] = 1
    
    for sentence in sentences:
        previous_word, previous_tag = sentence[0]
        for word, tag in sentence[1:]:
            # Emission counts
            if tag not in unique_tags:
                unique_tags[tag] = {}
            if word in unique_tags[tag]:
                unique_tags[tag][word] += 1
            else:
                unique_tags[tag][word] = 1

            # Hapax counts
            if word_counts[word] == 1:
                hapax_tag_counts[tag] += 1
                total_hapax += 1

            # Transition counts
            if previous_tag not in unique_tag_sequence:
                unique_tag_sequence[previous_tag] = {}
            if tag in unique_tag_sequence[previous_tag]:
                unique_tag_sequence[previous_tag][tag] += 1
            else:
                unique_tag_sequence[previous_tag][tag] = 1

            previous_word = word
            previous_tag = tag

    # 4. Calculate Transition Probabilities (Same as Viterbi 1)
    for prev_tag in unique_tag_sequence:
        for curr_tag in unique_tag_sequence[prev_tag]:
            numerator = unique_tag_sequence[prev_tag][curr_tag] + epsilon_for_pt
            denominator = sum(unique_tag_sequence[prev_tag].values()) + epsilon_for_pt * (len(unique_tag_sequence[prev_tag]) + 1)
            trans_prob[prev_tag][curr_tag] = numerator / denominator

    # 5. Calculate Emission Probabilities (Using Hapax Smoothing)
    for e_tag in unique_tags:
        # P(Tag | Hapax)
        # We add a tiny smoothing (+1 / +len) here to ensure no tag has 0 hapax probability
        hapax_prob = (hapax_tag_counts[e_tag] + 1) / (total_hapax + len(unique_tags))
        
        # Scale the epsilon based on Hapax probability
        scaled_epsilon = emit_epsilon * hapax_prob
        
        total_tokens = sum(unique_tags[e_tag].values())
        vocab_size = len(unique_tags[e_tag])
        
        # Denominator for Laplace smoothing
        denominator = total_tokens + scaled_epsilon * (vocab_size + 1)

        for e_word in unique_tags[e_tag]:
            emit_prob[e_tag][e_word] = (unique_tags[e_tag][e_word] + scaled_epsilon) / denominator
        
        # Set the probability for unseen words (NEWWORD)
        emit_prob[e_tag]['NEWWORD'] = scaled_epsilon / denominator
    
    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    """
    log_prob = {}
    predict_tag_seq = {}
    
    if i == 0:
        for tag in emit_prob:
            if tag == 'START':
                log_prob[tag] = 0 # log(1)
            else:
                log_prob[tag] = log(epsilon_for_pt) # Very small prob
            predict_tag_seq[tag] = []
            predict_tag_seq[tag].append(tag)
    else:
        for i_tag in emit_prob:
            max_prob = -float('inf')
            optimal_prev_tag = None
            
            for last_i_tag in prev_prob:
                # Transition Prob
                if i_tag in trans_prob[last_i_tag]:
                    transition = trans_prob[last_i_tag][i_tag]
                else:
                    transition = epsilon_for_pt
                
                # Emission Prob
                if word in emit_prob[i_tag]:
                    emission = emit_prob[i_tag][word]
                else: 
                    emission = emit_prob[i_tag]['NEWWORD']
                
                # Calculate Log Prob
                if transition > 0 and emission > 0:
                    current_prob = prev_prob[last_i_tag] + log(transition) + log(emission)
                else:
                    current_prob = -float('inf')

                if current_prob > max_prob:
                    max_prob = current_prob
                    optimal_prev_tag = last_i_tag

            log_prob[i_tag] = max_prob
            if optimal_prev_tag:
                predict_tag_seq[i_tag] = prev_predict_tag_seq[optimal_prev_tag] + [optimal_prev_tag]
            else:
                predict_tag_seq[i_tag] = []
                
    return log_prob, predict_tag_seq

def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags
    '''
    init_prob, emit_prob, trans_prob = training(train)
    
    predicts = []
    
    for sen in range(len(test)):
        sentence = test[sen]
        length = len(sentence)
        log_prob = {}
        predict_tag_seq = {}
        
        for i in range(length):
            log_prob, predict_tag_seq = viterbi_stepforward(i, sentence[i], log_prob, predict_tag_seq, emit_prob, trans_prob)
            
        last_tag = max(log_prob, key=log_prob.get)
        correct_tags = predict_tag_seq[last_tag] 
        correct_tags.append(last_tag)
        predicts.append(list(zip(sentence, correct_tags[1:])))
        
    return predicts