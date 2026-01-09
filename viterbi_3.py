import math
from collections import defaultdict, Counter
from math import log

# Tuning parameters
epsilon_for_pt = 1e-5
emit_epsilon = 1e-5

def get_word_pattern(word):
    """
    Classifies a word into a morphological pattern (pseudoword).
    Order matters: check more specific suffixes first.
    """
    # 1. Numbers
    if any(char.isdigit() for char in word):
        return "X-NUM"
    
    # 2. Hyphens
    if '-' in word:
        return "X-HYPHEN"
    
    # 3. Suffixes 
    # Length checks ensure we don't match short words (e.g. "sing" -> "X-ING")
    if word.endswith("ing") and len(word) > 4:
        return "X-ING"
    if word.endswith("tion") and len(word) > 5:
        return "X-TION"
    if word.endswith("er") and len(word) > 3:
        return "X-ER"
    if word.endswith("est") and len(word) > 4:
        return "X-EST"
    if word.endswith("ly") and len(word) > 3:
        return "X-LY"
    if word.endswith("ity") and len(word) > 4:
        return "X-ITY"
    if word.endswith("y") and len(word) > 2:
        return "X-Y"
    if word.endswith("al") and len(word) > 3:
        return "X-AL"
    if word.endswith("able") and len(word) > 5:
        return "X-ABLE"
    if word.endswith("ous") and len(word) > 4:
        return "X-OUS"
    if word.endswith("ed") and len(word) > 3:
        return "X-ED"
    if word.endswith("s") and len(word) > 3:
        return "X-S"
        
    # 4. Fallback for other hapax/unknown words
    return "X-UNKNOWN"

def training(sentences):
    """
    Computes initial tags, emission words and transition tag-to-tag probabilities
    Using Hapax Legomena pattern matching.
    """
    init_prob = defaultdict(lambda: 0)
    emit_prob = defaultdict(lambda: defaultdict(lambda: 0))
    trans_prob = defaultdict(lambda: defaultdict(lambda: 0))
    
    # 1. Count word frequencies to identify Hapax Legomena
    word_counts = defaultdict(int)
    for sentence in sentences:
        for word, tag in sentence[1:-1]: # Skip START and END tuples
            word_counts[word] += 1
            
    # 2. Build counts for transitions and emissions
    emit_counts = defaultdict(lambda: defaultdict(int))
    trans_counts = defaultdict(lambda: defaultdict(int))
    tag_total_tokens = defaultdict(int)
    
    init_prob['START'] = 1
    
    for sentence in sentences:
        previous_word, previous_tag = sentence[0]
        for word, tag in sentence[1:]:
            
            # --- Emission Counts ---
            # Always count the exact word
            emit_counts[tag][word] += 1
            tag_total_tokens[tag] += 1
            
            # If it's a Hapax word (appears once), we ALSO count it towards its pattern
            if word_counts[word] == 1:
                pattern = get_word_pattern(word)
                emit_counts[tag][pattern] += 1 
                # Note: We don't increment tag_total_tokens for patterns to keep probs valid

            # --- Transition Counts ---
            trans_counts[previous_tag][tag] += 1

            previous_word = word
            previous_tag = tag

    # 3. Calculate Transition Probabilities
    for prev_tag in trans_counts:
        total_transitions = sum(trans_counts[prev_tag].values())
        vocab_size = len(trans_counts[prev_tag])
        for curr_tag in trans_counts[prev_tag]:
            numerator = trans_counts[prev_tag][curr_tag] + epsilon_for_pt
            denominator = total_transitions + epsilon_for_pt * (vocab_size + 1)
            trans_prob[prev_tag][curr_tag] = numerator / denominator

    # 4. Calculate Emission Probabilities (With Pattern Smoothing)
    for tag in emit_counts:
        total_tokens = tag_total_tokens[tag]
        vocab_size = len(emit_counts[tag])
        
        denominator = total_tokens + emit_epsilon * (vocab_size + 1)
        
        for word_or_pattern in emit_counts[tag]:
            count = emit_counts[tag][word_or_pattern]
            emit_prob[tag][word_or_pattern] = (count + emit_epsilon) / denominator
            
    return init_prob, emit_prob, trans_prob

def viterbi_stepforward(i, word, prev_prob, prev_predict_tag_seq, emit_prob, trans_prob):
    """
    Does one step of the viterbi function
    """
    log_prob = {}
    predict_tag_seq = {}
    
    # --- Case 1: Start of Sentence (i=0) ---
    # We must enforce START tag logic strictly here
    if i == 0:
        # We initialize all known tags. Only START gets probability 0 (log(1)).
        # Others get epsilon.
        for tag in emit_prob:
            if tag == 'START':
                log_prob[tag] = 0
            else:
                log_prob[tag] = log(epsilon_for_pt)
            predict_tag_seq[tag] = [tag]
        
        # Explicitly ensure START is present if not in emit_prob
        if 'START' not in log_prob:
            log_prob['START'] = 0
            predict_tag_seq['START'] = ['START']
            
        return log_prob, predict_tag_seq

    # --- Case 2: General Step (i > 0) ---
    for i_tag in emit_prob:
        max_prob = -float('inf')
        optimal_prev_tag = None
        
        # 1. Emission Probability Lookup
        if word in emit_prob[i_tag]:
            # Exact word match
            emission_prob = emit_prob[i_tag][word]
        else:
            # Pattern match
            pattern = get_word_pattern(word)
            if pattern in emit_prob[i_tag]:
                emission_prob = emit_prob[i_tag][pattern]
            else:
                # Fallback to X-UNKNOWN or raw epsilon
                if 'X-UNKNOWN' in emit_prob[i_tag]:
                    emission_prob = emit_prob[i_tag]['X-UNKNOWN']
                else:
                    emission_prob = emit_epsilon
        
        log_emission = log(emission_prob)

        # 2. Iterate over previous tags to find best transition
        for last_i_tag in prev_prob:
            if prev_prob[last_i_tag] == -float('inf'):
                continue
                
            # Transition Probability
            if i_tag in trans_prob[last_i_tag]:
                transition = trans_prob[last_i_tag][i_tag]
            else:
                transition = epsilon_for_pt
            
            # Sum Log Probabilities
            current_prob = prev_prob[last_i_tag] + log(transition) + log_emission

            if current_prob > max_prob:
                max_prob = current_prob
                optimal_prev_tag = last_i_tag

        log_prob[i_tag] = max_prob
        
        if optimal_prev_tag:
            predict_tag_seq[i_tag] = prev_predict_tag_seq[optimal_prev_tag] + [optimal_prev_tag]
        else:
            predict_tag_seq[i_tag] = [] 
            
    return log_prob, predict_tag_seq

def viterbi_3(train, test):
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
            
        # Reconstruction of the best path
        last_tag = max(log_prob, key=log_prob.get)
        correct_tags = predict_tag_seq[last_tag] 
        correct_tags.append(last_tag)
        
        # The zip aligns (START, START), (Word1, Tag1)...
        predicts.append(list(zip(sentence, correct_tags[1:])))
        
    return predicts