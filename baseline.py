"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
        '''
        input:  training data (list of sentences, with tags on the words). E.g.,  [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
                test data (list of sentences, no tags on the words). E.g.,  [[word1, word2], [word3, word4]]
        output: list of sentences, each sentence is a list of (word,tag) pairs.
                E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
        '''
        word_bag = []
        word_bag2 = []
        test_bag1 = []
        test_bag2 = []

        unique_words = {}
        unique_tags = {}


        for data in train:
                for entry in data:
                        word, tag = entry
                        any_tag = tag
                        if word not in unique_words:
                                unique_words[word] = {}
                        if tag in unique_words[word]:
                                unique_words[word][tag] += 1
                        else:
                                unique_words[word][tag] = 1
                        if tag not in unique_tags:
                                unique_tags[tag] = 1
                        else:
                                unique_tags[tag] +=1

        tag_dictionary = {} 
        most_common_tag = max(unique_tags, key = unique_tags.get)
                        
        for data2 in test:
                for entry2 in data2:
                        if entry2 in unique_words:
                                tag_list = unique_words[entry2]
                                largest_tag = max(unique_words[entry2], key = unique_words[entry2].get)
                                
                                test_bag2.append((entry2, largest_tag))
                        else:
                                test_bag2.append((entry2, most_common_tag))
                test_bag1.append(test_bag2)
                test_bag2 = []

        return test_bag1