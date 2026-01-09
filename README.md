# Natural-Language-Processing-NLP-for-Parts-of-Speech-tagging


This project implements a sequence of **Part-of-Speech (POS) Tagging** algorithms, evolving from a simple statistical baseline to a sophisticated Hidden Markov Model (HMM) capable of handling unseen words through morphological analysis. The tagger is trained and tested on the **Brown Corpus**.

## Project Overview

Part-of-Speech tagging is the process of assigning grammatical tags (e.g., Noun, Verb, Adjective) to words in a sentence. This project explores the trade-offs between simple frequency-based approaches and probabilistic models that utilize context and word morphology.

The core of the project is the **Viterbi algorithm**, a dynamic programming approach used to decode the most likely sequence of hidden states (tags) given a sequence of observations (words).

## Algorithms Implemented

### 1. Baseline Tagger
* **Logic:** Assigns the most frequent tag seen for each word in the training data. Unseen words are assigned the most common tag in the entire corpus (e.g., `NOUN`).
* **Purpose:** Establishes a performance floor for comparison.
* **Limitation:** Completely ignores sentence context.

### 2. Viterbi 1: Standard HMM
* **Logic:** Implements the standard Viterbi algorithm using:
    * **Transition Probabilities:** $P(Tag_i | Tag_{i-1})$ — The likelihood of a tag following the previous tag.
    * **Emission Probabilities:** $P(Word | Tag)$ — The likelihood of a specific word given a tag.
* **Smoothing:** Uses Laplace smoothing to handle zero-probability transitions.
* **Limitation:** Treats all unknown words identically, assigning them a uniform low probability.

### 3. Viterbi 2: Hapax Legomena Smoothing
* **Logic:** Improves handling of unknown words by analyzing **Hapax Legomena** (words appearing only once in the training set).
* **Innovation:** Instead of a uniform probability for unknowns, the model dynamically calculates the probability of an unknown word belonging to a specific tag based on the distribution of unique words for that tag.
* **Result:** Significantly improves accuracy on unseen words (e.g., knowing that a new word is more likely to be a Noun than a Conjunction).

## Performance

The models were evaluated on the Brown Corpus development set. The progression shows a clear improvement in the model's ability to generalize to new data.

| Algorithm | Overall Accuracy | Unseen Word Accuracy |
| :--- | :--- | :--- |
| **Baseline** | ~92% | ~20% |
| **Viterbi 1** | ~94% | ~20% |
| **Viterbi 2** | ~95% | ~65%+ |

## Usage

### Prerequisites
* Python 3.x
* No external dependencies required (uses standard Python libraries).

### Running the Tagger
To train and test a specific algorithm (e.g., `viterbi_2`) on the provided data:

```bash

python3 mp7.py --train data/brown-training.txt --test data/brown-dev.txt --algorithm viterbi_2
python3 mp7.py --train data/brown-training.txt --test data/brown-dev.txt --algorithm viterbi_1
python3 mp7.py --train data/brown-training.txt --test data/brown-dev.txt --algorithm baseline

