# Classes and functions to help with translating text to idx and vice versa

import glob
import numpy as np
from gensim.models import KeyedVectors
import torch

import data_prep

class Vocab:
    def __init__(self, data_dir):
        # Construct a vocab dictionary and corresponding embedding matrix
        self.vocab_dict = self.create_vocab(data_dir)# make vocab dictionary to get index from key
        self.vocab_list = list(self.vocab_dict) # vocab list to get key from index
        self.embeddings = self.get_embeddings(data_dir)

    def create_vocab(self, data_dir):
        # Get all vocabulary
        f = open(glob.glob(data_dir+"All_QA_Pairs*.txt")[0], encoding='cp1252')
        vocab = {'<pad>' : 0}
        idx = 1
        for line in f:
            _, q, a = line.split('|')
            text = data_prep.prepare_text(q + " " + a)

            for word in text.split(' '):
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
        return vocab

    def get_embeddings(self, data_dir):
        if len(glob.glob(data_dir + "word_vec.npy")) > 0:
            vectors = np.load(data_dir + "word_vec.npy") # Word vectors stored as numpy matrix
        else:
            wv = KeyedVectors.load_word2vec_format("bio_embedding_extrinsic", binary=True) # Load pretrained word vector
            dim = len(wv[0]) # dimension of the vectors
            vectors = np.zeros((len(self.vocab_list), dim))
            # Only want to keep vocab that will be used
            for word in self.vocab_list:
                if word not in wv:
                    # Generate a random vector - could try something more sophisticated than this
                    vectors[self.vocab_dict[word]] = np.random.randn(dim)
                else:
                    # Use pretrained word vector
                    vectors[self.vocab_dict[word]] = wv[word]
            
            np.save(data_dir + "word_vec", vectors)
            
        vectors = torch.FloatTensor(vectors)
        return vectors

    def word2idx(self, word):
        return self.vocab_dict[word]

    def sentence_to_idx(self, sentence):
        return [self.word2idx(w) for w in data_prep.prepare_text(sentence).split(' ')]

    def idx2word(self, idx):
        return self.vocab_list[idx]
    
    def idx_to_sentence(self, idxs):
        sentence = ''
        for idx in idxs:
            if self.idx2word(idx) == '<pad>':
                break
            else:
                sentence += self.vocab_list[idx] + ' '
        return sentence

class Ans_Translator:
    def __init__(self, data_dir):
        # Get all unique answers
        self.answer_dict = self.create_answer_dict(data_dir)
        self.answer_list = list(self.answer_dict)

    # Get list of possible answers
    def create_answer_dict(self, data_dir):
        f = open(glob.glob(data_dir + "All_QA_Pairs*.txt")[0], encoding='cp1252')
        all_answers = []
        for line in f:
            _, _, a = line.split('|')
            all_answers.append(data_prep.prepare_text(a))

        answers = list(set(all_answers))
        answers = {answers[i]: i for i in range(len(answers))} # Create a dictionary with a unique index for each answer
        return answers

    def ans_to_label(self, ans):
        return self.answer_dict[ans]
    
    def label_to_ans(self, label):
        return self.answer_list[label]
