# Classes and functions to help with translating text to idx and vice versa

import glob
import numpy as np
from gensim.models import KeyedVectors
import torch
from transformers import AutoTokenizer

import data_prep

class Vocab:
    def __init__(self, data_dirs):
        # Construct a tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.1')

    def sentence_to_idx(self, sentence, max_len=None):
        if max_len:
            return self.tokenizer.encode(sentence, add_special_tokens=True, padding='max_length', max_length=max_len)
        else:
            return self.tokenizer.encode(sentence, add_special_tokens=True)

    def idx_to_sentence(self, idxs):
        self.tokenizer.decode(idxs, skip_special_tokens=True)

class Ans_Translator:
    def __init__(self, data_dir):
        # Get all unique answers
        self.answer_dict = self.create_answer_dict(data_dir)
        self.answer_list = list(self.answer_dict)

    # Get list of possible answers
    def create_answer_dict(self, data_dir):
        f = open(glob.glob(data_dir + "All_QA_Pairs*.txt")[0], encoding='utf-8')
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
