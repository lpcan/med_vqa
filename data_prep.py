# Functions to prep input data

import torch
import torch.utils.data as data
import re
from PIL import Image
import numpy as np
import glob

# Create the dataset type
class VQADataset(data.Dataset):
    def __init__(self, data_dir, img_dir, vocab, ans_translator, transform=None):
        self.data_dir = data_dir
        self.img_dir = img_dir
        self.images = []
        self.questions = []
        self.answers = []
        f = open(glob.glob(data_dir+"All_QA_Pairs*.txt")[0], encoding='utf-8')
        for line in f:
            img, q, a = line.split('|')
            self.images.append(img)
            self.questions.append(prepare_text(q))
            self.answers.append(prepare_text(a))
        self.transform = transform

        self.vocab = vocab
        # self.answer_vocab = Vocab(self.answers)
        self.max_q_len = max([len(vocab.sentence_to_idx(q)) for q in self.questions])
        self.max_ans_len = max([len(a.split(' ')) for a in self.answers])
        self.ans_translator = ans_translator
        
    def __getitem__(self, idx):
        # Find the id of the image
        img_id = self.images[idx]
        # Open the image 
        img = Image.open(self.img_dir + img_id.strip() + ".jpg")
        img = img.resize((256, 256)) # resize to 256 x 256?
        v = np.asarray(img)
        if self.transform:
            v = self.transform(v)

        # Get the question and convert to a tensor. All sentences padded to max question length.
        q_words = self.questions[idx]
        q = torch.tensor(self.vocab.sentence_to_idx(q_words, self.max_q_len))

        # # Do the same with the answers
        # a_words = self.answers[idx]
        # a = np.array([self.answer_vocab.word2idx('<pad>')] * self.max_ans_len)
        # q[:len(prepare_text(a_words).split(' '))] = self.answer_vocab.sentence_to_idx(a_words)
        
        # Convert answer to label
        a = self.answers[idx]
        a = self.ans_translator.ans_to_label(a)

        return v, q, a

    def __len__(self):
        return len(self.questions)
        
# Prepare questions and answers
def prepare_text(text):
    text = text.lower().strip() # Lower case and strip white space

    # Replace some punctuation. Based on punctuation present in datasets
    text = text.replace('/', ' or ')
    text = text.replace('>', ' greater than ')
    text = text.replace('<', ' less than ')
    text = text.replace('’', '\'')
    # Replace any other non standard characters (other than hyphen) with a space
    text = re.sub('[^0-9a-zA-Z-]+', ' ', text)

    return text

# This class allows train/test split with different transforms
class DatasetFromSubset(data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        v, q, a = self.subset[index]
        if self.transform:
            v = self.transform(v)
        return v, q, a

    def __len__(self):
        return len(self.subset)
