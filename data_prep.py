# Functions to prep input data

import torch.utils.data as data
from gensim.models import KeyedVectors
import re
from PIL import Image
import numpy as np
import glob

# Create the dataset type
class VQADataset(data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.images = []
        self.questions = []
        self.answers = []
        f = open(glob.glob(data_dir+"All_QA_pairs*.txt")[0])
        for line in f:
            img, q, a = line.split('|')
            self.images.append(img)
            self.questions.append(prepare_text(q))
            self.answers.append(prepare_text(a))
        self.transform = transform

        self.question_vocab = Vocab(self.questions)
        self.answer_vocab = Vocab(self.answers)
        self.max_q_len = max([len(q.split(' ')) for q in self.questions])
        self.max_ans_len = max([len(a.split(' ')) for a in self.answers])

    def __getitem__(self, idx):
        # Find the id of the image
        img_id = self.images[idx]
        # Open the image 
        img = Image.open(self.data_dir + img_id.strip() + ".jpg")
        img = img.resize((256, 256)) # resize to 256 x 256?
        v = np.asarray(img)
        # Need to resize the images to have the same size

        # Get the question and convert to a tensor
        q_words = self.questions[idx]
        q = np.array([self.question_vocab.word2idx('<pad>')]  * self.max_q_len)
        q[:len(prepare_text(q_words).split(' '))] = self.question_vocab.sentence_to_idx(q_words)

        # Do the same with the answers
        a_words = self.answers[idx]
        a = np.array([self.answer_vocab.word2idx('<pad>')] * self.max_ans_len)
        q[:len(prepare_text(a_words).split(' '))] = self.answer_vocab.sentence_to_idx(a_words)
        
        return v, q, a

    def __len__(self):
        return len(self.questions)
        
# Create the vocab dictionary
def create_dict(path):
    wv = KeyedVectors.load_word2vec_format(path, binary=True)
    # something else?
    return wv

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

class Vocab:
    def __init__(self, sentences):
        self.vocab_dict = self.construct_vocab_dict(sentences)

    def construct_vocab_dict(self, sentences):
        dict = {'<pad>': 0}
        idx = 1
        for s in sentences:
            # Split the sentences into individual tokens
            for w in prepare_text(s).split(' '):
                if w not in dict:
                    dict[w] = idx
                    idx += 1
        return dict

    def word2idx(self, word):
        return self.vocab_dict[word]

    def sentence_to_idx(self, sentence):
        return [self.word2idx(w) for w in prepare_text(sentence).split(' ')]

