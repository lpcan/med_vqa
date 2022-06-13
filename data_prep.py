# Functions to prep input data

from attr import dataclass
import torch.utils.data as data
from gensim.models import KeyedVectors, keyedvectors
from gensim import utils
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
        self.unique_answers = possible_answers(self)
        
        # Convert answers into labels
        self.answers = [self.unique_answers[a] for a in self.answers]

    def __getitem__(self, idx):
        # Find the id of the image
        img_id = self.images[idx]
        # Open the image 
        img = Image.open(self.data_dir + img_id.strip() + ".jpg")
        img = img.resize((256, 256)) # resize to 256 x 256?
        v = np.asarray(img)
        if self.transform:
            v = self.transform(v)

        # Get the question and convert to a tensor. All sentences padded to max question length.
        q_words = self.questions[idx]
        q = np.array([self.question_vocab.word2idx('<pad>')]  * self.max_q_len)
        q[:len(prepare_text(q_words).split(' '))] = self.question_vocab.sentence_to_idx(q_words)

        # # Do the same with the answers
        # a_words = self.answers[idx]
        # a = np.array([self.answer_vocab.word2idx('<pad>')] * self.max_ans_len)
        # q[:len(prepare_text(a_words).split(' '))] = self.answer_vocab.sentence_to_idx(a_words)
        
        # Convert answer to label
        a = self.answers[idx]
        
        return v, q, a

    def __len__(self):
        return len(self.questions)
        
# Create the vocab dictionary
def create_dict(path, data_dir):
    if len(glob.glob(data_dir + "vocab")) > 0:
        # Use previously calculated word vectors for this dataset
        wv = KeyedVectors.load_word2vec_format(data_dir+"word_vec", binary=True)
    else:
        wv = KeyedVectors.load_word2vec_format(path, binary=True)
        
        # Only want to keep vocab that will be used
        wv_new = {}
        # Create an entry for each word in our vocab
        f = open(glob.glob(data_dir+"All_QA_pairs*.txt")[0])
        for line in f:
            _, q, a = line.split('|')
            text = q + a 
            text = prepare_text(text)
            for word in text.split(' '):
                if word not in wv_new:
                    if word not in wv:
                        # Randomly generate a vector
                        wv_new[word] = np.random.randn(len(wv[0]))
                    else:
                        # Use pretrained word vector
                        wv_new[word] = wv[word]
        m = keyedvectors.Word2VecKeyedVectors(vector_size=len(wv[0]))
        m.add_vectors(list(wv_new.keys()), np.array(list(wv_new.values())))
        with utils.open(data_dir + "word_vec", 'wb') as fout:
            fout.write(utils.to_utf8("%s %s\n" % (len(wv_new), len(wv[0]))))
            for word, row in wv_new.items():
                fout.write(utils.to_utf8(word) + b" " + row.tostring())
        wv = m
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

# Get list of possible answers
def possible_answers(dataset):
    answers = list(set(dataset.answers))
    answers = {answers[i]: i for i in range(len(answers))} # Create a dictionary with a unique index for each answer
    return answers


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

