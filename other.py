"""Dataset analysis functions"""

import re
from cv2 import rotate
import matplotlib.pyplot as plt
import glob
import numpy as np

from data_prep import create_dict, prepare_text

# Run a check of all words in VQA-RAD dataset, find those that don't exist in the dictionary, and write to a file
def check_vocab():
    f = open(glob.glob(data_dir + "All_QA_Pairs*")[0], "r")
    dict = create_dict("bio_embedding_extrinsic")
    bad_words = open("bad_words2.txt", "w+")
    for i, line in enumerate(f):
        # Replace some punctuation. Based on punctuation present in datasets
        line = line.replace('/', ' or ')
        line = line.replace('>', ' greater than ')
        line = line.replace('<', ' less than ')
        line = line.replace('’', '\'')
        # Replace any other non standard characters with a space, except for hyphen
        line = re.sub('[^0-9a-zA-Z-]+', ' ', line)
            
        for word in line.split():
            # skip synpic* words
            if re.search("synpic[0-9]*", word):
                continue

            # try to find the word in our dictionary
            try:
                dict[word.lower()]
            except KeyError:
                bad_words.write(f"{i} {word.lower()}\n")

    f.close()
    bad_words.close()

def count_unique_answers():
    answers = {}
    f = open(glob.glob(data_dir+"All_QA_pairs*")[0], "r")
    f_out = open("data_analysis/unique_answers_" + data_dir.split('/')[-2] + ".txt", "w+")

    for line in f:
        _, _, a = line.split("|")
        a = a.strip().lower()
        if a not in answers:
            answers[a] = 1
        else:
            answers[a] += 1
    dict_sorted = sorted(answers.items(), key=lambda x:x[1])
    dict_sorted.reverse()
    for tup in dict_sorted:
        key, val = tup
        f_out.write(f"{key} : {val}\n")
    f_out.close()
    f.close()

# Make a pie chart of the frequencies of the different answers
def chart_unique_answers():
    f = open("data_analysis/unique_answers_" + data_dir.split('/')[-2] + ".txt", "r")
    other = 0
    ans = []
    for line in f:
        num = int(line.split(":")[-1])
        if num < 3:
            other += 1
        else:
            ans.append(int(line.split(":")[-1]))
    ans.append(other)
    plt.pie(ans)
    plt.show()

# Get distributions of lengths of questions/answers
def length_distribution(part='question'):
    f = open(glob.glob(data_dir + "All_QA_Pairs*")[0], "r")
    lengths = {}
    for line in f:
        if part == "question":
            pos = -2
        else:
            pos = -1
        text = line.split('|')[pos].strip()
        length = len(text.split(' ')) # count number of words
        if length not in lengths:
            lengths[length] = 1
        else:
            lengths[length] += 1
    
    # Plot the distribution as a bar chart
    plt.bar(lengths.keys(), lengths.values())
    plt.title(f"{part.capitalize()} length distribution")
    plt.xlabel("Length")
    plt.xticks([x+1 for x in range(max(lengths.keys()))])
    plt.ylabel("Counts")
    plt.savefig(f"data_analysis/{part.capitalize()} length distribution {data_dir.split('/')[-2]}")
    plt.show()

class Node:
    def __init__(self, word):
        self.word = word
        self.count = 1
        self.children = {}
    def insert_sentence(self, sentence):
        curr = self
        for word in sentence.split(" ")[:4]: # only worry about the first four words
            if word in curr.children:
                curr = curr.children[word]
                curr.count += 1
            else:
                new_child = Node(word)
                curr.children[word] = new_child
                curr = new_child
    def show(self):
        print(f"{self.word} : {self.count} -> {[child for child in self.children.keys()]}")
        for child in self.children.keys():
            self.children[child].show()
    # Get the entire tree back for display as a pie chart
    def get_tree(self, level, word_lists, count_lists):
        children = sorted(self.children, key=lambda x: self.children[x].count)
        children.reverse()
        for child in children:
            word_lists[level].append(child) # append this child to the right word list
            count_lists[level].append(self.children[child].count) # append the count to the right word list
            if level == 4:
                return word_lists, count_lists # finished traversing tree
            else:
                self.children[child].get_tree(level+1, word_lists, count_lists) # recurse
        return word_lists, count_lists

def chart_q_structure():
    f = open(glob.glob(data_dir + "All_QA_Pairs*")[0], "r")
    tree = Node(None)
    for line in f:
        _, q, _ = line.split('|')
        tree.insert_sentence(prepare_text(q))
    
    # Now use the tree to construct a nested pie chart
    word_lists, count_lists = tree.get_tree(0, [[], [], [], []], [[], [], [], []])
    cmap = plt.cm.winter, plt.cm.cool, plt.cm.summer, plt.cm.autumn, plt.cm.gist_heat, plt.cm.copper, plt.cm.pink
    colours = [c(0.6) for c in cmap]
    total = sum(count_lists[0])
    pie_colours = [[], [], [], []]
    
    for level in range(4):
        running_tot = 0
        for i, count in enumerate(count_lists[level]):
            running_tot += count
            if count / total < 0.02: # this wedge is too small, make it white
                pie_colours[level].append((1,1,1))
                word_lists[level][i] = ''
            # assign the correct colours to the next wedge
            elif running_tot <= count_lists[0][0]:
                pie_colours[level].append(colours[0])
            elif running_tot <= sum(count_lists[0][:2]):
                pie_colours[level].append(colours[1])
            elif running_tot <= sum(count_lists[0][:3]):
                pie_colours[level].append(colours[2])
            elif running_tot <= sum(count_lists[0][:4]):
                pie_colours[level].append(colours[3])
            elif running_tot <= sum(count_lists[0][:5]):
                pie_colours[level].append(colours[4])
            elif running_tot <= sum(count_lists[0][:6]):
                pie_colours[level].append(colours[5])
            elif running_tot <= sum(count_lists[0][:7]):
                pie_colours[level].append(colours[6])
            else:
                pie_colours[level].append((1, 0, 1))

        
    size = 0.2
    plt.figure()
    ax = plt.axes()
    label_distances = [0.7, 0.8, 0.89, 0.9]
    for level in range(4):
        wedge, text = ax.pie(count_lists[3-level], 
               radius = 1-level*size, 
               colors=pie_colours[3-level],
               wedgeprops=dict(width=size,edgecolor='w'),
               labels=word_lists[3-level],
               labeldistance=label_distances[3-level], 
               rotatelabels=True, 
               textprops={'fontsize': 8})
        for i,t in enumerate(text):
            t.set_horizontalalignment('center')
            t.set_verticalalignment('center')
            t.set_fontsize(min(max(2*(7-len(word_lists[3-level][i])), 4), 8))
    ax.set(aspect='equal')
    plt.title(f"Question structure for {data_dir.split('/')[-2]}")
    plt.savefig(f"q structure {data_dir.split('/')[-2]}")
    plt.show()

# data_dir = "../Datasets/ImageClef-2019-VQA-Med-Training/"
data_dir = "../Datasets/VQA-RAD/"
chart_q_structure()