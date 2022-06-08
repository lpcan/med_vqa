"""Dataset analysis functions"""

import re
import matplotlib.pyplot as plt
import glob

from data_prep import create_dict

data_dir = "../Datasets/ImageClef-2019-VQA-Med-Training/"
#data_dir = "../Datasets/VQA-RAD/"

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
    plt.savefig(f"data_analysis/{part.capitalize()} length distribution")
    plt.show()

