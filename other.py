import re

from data_prep import create_dict

# Run a check of all words in VQA-RAD dataset, find those that don't exist in the dictionary, and write to a file
def check_vocab():
    f = open("../Datasets/ImageClef-2019-VQA-Med-Training/All_QA_Pairs_train.txt", "r")
    dict = create_dict("bio_embedding_extrinsic")
    bad_words = open("bad_words2.txt", "w+")
    for i, line in enumerate(f):
        # Need to do some replacements of non alphanumeric characters
        # line = line.replace('/', ' or ')
        # line = line.replace('\t', ' ')
        # line = line.replace('>', ' greater than ')
        # line = line.replace('<', ' less than ')
        # line = line.replace('|', ' ')
        # line = line.replace(',', ' ')
        # line = line.replace('’', '\'')
        # line = ''.join(c for c in line if (c.isalnum() and c.isascii()) or c == ' ' or c == '-')

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