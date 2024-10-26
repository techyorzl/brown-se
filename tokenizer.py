import nltk
import os
import pickle
from nltk.stem.porter import PorterStemmer
import math

fileNames = os.listdir('brown')  # List the name of all the docs inside the brown corpus

stemmer = PorterStemmer()  # Object for Porter stemmer

# Data structures to store term frequencies and distinct words
tf_dict = {}
distinct_words = set()    # Distinct words before normalization
normalized_words = set()  # Distinct words after normalization

# Process each file in the corpus
for files in fileNames:
    file_path = os.path.join('brown', files)
    tf_dict[str(files)] = {}

    # Reading all the files word by word
    with open(file_path, "r") as f:
        content = f.read()
        for word in content.split():
            ind = word.find('/')
            if word.find(',') == -1 and word.find("'") == -1 and word.find('(') == -1 and ind != -1:
                word = word[:ind].lower()  # Convert to lowercase
                distinct_words.add(word)
                word = stemmer.stem(word)  # Apply Porter stemmer
                normalized_words.add(word)

                if word in tf_dict[files]:
                    tf_dict[files][word] += 1
                else:
                    tf_dict[files][word] = 1

print(f"Number of distinct words before normalization: {len(distinct_words)}")

# Inverted index construction
invertedIndex = {}
for term in normalized_words:
    invertedIndex[term] = []
    for file in fileNames:
        if term in tf_dict[file]:
            invertedIndex[term].append(file)

# Calculating TF-IDF values
n = len(fileNames)
tf_idf = {}

for files in fileNames:
    tf_idf[files] = {}
    for key in tf_dict[files]:
        tf = 1 + math.log(tf_dict[files][key], 10)
        idf = math.log(n / (1.0 * len(invertedIndex[key])), 10)
        tf_idf[files][key] = tf * idf

# Dumping all data structures to pickle files
pickle.dump(distinct_words, open("dist_words.p", "wb"))
pickle.dump(tf_dict, open("termFr_dict.p", "wb"))
pickle.dump(tf_idf, open("termFr_idf.p", "wb"))
pickle.dump(invertedIndex, open("invertedIndex.p", "wb"))
