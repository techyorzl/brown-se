# Brown Corpus Text Processing and Trie based Search Engine

This repository contains a Python project that processes the Brown corpus, performs stemming, and constructs an inverted index, along with calculating term frequency-inverse document frequency (TF-IDF) values. The processed data is serialized and stored using the `pickle` module for easy retrieval and analysis.

## Project Structure

The project directory contains the following files:

### `brown/`
- **Description**: This directory contains the text documents from the Brown corpus, which are processed in this project.

### `app.py`
- **Description**: This is the main script that performs the following operations:
  - Reads all documents in the `brown/` directory.
  - Tokenizes the text and filters out unwanted characters and punctuation.
  - Applies the Porter stemming algorithm to reduce words to their base forms.
  - Constructs a term frequency dictionary (`termFr_dict`) for each document.
  - Builds an inverted index to map each term to the documents containing it.
  - Calculates the TF-IDF values for each term in each document.
  - Serializes and saves the results (distinct words, term frequencies, TF-IDF values, and the inverted index) into separate `.p` files using `pickle`.

### `dist_words.p`
- **Description**: A pickle file that stores a set of distinct words extracted from the Brown corpus before normalization (stemming). This file can be loaded to retrieve the unique words for further analysis.

### `termFr_dict.p`
- **Description**: A pickle file containing a dictionary where each key is a document name, and the corresponding value is another dictionary representing the term frequencies of words in that document. This data structure provides insights into the frequency distribution of terms in each document.

### `termFr_idf.p`
- **Description**: A pickle file that stores the TF-IDF values for each term in every document. The TF-IDF value helps assess the importance of a term in a document relative to the entire corpus, which is crucial for information retrieval and text mining tasks.

### `invertedIndex.p`
- **Description**: A pickle file that contains an inverted index, mapping each term (after normalization) to a list of documents that contain it. This structure allows for efficient retrieval of documents based on specific terms.

## Requirements

To run this project, you need to have the following Python packages installed:

- `nltk`
- `pickle` (comes with the standard library)
- `math` (comes with the standard library)
- `os` (comes with the standard library)

You can install the required packages using pip:

```bash
pip install nltk
