#NOTE: Once you download this file, run the the command: streamlit run ui_indi.py 
# in the cmd with correct file directory.

import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import math
import string

nltk.download('stopwords')

def tokenize_document(document, n=1):
    """
    Tokenize a document into phrases of length n.

    Args:
        document (str): The document text to be tokenized.
        n (int): The length of phrases (e.g., 1 for single words, 2 for bi-grams, 3 for tri-grams).

    Returns:
        list: A list of phrases (tokens) extracted from the document.
    """
    tokens = word_tokenize(document)  # Tokenize the document into words
    phrases = list(ngrams(tokens, n))  # Generate n-grams from the tokens
    return phrases

def Hash_calculation(doc):
    q = 1e10 + 7
    base = 31
    hashed_doc = []
    for word in doc:
        hash_val = 0
        for i in range(len(word)):
            hash_val += (ord(word[i]) * pow(base, (len(word) - i)))
        hash_val %= q
        hashed_doc.append(hash_val)        
    return hashed_doc

def main():
    st.title("Document Similarity Checker")

    uploaded_file1 = st.file_uploader("Upload Document 1", type=["txt"])
    uploaded_file2 = st.file_uploader("Upload Document 2", type=["txt"])

    if uploaded_file1 is not None and uploaded_file2 is not None:
        document1 = uploaded_file1.read().decode("utf-8").lower()
        document2 = uploaded_file2.read().decode("utf-8").lower()

        document1 = document1.translate(str.maketrans('', '', string.punctuation))
        document2 = document2.translate(str.maketrans('', '', string.punctuation))

        tokens_doc1 = tokenize_document(document1, n=1)
        tokens_doc2 = tokenize_document(document2, n=1)

        converted_doc1 = [''.join(i) for i in tokens_doc1]
        converted_doc2 = [''.join(i) for i in tokens_doc2]

        stop_words = set(stopwords.words('english'))

        without_stopwords_doc1 = [w for w in converted_doc1 if not w.lower() in stop_words]
        without_stopwords_doc2 = [w for w in converted_doc2 if not w.lower() in stop_words]

        stemmed_doc1 = [PorterStemmer().stem(w) for w in without_stopwords_doc1]
        stemmed_doc2 = [PorterStemmer().stem(w) for w in without_stopwords_doc2]

        hashed_doc_1 = Hash_calculation(stemmed_doc1)
        hashed_doc_2 = Hash_calculation(stemmed_doc2)

        set_doc1 = set(hashed_doc_1)
        set_doc2 = set(hashed_doc_2)
        st.write(set_doc1)
        st.write(set_doc2)
        # st.write(len(set_doc1))
        # st.write(len(set_doc2))
        
        similarity_percentage = (2 * len(set_doc1.intersection(set_doc2)) * 100) // (len(set_doc1) + len(set_doc2))

        st.write(f"Similarity between Document 1 and Document 2: {similarity_percentage}%")

if __name__ == "__main__":
    main()
