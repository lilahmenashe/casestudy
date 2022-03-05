import os
import csv
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize


# downloading nltk stopwords list & punkt tokenizer models
nltk.download('stopwords')
nltk.download('punkt')


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.corpus import stopwords
import re
stopWords = stopwords.words('english')

# This function takes the string input, clean it and returns as the clean version
def preprocess(text):
    # removes extra lines
    from nltk.tokenize import word_tokenize
    text_tokens = word_tokenize(text)

    # removes stop words
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    processed_text = (" ").join(tokens_without_sw)

    # removes extra lines
    processed_text = " ".join(text.split())

    # remove urls using regular expression
    processed_text = re.sub('http://\S+|https://\S+', '', processed_text)

    # remove words with numbers
    processed_text = ''.join([i for i in processed_text if not i.isdigit()])
    return processed_text


df_reddit = pd.read_csv('reddit.csv')
df_twitter = pd.read_csv("twitter.csv")

# create a new dataframe by combining the two dataframes (df_reddit & df_twitter)
num_samples = 100

full_data = pd.DataFrame(columns=['text'])
for idx, row in df_reddit.iterrows():
    full_data = full_data.append({'text':preprocess(row['body'])},ignore_index=True)

    # remove this if statement to read the full dataset
    if len(full_data) > num_samples:
        break
print('loading reddit completed')
for idx, row in df_twitter.iterrows():
    full_data = full_data.append({'text':preprocess(row['tweet'])},ignore_index=True)
    # remove this if statement to read the full dataset
    if len(full_data) > num_samples:
        break
print('loading twitter completed')


from sentence_transformers import SentenceTransformer, util
import torch
# loading sentence transformer MinLM model fine tuned on a large dataset
embedder = SentenceTransformer('all-MiniLM-L6-v2')

corpus = list(full_data['text'])
unique_words = set()

# build set with unique words
for sentence in corpus:
    for w in sentence.split():
        unique_words.add(w)
corpus = list(unique_words)

# compute corpus embedding
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Queries examples
query = "iphone"

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(corpus))
q_embedding = embedder.encode(query, convert_to_tensor=True)

# use cosine-similarity between query embedding and corpus embedding
cos_scores = util.cos_sim(q_embedding, corpus_embeddings)[0]

# use torch.top_k to find the top 5 scores
top_results = torch.topk(cos_scores, k=top_k)

print("\nQuery:", query)
print("Top %d most similar words in corpus:"%top_k)

for score, idx in zip(top_results[0], top_results[1]):
    print(corpus[idx], "(Score: {:.4f})".format(score))


query = "coronavirus"

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(corpus))
q_embedding = embedder.encode(query, convert_to_tensor=True)

# use cosine-similarity between query embedding and corpus embedding
cos_scores = util.cos_sim(q_embedding, corpus_embeddings)[0]

# use torch.top_k to find the top 5 scores
top_results = torch.topk(cos_scores, k=top_k)

print("\nQuery:", query)
print("Top %d most similar words in corpus:"%top_k)

for score, idx in zip(top_results[0], top_results[1]):
    print(corpus[idx], "(Score: {:.4f})".format(score))

query = "Tesla"

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = min(5, len(corpus))
q_embedding = embedder.encode(query, convert_to_tensor=True)

# use cosine-similarity between query embedding and corpus embedding
cos_scores = util.cos_sim(q_embedding, corpus_embeddings)[0]

# use torch.top_k to find the top 5 scores
top_results = torch.topk(cos_scores, k=top_k)

print("\nQuery:", query)
print("Top %d most similar words in corpus:"%top_k)

for score, idx in zip(top_results[0], top_results[1]):
    print(corpus[idx], "(Score: {:.4f})".format(score))

