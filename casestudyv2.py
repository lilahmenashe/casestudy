import string
import stringprep

import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re

# downloading nltk stopwords list & punkt tokenizer models
nltk.download('stopwords')
nltk.download('punkt')

# Data cleaning and pre-processing
# function to preprocess input data and transform it into a suitable format for further semantic search implementation.
def preprocess(text):
    # tokenization of sentence into words
    text_tokens = word_tokenize(text)

    # removes stop words
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('english')]
    processed_text = " ".join(tokens_without_sw)

    # remove whitespace and newlines
    processed_text = " ".join(text.split())

    # remove urls using regular expression
    processed_text = re.sub('http://\S+|https://\S+', '', processed_text)

    # remove words with numbers
    processed_text = ''.join([i for i in processed_text if not i.isdigit()])

    # remove punctuation
    processed_text = processed_text.translate(str.maketrans('', '', string.punctuation))

    # convert to lower case
    processed_text = processed_text.lower()
    return processed_text


df_reddit = pd.read_csv('reddit.csv')
df_twitter = pd.read_csv("twitter.csv")

# define the number of samples you'd like to use in the combined DataFrame
num_samples = 3000

# create a new dataframe by combining the two dataframes (df_reddit & df_twitter)
full_data = pd.DataFrame(columns=['text'])

# I replaced iterrows with to_dic since it's much faster
for row in df_reddit.to_dict('rows_iteration'):
    full_data = full_data.append({'text': preprocess(row['body'])}, ignore_index=True)

    # remove this if statement to read the full dataset
    if len(full_data) > num_samples:
        break
print('Loading....')

for row in df_twitter.to_dict('rows_iteration'):
    full_data = full_data.append({'text': preprocess(row['tweet'])}, ignore_index=True)

    # remove this if statement to read the full dataset
    if len(full_data) > num_samples:
        break
print('Thanks for your patience!')


from sentence_transformers import SentenceTransformer, util
import torch

# download the pre-trained sentence transformer.
model = SentenceTransformer('all-MiniLM-L6-v2')

# provide the combined DataFrame to the model
corpus = list(full_data['text'])
unique_words = set()

# build list with unique words
for sentence in corpus:
    for w in sentence.split():
        unique_words.add(w)
corpus = list(unique_words)

# The words stored in the corpus list are encoded by calling model.encode()
corpus_embeddings = model.encode(corpus, convert_to_tensor=True)


def semantic_search(query):
    # Find the closest 5 words of the corpus for each query sentence based on cosine similarity
    top_k = min(5, len(corpus))
    q_embedding = model.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(q_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("\nQuery:", query)
    print("Top %d most similar words in corpus:"%top_k)

    for score, idx in zip(top_results[0], top_results[1]):
        print(corpus[idx], "(Score: {:.4f})".format(score))

    return semantic_search


# Queries input from ths user
while True:
    query = input("\nPlease enter a query: ")
    semantic_search(query)



