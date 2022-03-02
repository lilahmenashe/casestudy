# pip install google-cloud-bigquery
from google.cloud import bigquery
from google.oauth2 import service_account


# connect to the given json file
credentials = service_account.Credentials.from_service_account_file('nwo-sample-5f8915fdc5ec.json')

# using the given project_id from json file
project_id = 'nwo-sample'
client = bigquery.Client(credentials=credentials, project=project_id)

# Sql query to get the body text column from reddit.
reddit_query_test = """
    SELECT body 
    FROM `nwo-sample.graph.reddit`
    LIMIT 10000
"""

# Sql query to get the tweet column from Twitter
twitter_query_test = """
    SELECT tweet
    FROM `nwo-sample.graph.tweets`
    LIMIT 10000
"""


# function to convert query to dataframe
def query_to_df(query):
    query_name = client.query(query)
    save_results = query_name.result()
    return save_results.to_dataframe()


# calling the function above to convert reddit & twitter queries to dataframe
reddit = query_to_df(reddit_query_test)
twitter = query_to_df(twitter_query_test)

# converting dataframe to .csv file
reddit.to_csv('reddit.csv', index=False)
twitter.to_csv('twitter.csv', index=False)


import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import nltk
nltk.download('stopwords')
nltk.download('punkt')

from nltk.corpus import stopwords
import re
stopWords = stopwords.words('english')


# takes a string as input and returns a clean version
def preprocess(text):
    # remove extra lines
    from nltk.tokenize import word_tokenize
    text_tokens = word_tokenize(text)

    # remove stop words
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    processed_text = (" ").join(tokens_without_sw)

    # remove extra lines
    processed_text = " ".join(text.split())

    # remove urls using regular expression
    processed_text = re.sub('http://\S+|https://\S+', '', processed_text)

    # remove words with numbers
    processed_text = ''.join([i for i in processed_text if not i.isdigit()])
    return processed_text


# returns vectorizer object and tf-idf matrix object
def get_TfidfVectorizer(train_df):
    # TfidfVectorizer is used to convert a collection of raw text into tf-idf matrix
    vectorizer = TfidfVectorizer()

    # vectorizer.fit_transform: learns a vocabulary and builds a term-frequency matrix from the learned vocabulary
    tfidf = vectorizer.fit_transform(train_df['text'].values.astype('U'))

    return vectorizer, tfidf


# this takes a query, a trained vectorizer, and tfidf object and returns the top-k most similar posts 
def get_most_similar_documents(query, train_vectorizer, tfidf, top_k = 10):

    # extracts the words/vocabulary
    words = train_vectorizer.get_feature_names()

    # to handle query we will create a separate vectorizer using the same vocabulary learnt in the previous step
    query_vectorizer = TfidfVectorizer().fit(words)

    # transform the query into tf-idf object
    transformed_query = query_vectorizer.transform([query])

    # since we transformed our query into tf-idf matrix
    # we can use cosine_similarity to compare with our original corpus
    cosine_similarities = cosine_similarity(transformed_query, tfidf).flatten()

    # Get most similar articles based on the input query
    related_articles_indices = cosine_similarities.argsort()[:-top_k:-1]

    for index in related_articles_indices:
        print(full_data.iloc[index]['text'])
        print('___________________________')


# loads the datasets
df_reddit = pd.read_csv('reddit.csv')
df_twitter = pd.read_csv("twitter.csv")


# build a new dataframe by combining the two dataframes
num_samples = 4000

full_data = pd.DataFrame(columns=['text'])
for index, row in df_reddit.iterrows():
    full_data = full_data.append({'text':preprocess(row['body'])},ignore_index=True)

    # remove this if statement to read the full dataset
    if len(full_data) > num_samples:
        break
for index, row in df_twitter.iterrows():
    full_data = full_data.append({'text':preprocess(row['tweet'])},ignore_index=True)
    # remove this if statement to read the full dataset
    if len(full_data) > num_samples:
        break


# usage
# build a vectorizer and tf-idf matrix given the data
# the function assumes the text field is named text
train_vectorizer, tfidf  = get_TfidfVectorizer(full_data)


# to find the most similar posts pass the query, vectorizer and the tf-idf obj
get_most_similar_documents("Apple iphone", train_vectorizer, tfidf,top_k=20)

# optional (uncomment if needed)
# This will loop and take input from the user in case you want to avoid loading and fitting the datasets
# while True:
    # query = input("Please enter query \n")
    # get_most_similar_documents(query, train_vectorizer, tfidf,top_k=100)


