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





