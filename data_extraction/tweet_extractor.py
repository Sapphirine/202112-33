#Import Statement
import tweepy
import csv
import sys
from datetime import datetime
import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer


#Twitter API keys and tokens
consumer_key = 'bIN9xkFeg5RsuGbQ4fbjyd1K7'
consumer_secret = 'b3oG1GQOsSHFdtPrTYRU6e8UNpOmOpZkV2hBhBsFNyLe4fywoy'

access_token = '701546611676155904-xDsgJZSZC5mImpmVt2GueauyStCBI1o'
access_token_secret = 'YPudWrkR8mQFYCWLZjBdGm62UKPEKkbzj7qrCM2704uLO'


#Authentication 
auth = tweepy.auth.OAuthHandler(consumer_key ,consumer_secret)
auth.set_access_token(access_token, access_token_secret)

#Stores Authentication and waits on Twitter's rate limit
api = tweepy.API(auth, wait_on_rate_limit = True)
#Writes data from function to chosen file                

vader = SentimentIntensityAnalyzer()

def tweetExtractor(stock):

    #csvFile = open(stock + '.csv', 'a')
    #csvWriter = csv.writer(csvFile)
    if stock == 'AMZN': 
      company = 'Amazon'
    if stock == 'GOOG': 
      company = 'Google'
    if stock == 'TSLA': 
      company = 'Tesla'
    if stock == 'MSFT': 
      company = 'Microsoft'
    if stock == 'AAPL': 
      company = 'Apple'

    print("extracting for company", company)

    query = company + " from:wsj OR from:reuters OR from:business OR from:cnbc OR from:RANsquawk OR from:wsjmarkets OR from:Benzinga OR from:Stocktwits OR from:BreakoutStocks OR from:bespokeinvest OR from:nytimesbusiness OR from:IBDinvestors OR from:WSJDealJournal"
    #print(query)
    today_date = datetime.today().strftime('%Y-%m-%d')
    #print(today_date)

    tweets = []
    for tweet in tweepy.Cursor(api.search_tweets, q = query, until = today_date, lang = "en", result_type='popular').items():
        status = api.get_status(tweet.id, tweet_mode="extended")
        try:
            text = status.retweeted_status.full_text
        except AttributeError:  # Not a Retweet
            text = status.full_text

        #print(text)
        
        text = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', " ", text)

        tweets.append({'timestamp': tweet.created_at, 'text': text, 'company': stock})

    tweets_df = pd.DataFrame.from_dict(tweets)
    scores = tweets_df['text'].apply(vader.polarity_scores).tolist()
    scores_df = pd.DataFrame(scores)
    # Join the DataFrames of the news and the list of dicts
    tweets_df = tweets_df.join(scores_df, rsuffix='_right')

    # Convert the date column from string to datetime
    tweets_df['timestamp'] = pd.to_datetime(tweets_df.timestamp).dt.date

    tweets_df = tweets_df.rename({'compound': 'sentiment'})

    #print(tweets_df)
    main_df = pd.read_csv('all_tweets.csv', index_col=False)
    main_df = main_df.append(tweets_df)
    print(main_df.shape)
    main_df.to_csv('all_tweets.csv', index=False)   
  

if __name__ == '__main__':

    stocks = ['AAPL', 'AMZN', 'GOOG', 'TSLA', 'MSFT']
    for stock in stocks:
      tweetExtractor(stock)