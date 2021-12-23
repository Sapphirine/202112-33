from airflow import DAG
#from airflow.operators.python_operator import PythonOperator
from airflow.operators.python import PythonOperator

from datetime import datetime, timedelta  
import datetime as dt
import pandas as pd
import yfinance as yf
import requests
import lxml
from functools import reduce

import pandas as pd
import pickle
import yfinance as yf

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

import tweepy
import csv
import sys
from datetime import datetime
import pandas as pd
import re
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras

def tweetExtractor():

  #csvFile = open(stock + '.csv', 'a')
  #csvWriter = csv.writer(csvFile)
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
  stock = 'AAPL'
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


def get_and_append_data(**kwargs):
    
    # read stocks raw data
    data = pd.read_csv('stock_raw_data.csv', index_col='Date')
    data_today = yf.download( 
        tickers = "AAPL",
        period = "1d",
        interval = "1d",
        group_by = 'ticker',
        auto_adjust = True,
        prepost = True,
        threads = True,
        proxy = None
    )
    data.append(data_today)
    
    data.drop_duplicates(inplace=True)
    
    data.to_csv('appl_stock_raw_data.csv')
    print('Data Fetched, New Raw Data size', len(data))
    #return data_today['Close'][0]
#get_and_append_data()

def pre_process_data(**kwargs): 
    
    # create columns
    data = pd.read_csv('appl_stock_raw_data.csv', index_col = 'Date')
    cols = ['Open', 'High', 'Close', 'Volume', 'Low']
    for col in cols:
        indicator_name = col + '_MA_10d'
        data[indicator_name] = data[col].rolling(10).mean()
    
    data.dropna(inplace=True)
    data.drop_duplicates(keep='last', inplace=True)
    
    # Get the High column
    close_df = data['Close']
    # shift it back a day
    shifted_df = close_df.shift(-1)
    shifted_df = shifted_df.to_frame()
    shifted_df = shifted_df.rename(columns={'Close': 'Close_future'})
    merged_df = pd.concat([data, shifted_df], axis=1)
    
    print("Created feature columns with names", merged_df.columns)
    merged_df.to_csv('appl_merged_df.csv')
    #return merged_df


def prepare_train_test_data(**kwargs): 
    
#     ti = kwargs['ti']
#     merged_df = ti.xcom_pull(task_ids='pre_process_task')
    
    merged_df = pd.read_csv('appl_merged_df.csv', index_col='Date')
    x_cols  = ['Open_MA_10d', 'High_MA_10d', 'Close_MA_10d', 'Volume_MA_10d',
       'Low_MA_10d']
    y_cols = ['Close_future']

    data = merged_df[x_cols + y_cols]
    data.dropna(inplace=True)
    
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    test_data = scaled_data[-60,:]

    test_data.to_csv('test_data.csv')
    
    print("Obtained Data for prediction for len: ", test_data.shape)

    
def test_and_predict(**kwargs): 
    # read test data
    x_test = pd.read_csv('test_data.csv')
    
    model = keras.models.load_model('LSTMStockmodel.h5')
    print("Read Test Data, data of len: ", len(y_test))
    
    x_test = np.array(x_test)

    # Reshape the data
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))

    # Get the models predicted price values 
    predictions = model.predict(x_test)
    
    predictions.to_csv('LSTM_predictions.csv')

default_args = {
     'owner': 'tugup',
     'depends_on_past': False,
     'email': ['tg2749@columbia.edu'],
     'email_on_failure': False,
     'email_on_retry': False ,
     'retries': 0
    }

dag = DAG( 'stocks_analysis_ETL_7AM',
            default_args=default_args,
            description='Collect Stock Prices For Analysis',
            catchup=False, 
            start_date= datetime(2021,12,5), 
            schedule_interval= '0 7 * * *'  
          )  

##########################################
#3. DEFINE AIRFLOW OPERATORS
##########################################

tweetExtractor_task = PythonOperator(task_id = 'tweetExtractor_task', 
                                   python_callable = tweetExtractor, 
                                   provide_context = True,
                                   dag= dag )

get_and_append_data_task = PythonOperator(task_id = 'get_and_append_data_task', 
                                   python_callable = get_and_append_data, 
                                   provide_context = True,
                                   dag= dag )


pre_process_data_task= PythonOperator(task_id = 'pre_process_data_task', 
                                 python_callable = pre_process_data,
                                 provide_context = True,
                                 dag= dag)

prepare_train_test_data_task = PythonOperator(task_id = 'prepare_train_test_data_task', 
                                  python_callable = prepare_train_test_data,
                                  provide_context = True,
                                  dag= dag)      

test_and_predict_task = PythonOperator(task_id = 'test_and_predict_task', 
                                  python_callable = test_and_predict,
                                  provide_context = True,
                                  dag= dag)

##########################################
#4. DEFINE OPERATORS HIERARCHY
##########################################

tweetExtractor_task >> get_and_append_data_task  >> pre_process_data_task >> prepare_train_test_data_task >> test_and_predict_task

