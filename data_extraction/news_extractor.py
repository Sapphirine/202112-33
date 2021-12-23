from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import os
import pandas as pd
import matplotlib.pyplot as plt
import requests
# NLTK VADER for sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import json
from datetime import datetime
from datetime import timedelta
analyzer = SentimentIntensityAnalyzer()

def get_finance_news(ticker): 

	today_date = datetime.today()
	from_date = today_date - timedelta(days=1)

	print(ticker)
	#print(q)
	q = 'https://eodhistoricaldata.com/api/news?api_token=OeAFFmMliFG5orCUuwAKQ8l4WWFQ67YX&s=' + ticker + '&offset=0&limit=1000&from=' + from_date.strftime('%Y-%m-%d') + '&to=' + today_date.strftime('%Y-%m-%d')
	x = requests.get(q)
	json_values = x.json()
	print(len(json_values))
	news = []
	for i in range(len(json_values)):
		news.append(
    		{'Date': json_values[i]['date'], 
    		'headline': json_values[i]['content']
    		}
    		)
	news_df = pd.DataFrame.from_dict(news)
	news_df['Date'] = pd.to_datetime(news_df['Date']).dt.date
	news_df['company'] = ticker

	news_df['sentiment'] = news_df['headline'].apply(analyzer.polarity_scores).apply(lambda x : x['compound'])
	
	try:
		main_df = pd.read_csv('all_news.csv', index_col=False)
		main_df = main_df.append(news_df)
		print(main_df.shape)
		main_df.to_csv('all_news.csv', index=False)
	except Exception as e: 
		print(news_df.shape)
		news_df.to_csv('all_news.csv', index=False)

    
if __name__ == '__main__': 
	tickers = ['AMZN.US','AAPL.US','GOOGL.US','TSLA.US','MSFT.US']
	for ticker in tickers:
		get_finance_news(ticker)
