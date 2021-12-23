
## Stock Price Prediction using Financial Tweets

The files for the project can be found as below: 

1. notebooks/ : Contains the Ipython notebooks that were required for building models and analysis
2. pics/ : The folder contains the model results
3. App/: Contains the code for the web-application
4. models: Contains the trained LSTM model
5. data_extraction/ : Scripts for tweet data extraction from financial accounts, along with news extraction script. It contains the Airflow pipeline as well.

To run the web app locally, install Streamlit using pip and run 'streamlit run app.py' at the 'App' directory.

To access the application deployed on Streamlit Cloud, access https://share.streamlit.io/kguo98/stock_prediction_app/main/app.py