import pandas as pd
import json
from pathlib import Path
import re
from nltk.corpus import stopwords
from datetime import datetime
from tqdm import tqdm

from config import *
from utils import *


class DataLoader:
    """
        DataLoader class is responsible for loading data, cleansing and transforming for the model
    """

    def __init__(self):
        """

        """
        self.read_from_local_file = Config.read_from_local
        self.stock_name = Config.stock_name
        self.from_date = Config.from_date
        self.to_date = Config.to_date
        self.news = None
        self.stocks = None
        self.stocks_final = None
        self.price = []
        self.headlines = []
        self.clean_headlines = []

    def load_data(self):
        """
            Loading data and removing unmatched dates across news and stocks datasets
        :return:
        """
        print('[' + str(datetime.now()) + '] - ' + 'Loading Datasets...')

        self.news = DataLoader.get_news_data(self.read_from_local_file)
        self.stocks = DataLoader.get_stock_data(self.stock_name, self.from_date, self.to_date,
                                                self.read_from_local_file)

        print('[' + str(datetime.now()) + '] - ' +
              "Loaded {0} rows with {1} columns of news headlines".format(
                self.news.shape[0], self.news.shape[1]))

        print('[' + str(datetime.now()) + '] - ' +
              "Duplicate Rows except first occurrence based on all columns are : {0}".format(
                  self.news[self.news.duplicated()].shape[0]))

        print('[' + str(datetime.now()) + '] - ' +
              "Loaded {0} rows with {1} columns of historical prices of {2} stock".format(
                self.stocks.shape[0], self.stocks.shape[1], self.stock_name))

        if len(set(self.news.Date)) > len(set(self.stocks.Date)):
            self.news = self.news[self.news.Date.isin(self.stocks.Date)]

        if len(set(self.stocks.Date)) > len(set(self.news.Date)):
           self.stocks = self.stocks[self.stocks.Date.isin(self.news.Date)]

        if len(set(self.stocks.Date)) == len(set(self.news.Date)):
            print('[' + str(datetime.now()) + '] - ' + "Equal unique Dates in two dataframes of {}".format(len(set(self.stocks.Date))))
        else:
            print('[' + str(datetime.now()) + '] - ' + "Stocks Length: {}".format(len(self.stocks.Date)))
            print('[' + str(datetime.now()) + '] - ' + "News   Length: {}".format(len(self.news.Date)))

    def create_prices_with_headlines_list(self):
        """
            Create a list of the opening prices and their corresponding daily headlines from the news
        :return:
        """

        price = []
        headlines = []

        # Track progress
        for row in tqdm(self.stocks_final.iterrows()):
            daily_headlines = []
            date = row[1]['Date']
            price.append(row[1]['Close'])
            #price.append(row[1]['Open'])
            for row_ in self.news[self.news.Date == date].iterrows():
                daily_headlines.append(row_[1]['Headline'])

            headlines.append(daily_headlines)

        self.price = price
        self.headlines = headlines

        if len(price) == len(headlines):
            print('\n[' + str(datetime.now()) + '] - ' + 'Prices and Headlines list are equals')

    def get_clean_headlines(self):
        clean_headlines = []
        for daily_headlines in self.headlines:
            clean_daily_headlines = [DataLoader.clean_text(headline) for headline in daily_headlines]
            clean_headlines.append(clean_daily_headlines)

        self.clean_headlines = clean_headlines
        return clean_headlines

    def calculate_prices_periods_differences(self):
        # Calculate the difference in opening prices between the following and current day.
        # The model will try to predict how much the Open value will change based on the news.
        # dj = self.stocks.set_index('Date').diff(periods=1)
        # dj['Date'] = dj.index
        # dj = dj.reset_index(drop=True)
        # # Remove unneeded features
        # dj = dj.drop(['High', 'Low', 'Close', 'Volume', 'Adj Close'], 1)
        # dj = dj[dj.Open.notnull()]
        # self.stocks_final = dj

        # Not Difference Price
        dj = self.stocks
        dj = dj.drop(['High', 'Low', 'Open', 'Volume', 'Adj Close'], 1)
        dj = dj[dj.Close.notnull()]
        self.stocks_final = dj

    def combine_news_dataframes(dfs_list):
        cols = ['Date', 'Headline']
        dfs = []

        for df in dfs_list:
            df = df[cols]
            df.columns = cols
            dfs.append(df)

        df = pd.concat(dfs)
        df['Headline'] = df['Headline'].str.strip()
        df = df[~df.duplicated()]
        df = df.sort_values(by='Date')
        df = df.reset_index(drop=True)

        return df

    def normalize_prices(self):
        price_array = self.price
        max_price = max(price_array)
        min_price = min(price_array)

        norm_price = []
        for p in price_array:
            norm_price.append(normalize(p, max_price, min_price))
        return norm_price

    def unnormalize_prices(self, predictions):
        price_array = self.price
        max_price = max(price_array)
        min_price = min(price_array)

        unnorm_predictions = []
        for pred in predictions:
            unnorm_predictions.append(unnormalize(pred, max_price, min_price))
        return unnorm_predictions


    @staticmethod
    def get_news_data(from_local=False):
        if from_local:
            news = pd.read_csv('./dataset/final/news.csv', quotechar='"', sep=',')
        else:
            crawled = DataLoader.read_crawled_news('dataset/crawled')
            kaggle = DataLoader.read_kaggle_news('dataset/kaggle')
            webhose = DataLoader.read_webhose_news('dataset/webhose')

            news = DataLoader.combine_news_dataframes([crawled, kaggle, webhose])
            news.to_csv('./dataset/final/news.csv', index=False, quotechar='"', sep=',')
        return news

    @staticmethod
    def get_stock_data(stock_name, from_date, to_date, from_file=False):
        if from_file:
            stock_df = pd.read_csv("./dataset/final/" + stock_name + '.csv')
        else:
            stock_df = pdr.get_data_yahoo(stock_name, from_date, to_date)
            stock_df.reset_index(level=0, inplace=True)
            stock_df['Date'] = stock_df['Date'].dt.date
            stock_df.to_csv('./dataset/final/' + stock_name + '.csv', index=False, sep=',')
        return stock_df

    @staticmethod
    def clean_text(text, remove_stopwords=True):
        """Remove unwanted characters and format the text to create fewer nulls word embeddings"""

        # A list of contractions from
        # http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
        contractions = {
            "ain't": "am not",
            "aren't": "are not",
            "can't": "cannot",
            "can't've": "cannot have",
            "'cause": "because",
            "could've": "could have",
            "couldn't": "could not",
            "couldn't've": "could not have",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hadn't've": "had not have",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'd've": "he would have",
            "he'll": "he will",
            "he's": "he is",
            "how'd": "how did",
            "how'll": "how will",
            "how's": "how is",
            "i'd": "i would",
            "i'll": "i will",
            "i'm": "i am",
            "i've": "i have",
            "isn't": "is not",
            "it'd": "it would",
            "it'll": "it will",
            "it's": "it is",
            "let's": "let us",
            "ma'am": "madam",
            "mayn't": "may not",
            "might've": "might have",
            "mightn't": "might not",
            "must've": "must have",
            "mustn't": "must not",
            "needn't": "need not",
            "oughtn't": "ought not",
            "shan't": "shall not",
            "sha'n't": "shall not",
            "she'd": "she would",
            "she'll": "she will",
            "she's": "she is",
            "should've": "should have",
            "shouldn't": "should not",
            "that'd": "that would",
            "that's": "that is",
            "there'd": "there had",
            "there's": "there is",
            "they'd": "they would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "wasn't": "was not",
            "we'd": "we would",
            "we'll": "we will",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "where'd": "where did",
            "where's": "where is",
            "who'll": "who will",
            "who's": "who is",
            "won't": "will not",
            "wouldn't": "would not",
            "you'd": "you would",
            "you'll": "you will",
            "you're": "you are"
        }

        # Convert words to lower case
        text = text.lower()

        # Replace contractions with their longer forms
        if True:
            text = text.split()
            new_text = []
            for word in text:
                if word in contractions:
                    new_text.append(contractions[word])
                else:
                    new_text.append(word)
            text = " ".join(new_text)

        # Format words and remove unwanted characters
        text = re.sub(r'&amp;', '', text)
        text = re.sub(r'0,0', '00', text)
        text = re.sub(r'[_"\-;%()|.,+&=*%.,!?:#@\[\]]', ' ', text)
        text = re.sub(r'\'', ' ', text)
        text = re.sub(r'\$', ' $ ', text)
        text = re.sub(r'u s ', ' united states ', text)
        text = re.sub(r'u n ', ' united nations ', text)
        text = re.sub(r'u k ', ' united kingdom ', text)
        text = re.sub(r'j k ', ' jk ', text)
        text = re.sub(r' s ', ' ', text)
        text = re.sub(r' yr ', ' year ', text)
        text = re.sub(r' l g b t ', ' lgbt ', text)
        text = re.sub(r'0km ', '0 km ', text)

        # Optionally, remove stop words
        if remove_stopwords:
            text = text.split()
            stops = set(stopwords.words("english"))
            text = [w for w in text if not w in stops]
            text = " ".join(text)

        return text

    @staticmethod
    def read_webhose_news(directory_in_str: str = 'dataset'):
        news = []
        pathlist = Path(directory_in_str).glob('**/*.json')

        for path in pathlist:
            with open(path) as json_file:
                data = json.load(json_file)
                item = list()
                item.append(data['published'])
                item.append(data['title'])
                news.append(item)

        df = pd.DataFrame(news, columns=['Date', 'Headline'])
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.date
        df = df[~df.duplicated()]
        return df

    @staticmethod
    def read_kaggle_news(directory_in_str: str = 'dataset'):
        df = pd.read_csv(directory_in_str + '/RedditNews.csv', sep=',')
        df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.date
        df['News'] = df['News'].str.strip()
        df['News'] = df['News'].str.replace(r'^b"|"$', '', regex=True)
        df['News'] = df['News'].str.replace(r'^b''|\'$', '', regex=True)
        df['News'] = df['News'].str.replace(r'^|"$', '', regex=True)
        df['News'] = df['News'].str.replace(r'^\'|\'$', '', regex=True)
        df['News'] = df['News'].str.strip()
        df = df[~df.duplicated()]
        df.columns = ['Date', 'Headline']
        return df

    @staticmethod
    def read_crawled_news(directory_in_str: str = 'dataset'):
        pathlist = Path(directory_in_str).glob('**/*.csv')
        df_list = []
        for path in pathlist:
            tmp_df = pd.read_csv(str(path))
            df_list.append(tmp_df)
        df = pd.concat(df_list)
        df = df[['Timestamp', 'Metadata', 'Headline', 'URL', 'ShortDesc']]
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
        df['Date'] = df['Timestamp'].dt.date
        df = df.where((pd.notnull(df)), None)
        df['Metadata'] = df['Metadata'].str.strip()
        df['Headline'] = df['Headline'].str.strip()
        df['URL'] = df['URL'].str.strip()
        df['ShortDesc'] = df['ShortDesc'].str.strip()
        df = df[~df.duplicated()]
        return df