import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler

from dataloader import *


class ArimaModel:

    def __init__(self):
        """

        """
        self.stock_name = Config.stock_name
        self.stocks = []

    def load_data(self):
        self.stocks = DataLoader.get_stock_data(self.stock_name, Config.from_date, Config.to_date)

    def stock_history(self):
        df = self.stocks

        # setting index as date
        df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
        df.index = df['Date']

        # plotting the target variable
        plt.figure(figsize=(16, 8))
        plt.title(self.stock_name + " Stock History")
        plt.xlabel('Date')
        plt.ylabel("Close Price")
        plt.plot(df['Close'], label='Close Price history')
        #plt.show()
        plt.savefig('./images/' + self.stock_name + 'stock_history' + '.png')

    def create_arima_model(self):
        data = self.stocks.sort_index(ascending=True, axis=0)

        train = data[:1490]
        valid = data[1490:]

        print(train.shape)
        print(valid.shape)

        training = train['Close']
        validation = valid['Close']

        model = auto_arima(training, start_p=1, start_q=1, max_p=3, max_q=3, m=12, start_P=0, seasonal=True, d=1, D=1,
                           trace=True, error_action='ignore', suppress_warnings=True)
        model.fit(training)

        forecast = model.predict(n_periods=321)
        forecast = pd.DataFrame(forecast, index=valid.index, columns=['Prediction'])

        rms = np.sqrt(np.mean(np.power((np.array(valid['Close']) - np.array(forecast['Prediction'])), 2)))
        print("Root Mean Square Error", rms)

        plt.title(self.stock_name + " Stock History")
        plt.plot(train['Close'], color = 'green')
        plt.plot(valid['Close'], color = 'blue')
        plt.plot(forecast['Prediction'], color='red')
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        #plt.show()
        plt.savefig('./images/arima_stock_prediction.png')



if __name__ == "__main__":
    arima = ArimaModel()
    arima.load_data()
    arima.stock_history()
    arima.create_arima_model()