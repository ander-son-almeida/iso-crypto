# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 14:51:32 2022

@author: Anderson Almeida
"""

import pandas as pd, mplfinance as mpf, matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from datetime import timedelta
from datetime import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf
import numpy as np
from binance.client import Client
import datetime as dt
import mplcyberpunk
import time
import sys


symbol = "BTCUSDT"
interval= '1m'
media_movel = 5

# lendo a previsao
data_set = pd.read_csv(r'S:\Área de Trabalho\IsoActions\crypto\results\result_bitcoin_0.8088121151083562.csv')

# renomeando a coluna de data
data_set.rename( columns={'Unnamed: 0':'Date'}, inplace=True )

# convertendo a data em datime
data_set['Date'] = pd.to_datetime(data_set['Date'])

# convertendo em milesegundos
data_set['Date_ns'] = data_set['Date'].astype(np.int64) / int(1e6)

# convertendo data inicio e final em milesegundo
start = int(data_set['Date_ns'][0])
end = int(data_set['Date_ns'][-1:])


api_key = 'avnQD2XcghBpIkd5Y3R8PdGsKqwAbWu5A3thoNz56CpbNPIuqJ01y0MBHtCpsALc' 
api_secret = 'cuG1WT5rtA0y6O8bx8T29BQBdV1DvLRb6yIN78HpU3C5LMLtGstVVC78RuPFyZnm'
client = Client(api_key, api_secret)

klines = client.get_historical_klines(symbol, interval, start) 
data_backtest = pd.DataFrame(klines)

# create colums name
data_backtest.columns = ['Date','Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore']
            
# change the timestamp
data_backtest.index = [dt.datetime.fromtimestamp(x/1000.0) for x in data_backtest.close_time]

#media movel

data_backtest['media_movel'] = data_backtest['Close'].rolling(media_movel).mean()

# # convertendo a data em datime
# data_backtest['Date'] = pd.to_datetime(data_backtest['Date'])


# plot simples
plt.figure()
plt.plot(data_set['Date'], data_set['predict'], label='Predição', marker='o')
plt.plot(data_backtest['media_movel'], label='backtest', marker='o')
plt.legend()























































