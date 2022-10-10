# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 14:51:32 2022

@author: Anderson Almeida
"""

import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from binance.client import Client
import datetime as dt


symbol = "BTCUSDT"
interval= '1m'
media_movel = 5

# enter binance API key 
api_key = '?????' 
api_secret = '????'
client = Client(api_key, api_secret)

###############################################################################
# read CSV from predtc
data_set = pd.read_csv(r'\results\result_BTCUSDT_0.8046406774052619.csv')

# renomeando a coluna de data
data_set.rename( columns={'Unnamed: 0':'Date'}, inplace=True )

# renaming the date column
data_set['Date'] = pd.to_datetime(data_set['Date'])

# convert to milliseconds
data_set['Date_ns'] = data_set['Date'].astype(np.int64) / int(1e6)

# converting start and end date to millisecond
start = int(data_set['Date_ns'][0])
end = int(data_set['Date_ns'][-1:])

klines = client.get_historical_klines(symbol, interval, start) 
data_backtest = pd.DataFrame(klines)

# create colums name
data_backtest.columns = ['Date','Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore']
            
# change the timestamp
data_backtest.index = [dt.datetime.fromtimestamp(x/1000.0) for x in data_backtest.close_time]

# moving average
data_backtest['media_movel'] = data_backtest['Close'].rolling(media_movel).mean()

###############################################################################
# simple plot
a = 1.012

plt.figure()
plt.plot(data_set['Date'], data_set['predict'], label='Predição', marker='o')
plt.plot(data_backtest['media_movel']*a, label='backtest', marker='o')
plt.legend()























































