# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 19:39:58 2022

@author: Anderson Almeida
"""

import pandas as pd
from binance.client import Client
import datetime as dt
import numpy as np

media_movel = 14

# enter binance API key 
api_key = '?????' 
api_secret = '????'
client = Client(api_key, api_secret)

symbol = "BTCUSDT"
interval= '1m'
Client.KLINE_INTERVAL_5MINUTE 
klines = client.get_historical_klines(symbol, interval, "1 Jan, 2020", "22 Jul, 2022") # choose dataset size!
data = pd.DataFrame(klines)

 # create colums name
data.columns = ['Datetime','Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore']
            
# change the timestamp
data.index = [dt.datetime.fromtimestamp(x/1000.0) for x in data.close_time]
data = data.to_records()

# save npy format
np.save(r'\dataset_crypto\BTCUSDT.npy',data)
