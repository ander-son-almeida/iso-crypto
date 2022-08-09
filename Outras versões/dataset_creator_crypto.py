# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 19:39:58 2022

@author: Anderson Almeida
"""

import pandas as pd
from binance.client import Client
import datetime as dt
import numpy as np
# client configuration

media_movel = 14

api_key = 'avnQD2XcghBpIkd5Y3R8PdGsKqwAbWu5A3thoNz56CpbNPIuqJ01y0MBHtCpsALc' 
api_secret = 'cuG1WT5rtA0y6O8bx8T29BQBdV1DvLRb6yIN78HpU3C5LMLtGstVVC78RuPFyZnm'
client = Client(api_key, api_secret)

symbol = "BTCUSDT"
interval= '5m'
Client.KLINE_INTERVAL_5MINUTE 
klines = client.get_historical_klines(symbol, interval, "1 Dec, 2017", "04 Jul, 2022") #atualizar aqui
data = pd.DataFrame(klines)
 # create colums name
data.columns = ['Datetime','Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore']
            
# change the timestamp
data.index = [dt.datetime.fromtimestamp(x/1000.0) for x in data.close_time]


#convert data to float and plot
# data=data.astype(float)
# data["Close"].plot(title = 'DOTUSDT', legend = 'Close')

# normalizando
# data = data/data.max()


#média movel
# data['media_movel'] = data['Close'].rolling(media_movel).mean()
# data.dropna(inplace=True)


data = data.to_records()

#salvando
np.save(r'S:\Área de Trabalho\IsoActions\crypto\dataset_crypto\data_set_crypto.npy',data)
