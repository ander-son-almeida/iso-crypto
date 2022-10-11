# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 19:39:58 2022

@author: Anderson Almeida
"""

import pandas as pd
from binance.client import Client
import datetime
from datetime import timedelta
from datetime import datetime
import datetime as dt
import numpy as np


symbol = "BTCUSDT" #BTCUSDT ETHUSDT
interval= '1m'

diretorio = r'S:\Área de Trabalho\IsoActions\crypto'
api_key = '???' 
api_secret = '????'
client = Client(api_key, api_secret)


# read dataset to update
data_set = np.load(diretorio + r'\dataset_crypto\data_set_crypto_{}.npy'.format(symbol),
                   allow_pickle=True)



#aplicando a atualizao

data_start_update = int(data_set['Datetime'][data_set['Datetime'].size - 1])

data_end_update = "{} {}, {}".format((datetime.today() - timedelta(days=1)).strftime("%d"),
                                        (datetime.today() - timedelta(days=1)).strftime("%b"),
                                        (datetime.today() - timedelta(days=1)).strftime("%Y"))


klines = client.get_historical_klines(symbol, interval, data_start_update, data_end_update) 
data_update = pd.DataFrame(klines)

# create colums name
data_update.columns = ['Datetime','Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore']
            
# change the timestamp
data_update.index = [dt.datetime.fromtimestamp(x/1000.0) for x in data_update.close_time]


data_update = data_update.to_records()


update = np.concatenate((data_set, data_update), axis=0)


#salvando
np.save(diretorio + r'\dataset_crypto\data_set_crypto_{}.npy'.format(symbol),update)
