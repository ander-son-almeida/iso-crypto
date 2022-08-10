# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 21:03:32 2022

@author: Anderson Almeida
"""

import pandas as pd, mplfinance as mpf, matplotlib.pyplot as plt
import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import yfinance as yf
import numpy as np
from binance.client import Client
import datetime as dt

data_set = np.load(r'G:\Outros computadores\My Computer\IsoActions\crypto\dataset_crypto\data_set_crypto.npy',
                   allow_pickle=True)

    
media_movel = 14

api_key = 'avnQD2XcghBpIkd5Y3R8PdGsKqwAbWu5A3thoNz56CpbNPIuqJ01y0MBHtCpsALc' 
api_secret = 'cuG1WT5rtA0y6O8bx8T29BQBdV1DvLRb6yIN78HpU3C5LMLtGstVVC78RuPFyZnm'
client = Client(api_key, api_secret)

symbol = "BTCUSDT"
interval= '5m'
Client.KLINE_INTERVAL_5MINUTE 
klines = client.get_historical_klines(symbol, interval, "04 Jul, 2022") #alterar data aqui
data = pd.DataFrame(klines)

 # create colums name
data.columns = ['Datetime','Open', 'High', 'Low', 'Close', 'Volume',
                'close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore']
            
# change the timestamp
data.index = [dt.datetime.fromtimestamp(x/1000.0) for x in data.close_time]

dataset = pd.DataFrame.from_records(data_set)

dataset['Close'] = dataset['Close'].astype(float)
data['Close'] = data['Close'].astype(float)

# normalizando data recente
data['Close'] = data['Close']/dataset['Close'].max()


#calculando a media movel recente do target
data['media_movel'] = data['Close'].rolling(media_movel).mean()
data.dropna(inplace=True)
target = data.to_records(index=True)
target = target['media_movel']


# normalizando e media movel do dataset, apensar do Close
dataset['Close'] = dataset['Close']/dataset['Close'].max()

dataset['media_movel'] = dataset['Close'].rolling(media_movel).mean()
dataset.dropna(inplace=True)




#fazendo isocrona

# determinando o numero de isocronas que vai ser criado para cada ativo
# a isocrona tem o tamanho do target
number_isoch = int(dataset['media_movel'].size/target.size)


#calulando o tamanho da array pra realizar as divisoes da isocrona
# na segunda linhas retiro os numero que sobram
resto = (dataset['media_movel'].size) - (number_isoch*target.size)
media_movel = dataset['media_movel'][:-resto]


#fazendo uma isocrona para cada quatro dias aproximadamente 
isoch = np.array_split(media_movel,number_isoch)

# erro relativo médio
# retorna valores absolutos
# cerne do código
menor_array = np.abs((isoch - target)/target)

#tirando uma média simples dos erros relativos para encontrar a array menor
media = np.mean(menor_array, axis=1)

#selecionando o indice da menor média
# o zero aqui é pq o argsort retorna uma array em ordem
# do menor para o maior, estou pegando o primeiro indice, ou seja
# o menor erro relativo cometido
ind_media_min = int(np.argsort(media)[0]) 

#criando isocrona mais próxima de 3 dias - primeira amostra
isoch_0 = isoch[ind_media_min] # isocronas de 2 dias proximas
isoch_1 = isoch[ind_media_min + 1][:-int(target.size/2)]  #isocrona da previsao, um dia a mais
# se quiser adicionar mais dias de previsão, precio criar mais uma isocrona para ser
# da sequencia
# isoch_2 = isoch[ind_media_min + 2][:-int(target.size/2)] # adicionando mais uma isocrona, foi necessario 
#pq os dias atuais são maiores que a previsão geralmente

#concatenando a isocrona mais proxima, com a de previsao
isoch_final = np.concatenate((isoch_0,isoch_1), axis = 0)
    
    # print(ticker_name, 'deu certo em salvar')
    
# except:
    
#     print('Falha ao criar isócrona...')



# calculando R-quadrado
from sklearn.metrics import r2_score

r2 = r2_score(target, isoch_final[:-int(np.abs(target.size - isoch_final.size))])
# print('coeficiente de correlação: ', r2)

###############################################################################
# graficos

# fazendo uma array do tamanho da isocrona final - grafico
xplot0 = np.linspace(0, isoch_final.size, isoch_final.size)

# fazendo uma array do tamanho do target - grafico
xplot1 = np.linspace(0, target.size, target.size)

# plot da previsao
plor_result = plt.figure(figsize=(12,6))
plt.title(r'Bitcoin - $R^{{2}} = {}$'.format(np.around(r2,decimals=2)))
plt.plot(xplot0,isoch_final, label='isocrona (previsão)')
plt.scatter(xplot0,isoch_final)
plt.plot(xplot1,target, label='target')
plt.scatter(xplot1,target)
plt.legend()

#salvando um arquivo npy da isocrona de previsao e um grafico
np.save(r'G:\Outros computadores\My Computer\IsoActions\crypto\results\result_target_bitcoin.npy',target)
np.save(r'G:\Outros computadores\My Computer\IsoActions\crypto\results\result_bitcoin.npy',isoch_final)
plor_result.savefig(r'G:\Outros computadores\My Computer\IsoActions\crypto\results\bitcoin.png', dpi=300)
        
        


