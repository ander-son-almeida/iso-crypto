# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 21:03:32 2022

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
import time
import sys

data_set = np.load(r'S:\Área de Trabalho\IsoActions\crypto\dataset_crypto\data_set_crypto.npy',
                   allow_pickle=True)

interval_target = 1 #minuto
media_movel = 5
target_size = 100 # o numero de linhas do target
count = 0

def crypto(interval_target, media_movel, data_set):
    
    try:
        ###############################################################################
        # dataset
        dataset = pd.DataFrame.from_records(data_set)
        
        # normalizando e media movel do dataset, apenas do Close
        dataset_close_max = dataset['Close'].astype(float).max()
        dataset['Close'] = dataset['Close'].astype(float)
        dataset['Close'] = dataset['Close']/dataset_close_max
        dataset['media_movel'] = dataset['Close'].rolling(media_movel).mean()
        dataset.dropna(inplace=True)
        
        
        ###############################################################################
        # target acionando API da binance
        api_key = 'avnQD2XcghBpIkd5Y3R8PdGsKqwAbWu5A3thoNz56CpbNPIuqJ01y0MBHtCpsALc' 
        api_secret = 'cuG1WT5rtA0y6O8bx8T29BQBdV1DvLRb6yIN78HpU3C5LMLtGstVVC78RuPFyZnm'
        client = Client(api_key, api_secret)
        
        symbol = "BTCUSDT"
        interval= '{}m'.format(interval_target) 
        klines = client.get_historical_klines(symbol, 
                                              interval, "{} {}, {}".format(datetime.today().strftime("%d"),
                                                                                      datetime.today().strftime("%b"),
                                                                                      datetime.today().strftime("%Y")))
        # create dataframe
        target = pd.DataFrame(klines)
        
        # create colums name
        target.columns = ['Datetime','Open', 'High', 'Low', 'Close', 'Volume',
                        'close_time', 'qav','num_trades','taker_base_vol','taker_quote_vol', 'ignore']
                    
        # change the timestamp
        target.index = [dt.datetime.fromtimestamp(x/1000.0) for x in target.close_time]
        
        #drop do target - montando target dos ultimos target_size
        N = int(target_size)
        target = target.iloc[N: , :]
        
        # normalizando conforme dataset close max
        target['Close'] = target['Close'].astype(float)
        target['Close'] = target['Close']/dataset_close_max
        
        #calculando a media movel do target
        target['media_movel'] = target['Close'].rolling(media_movel).mean()
        target.dropna(inplace=True)
        
        # convertendo media movel em array para fazer as isocronas
        target_media_movel = np.array(target['media_movel'])
        
        ###############################################################################
        #fazendo isocrona
        number_isoch = int(dataset['media_movel'].size/target_media_movel.size)
        
        
        # para criar um numero inteiro de isocronas, precisa tirar algumas linhas
        resto = (dataset['media_movel'].size) - (number_isoch*target_media_movel.size)
        
        #fazendo varias isocronas do tamanho do target
        isoch = np.array_split(dataset['media_movel'][:-resto], number_isoch)
        
        # erro relativo médio em todas isocronas
        menor_array = np.abs((isoch - target_media_movel)/target_media_movel)
        
        #media simples em todas as isocronas para encontrar o melhor ajuste
        #cada resultado da array é o resultado da media de uma isocrona
        media = np.mean(menor_array, axis=1)
        
        #selecionando o indice da menor média / melhor isocrona
        ind_media_min = int(np.argsort(media)[0]) 
        
        try:
            #criando isocrona mais próxima do target
            isoch_0 = isoch[ind_media_min] 
            isoch_1 = isoch[ind_media_min + 1] #Isocrona de previsao
            
            #concatenando a isocrona mais proxima, com a de previsao
            isoch_final = np.concatenate((isoch_0,isoch_1), axis = 0)
        except:
            print('Erro ao criar isócrona. Amostra muito grande!')
        
        ###############################################################################
        # calculando R-quadrado
        from sklearn.metrics import r2_score
        r2 = r2_score(target_media_movel, isoch_final[:-int(np.abs(target_media_movel.size - isoch_final.size))])
        
        ###############################################################################
        # graficos
        # fazendo xplot datatime da previsao
        # selecionando o datatime - index
        data_time_start = target.index[0]
        data_time_end = target.index[-1]

        # datatime da previsao
        #atencao: interval_dataset minutos pq é o intervalo fixado ate entao
        periodo = isoch_final.size - target.index.size
        future = timedelta(minutes=interval_target*periodo)   
        data_time_future = data_time_end + future
        data_time_prediction = pd.date_range(start = data_time_start, 
                                             end = data_time_future, 
                                             periods = isoch_final.size)
        
        # plot da previsao
        plot_result, ax = plt.subplots(figsize=(12,6))
        ax.set_title(r'BTCUSDT - $R^{{2}} = {}$'.format(np.around(r2,decimals=2)))
        ax.plot(target.index,target_media_movel*dataset_close_max, label='target',alpha=0.5, marker='.')
        ax.plot(data_time_prediction,isoch_final*dataset_close_max, label='isocrona (previsão)',alpha=0.5, marker='.')
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_xlabel('Horário')
        ax.set_ylabel('Valor (dolar)')
        ax.legend()
        _ = plt.xticks(rotation=45) 
        plt.close(plot_result)
        
        
        #salvando um arquivo npy da isocrona de previsao e um grafico
        # np.save(r'S:\Área de Trabalho\IsoActions\crypto\results\result_target_bitcoin_{}.npy'.format(np.around(r2,decimals=2)),target)
        # np.save(r'S:\Área de Trabalho\IsoActions\crypto\results\result_bitcoin_{}.npy'.format(np.around(r2,decimals=2)),isoch_final)
        if r2 > 0.1:
            plot_result.savefig(r'S:\Área de Trabalho\IsoActions\crypto\results\bitcoin_{}.png'.format(np.around(r2,decimals=2)), dpi=300)
        
        
        return r2, data_time_end, number_isoch, data_time_start, data_time_prediction

    except:
        print('Falha no ajuste!')


while True:
    
    try:
        count = count + 1
        (r2, data_time_end, number_isoch, 
         data_time_start, data_time_prediction) = crypto(interval_target, media_movel, data_set)
        
        print('\n Ciclo', count, 'de ajuste')
        print('\n Horário inicial do ajuste:', data_time_start)
        print('\n Horário final do ajuste:', data_time_end)
        print('\n Horário final da previsão:', data_time_prediction[-1])
        print('\n Nº Isócronas criadas: ', number_isoch)
        print("\n R² = ", np.around(r2, decimals=2))
        print('*'*50)
    except:
        print('Falha no loop')






















