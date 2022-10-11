# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 21:03:32 2022

@author: Anderson Almeida
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from datetime import timedelta
from datetime import datetime
import numpy as np
from binance.client import Client

directory = r'S:\Área de Trabalho\IsoActions\crypto'

symbol = "BTCUSDT" #BTCUSDT ETHUSDT
interval_target = 1 # minute
media_movel = 5
media_movel_cross = 100
target_size = 100 # number of target lines 
condicao = 0.8
count = 0

# enter binance API key 
api_key = '?????' 
api_secret = '????'

data_set = np.load(directory + r'\dataset_crypto\data_set_crypto_{}.npy'.format(symbol),
                   allow_pickle=True)


def crypto(symbol, interval_target, media_movel, data_set, condicao, count, api_key, api_secret):
    
    try:
        
        # dataset
        dataset = pd.DataFrame.from_records(data_set)
        
        # normalizing and moving average of the dataset, only "Close"
        dataset_close_max = dataset['Close'].astype(float).max()
        dataset['Close'] = dataset['Close'].astype(float)
        dataset['Close'] = dataset['Close']/dataset_close_max
        dataset['media_movel'] = dataset['Close'].rolling(media_movel).mean()
        dataset.dropna(inplace=True)
        
        
        ###############################################################################
        # target triggering API binance

        client = Client(api_key, api_secret)
        
        
        interval= '{}m'.format(interval_target) 
        klines = client.get_historical_klines(symbol, 
                                              interval, "{} {}, {}".format(datetime.today().strftime("%d"),
                                                                          datetime.today().strftime("%b"),
                                                                          datetime.today().strftime("%Y")))
        # create dataframe
        target = pd.DataFrame(klines)
        
        # create colums name
        target.columns = ['Datetime','Open', 'High', 'Low', 'Close', 'Volume',
                        'close_time', 'qav','num_trades','taker_base_vol',
                        'taker_quote_vol', 'ignore']
                    
        # change the timestamp
        target.index = [dt.datetime.fromtimestamp(x/1000.0) for x in target.close_time]

        
        #drop do target - mounting target from the last target_size
        N = int(np.abs(target_size - target['Close'].size))
        target = target.iloc[N: , :]
        
        # normalizing according to dataset close max
        target['Close'] = target['Close'].astype(float)
        target['Close'] = target['Close']/dataset_close_max
        
        
        # calculating the moving average of the target
        target['media_movel'] = target['Close'].rolling(media_movel).mean()
        target.dropna(inplace=True)
        
        # converting moving media to array to do the isochrones
        target_media_movel = np.array(target['media_movel'])
        
        ###############################################################################
        # building isochrones
        number_isoch = int(dataset['media_movel'].size/target_media_movel.size)
        
        # to create an integer number of isochrones, you need to take some lines
        resto = (dataset['media_movel'].size) - (number_isoch*target_media_movel.size)
        
        # making several isochrones of the target size
        isoch = np.array_split(dataset['media_movel'][:-resto], number_isoch)
        
        # mean relative error across all isochrones
        menor_array = np.abs((isoch - target_media_movel)/target_media_movel)
        
        # simple media across all isochrones to find the best fit
        # each result of the array is the result of the average of an isochrone
        media = np.mean(menor_array, axis=1)
        
        # selecting the lowest average / best isochrone index
        ind_media_min = int(np.argsort(media)[0]) 
        
        try:
            # creating isochrone closer to target
            isoch_0 = isoch[ind_media_min] 
            isoch_1 = isoch[ind_media_min + 1] # ISCOCHRONE PREDICT
            
            # concatenating the nearest isochron with the forecast
            isoch_final = np.concatenate((isoch_0,isoch_1), axis = 0)
        except:
            print('Error creating isochrone!')
        
        ###############################################################################
        # R-square
        from sklearn.metrics import r2_score
        r2 = r2_score(target_media_movel, isoch_final[:-int(np.abs(target_media_movel.size 
                                                                   - isoch_final.size))])
        
        ###############################################################################
        # Generating Trade Signals using Moving Average(MA) Crossover Strategy
        crossover = pd.DataFrame(isoch_final, columns=['short_window'])
        crossover['long_window'] = crossover['short_window'].rolling(window = media_movel_cross,
                                                                     min_periods = 1).mean()
        crossover['Signal'] = 0.0
        crossover['Signal'] = np.where(crossover['short_window'] > crossover['long_window'], 1.0, 0.0)
        crossover['Position'] = crossover['Signal'].diff()
        

        ###############################################################################
        # graphics
        # xplot datatime from predtc
        # select datatime - index
        data_time_start = target.index[0]
        data_time_end = target.index[-1]
        
        # datatime predict
        periodo = isoch_final.size - target.index.size
        future = timedelta(minutes=interval_target*periodo)   
        data_time_future = data_time_end + future
        data_time_prediction = pd.date_range(start = data_time_start, 
                                             end = data_time_future, 
                                             periods = isoch_final.size)
        
        # plot predict
    
        plot_result, ax = plt.subplots(figsize=(12,6))
        
        ax.set_title(r'BTCUSDT - $R^{{2}} = {}$'.format(np.around(r2,decimals=2)))
        
        # target
        ax.plot(data_time_prediction[data_time_prediction <= data_time_end],
                target['Close']*dataset_close_max,alpha=0.2) #variacao normal target
        
        #moving average target
        ax.plot(target.index,target_media_movel*dataset_close_max, 
                label='Target',alpha=0.5) 
        
        #moving average isochrone
        ax.plot(data_time_prediction,isoch_final*dataset_close_max, 
                label='Isócrona (previsão)',alpha=0.5) 
        
        #moving average long
        ax.plot(data_time_prediction,crossover['long_window']*dataset_close_max, 
                label='{} MA '.format(media_movel_cross),alpha=0.3, color='grey') 
        
        
        #ind buy and sell from predict
        ind = data_time_prediction >= data_time_end
        
        # plot ‘buy’ signals
        ax.plot(data_time_prediction[ind][crossover['Position'][ind] == 1], 
                 crossover['short_window'][ind][crossover['Position'][ind] == 1]*dataset_close_max, 
                 '^', markersize = 8, color = 'g', label = 'buy')
        
        # plot ‘sell’ signals
        ax.plot(data_time_prediction[ind][crossover['Position'][ind] == -1], 
                 crossover['short_window'][ind][crossover['Position'][ind] == -1]*dataset_close_max, 
                 'v', markersize = 8, color = 'r', label = 'sell')
        
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.xaxis.set_minor_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_xlabel('Horário')
        ax.set_ylabel('Valor (dolar)')
        ax.grid(alpha=1.0)
        ax.legend()
        _ = plt.xticks(rotation=45) 
        plt.close(plot_result)
        
        
        #save npy
        
        if r2 > condicao:
            plot_result.savefig(directory + r'\results\{}_{}.png'.format(symbol, r2), dpi=300)
            
            result = pd.DataFrame(isoch_final[ind]*dataset_close_max, index=data_time_prediction[ind], 
                                  columns=['predict'])

            result.to_csv(directory + r'\results\result_{}_{}.csv'.format(symbol, r2),index=True)


        return r2, data_time_end, number_isoch, data_time_start, data_time_prediction

    except:
        print('Adjustment failed!')


while True:
    
    try:
        count = count + 1
        (r2, data_time_end, number_isoch, 
          data_time_start, data_time_prediction) = crypto(symbol, interval_target, media_movel,
                                                          data_set, condicao, count, 
                                                          api_key, api_secret)
        print('\n Ciclo', count, 'de ajuste')
        print('\n Horário inicial do ajuste:', data_time_start)
        print('\n Horário final do ajuste:', data_time_end)
        print('\n Horário final da previsão:', data_time_prediction[-1])
        print('\n Nº Isócronas criadas: ', number_isoch)
        print("\n R-square = ", np.around(r2, decimals=2))
        print('*'*50)
    except:
        print('Loop failure!')
        
