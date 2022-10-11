# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 21:03:32 2022

@author: Anderson Almeida
"""

import pandas as pd
import datetime as dt
from datetime import timedelta
from datetime import datetime
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from binance.client import Client

from config.BinanceClient import BinanceClient


st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    page_icon="✅",
    layout="centered",
)

st.title("Real-Time / Live Data Science Dashboard")

directory = r'S:\Área de Trabalho\IsoActions\crypto'

symbol = "BTCUSDT" #BTCUSDT ETHUSDT
interval_target = 1 # minute
media_movel = 5
media_movel_cross = 100
target_size = 100 # number of target lines 
condicao = 0.8
count = 0

client = BinanceClient.getClient()

data_set = np.load(directory + r'\dataset_crypto\data_set_crypto_{}.npy'.format(symbol),
                   allow_pickle=True)

###############################################################################
# dataset
dataset = pd.DataFrame.from_records(data_set)

# normalizing and moving average of the dataset, only "Close"
dataset_close_max = dataset['Close'].astype(float).max()
dataset['Close'] = dataset['Close'].astype(float)
dataset['Close'] = dataset['Close']/dataset_close_max
dataset['media_movel'] = dataset['Close'].rolling(media_movel).mean()
dataset.dropna(inplace=True)

# creating a single-element container
placeholder = st.empty()

# settings
pd.options.plotting.backend = "plotly"

while True:    
    
    ###############################################################################
    # target triggering API binance  
    
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
        print('Erro ao criar isócrona. Amostra muito grande!')
    
    ###############################################################################
    #  R-square
    from sklearn.metrics import r2_score
    r2 = r2_score(target_media_movel, isoch_final[:-int(np.abs(target_media_movel.size 
                                                               - isoch_final.size))])

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
    
    ###############################################################################

    # target
    target = pd.DataFrame({'target_normal': target['Close']*dataset_close_max}, 
                          index=data_time_prediction[data_time_prediction <= data_time_end])

    #moving average target
    target_media = pd.DataFrame({'target_close': target_media_movel*dataset_close_max}, 
                          index=target.index)

    #moving average isochronea
    iso_media = pd.DataFrame({'isoch': isoch_final*dataset_close_max}, 
                          index=data_time_prediction)

    
    ###############################################################################
    
        
    fig1 = px.line(target, y = 'target_normal', color_discrete_sequence=px.colors.qualitative.Dark2)
    fig1['data'][0]['showlegend'] = True
    fig1['data'][0]['name'] = 'Target normal'
    
    fig2 = px.line(target_media, y = 'target_close', color_discrete_sequence=px.colors.qualitative.Bold)
    fig2['data'][0]['showlegend'] = True
    fig2['data'][0]['name'] = 'Média móvel'

    fig3 = px.line(iso_media, y = 'isoch', color_discrete_sequence=px.colors.qualitative.Set1)
    fig3['data'][0]['showlegend'] = True
    fig3['data'][0]['name'] = 'Isócrona'

    fig = go.Figure(data = fig1.data + fig2.data + fig3.data).update_layout(coloraxis=fig1.layout.coloraxis)
    fig.update_layout(xaxis_title= 'Horário',
                      yaxis_title="Valor (dolar)")
    

    count = count + 1
    
    # save npy
    if r2 > condicao:
        
        ind = data_time_prediction >= data_time_end

        result = pd.DataFrame(isoch_final[ind]*dataset_close_max, index=data_time_prediction[ind], 
                              columns=['predict'])

        result.to_csv(directory + r'\results\result_{}_{}.csv'.format(symbol, r2),index=True)
    
    with placeholder.container():

        fig
        
        container1 = st.container()
        col1, col2, col3 = st.columns(3)
        
        with container1:
            
            
            with col1:
                st.metric(label='Ciclo', value=count)

            with col2:
                st.metric(label="Nº Isócronas criadas", value=number_isoch)
                
            with col3:
                st.metric(label="R-square", value=np.around(r2, decimals=3))
                

        st.write('Horário inicial do ajuste: {}'.format(data_time_start))
        st.write('Horário final do ajuste: {}'.format(data_time_end))
        st.write('Horário final da previsão: {}'.format(data_time_prediction[-1]))
