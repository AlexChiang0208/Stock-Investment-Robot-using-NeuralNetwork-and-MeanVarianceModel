#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 14:43:33 2021

@author: alex_chiang
"""

'''
Step 1 - Features Engineering
The .csv is from yahoo finance(open data)
Using stock price to make my own features
'''

import pandas as pd
import numpy as np

inpu = '/Users/alex_chiang/Documents/GitHub/Stock-Investment-Robo-using-NeuralNetwork-and-MeanVarianceModel/input/'

# Get Adj Close from TWII and first 150 market value stocks in Taiwan
TWII = pd.read_csv(inpu+'TWII_CloseAdj.csv', parse_dates=True, index_col='Date')
TW150 = pd.read_csv(inpu+'TW150_CloseAdj.csv', parse_dates=True, index_col='Date')
ret60D_TW150 = ((TW150 - TW150.shift(60)) / TW150.shift(60)).dropna(how='all')
ret120D_TW150 = ((TW150 - TW150.shift(120)) / TW150.shift(120)).dropna(how='all')

# Calculate four momentum of stock return in 60 days and 120 days
TW150_mean60 = (ret60D_TW150.rolling(60).mean().dropna())
TW150_std60 = (ret60D_TW150.rolling(60).std().dropna())
TW150_skew60 = ret60D_TW150.rolling(60).skew().dropna()
TW150_kurt60 = ret60D_TW150.rolling(60).kurt().dropna()

TW150_mean120 = (ret120D_TW150.rolling(60).mean().dropna())
TW150_std120 = (ret120D_TW150.rolling(60).std().dropna())
TW150_skew120 = ret120D_TW150.rolling(60).skew().dropna()
TW150_kurt120 = ret120D_TW150.rolling(60).kurt().dropna()

# Get params of t-1
TW150_mean60_sft1 = TW150_mean60.shift(1).dropna(how='all')
TW150_std60_sft1 = TW150_std60.shift(1).dropna(how='all')
TW150_skew60_sft1 = TW150_skew60.shift(1).dropna(how='all')
TW150_kurt60_sft1 = TW150_kurt60.shift(1).dropna(how='all')
TW150_mean120_sft1 = TW150_mean120.shift(1).dropna(how='all')
TW150_std120_sft1 = TW150_std120.shift(1).dropna(how='all')
TW150_skew120_sft1 = TW150_skew120.shift(1).dropna(how='all')
TW150_kurt120_sft1 = TW150_kurt120.shift(1).dropna(how='all')

# Get y_predict: stock return after 3 month
predict60D_TW150 = ((TW150 - TW150.shift(60)) / TW150.shift(60)).dropna(how='all')
predict60D_TW150 = predict60D_TW150.shift(-60).dropna(how='all')

# Make a same timeline from 2004-9-10 to 2020-10-7
TW150_mean60 = TW150_mean60['2004-9-10':'2020-10-7']
TW150_std60 = TW150_std60['2004-9-10':'2020-10-7']
TW150_skew60 = TW150_skew60['2004-9-10':'2020-10-7']
TW150_kurt60 = TW150_kurt60['2004-9-10':'2020-10-7']
TW150_mean120 = TW150_mean120['2004-9-10':'2020-10-7']
TW150_std120 = TW150_std120['2004-9-10':'2020-10-7']
TW150_skew120 = TW150_skew120['2004-9-10':'2020-10-7']
TW150_kurt120 = TW150_kurt120['2004-9-10':'2020-10-7']
TW150_mean60_sft1 = TW150_mean60_sft1['2004-9-10':'2020-10-7']
TW150_std60_sft1 = TW150_std60_sft1['2004-9-10':'2020-10-7']
TW150_skew60_sft1 = TW150_skew60_sft1['2004-9-10':'2020-10-7']
TW150_kurt60_sft1 = TW150_kurt60_sft1['2004-9-10':'2020-10-7']
TW150_mean120_sft1 = TW150_mean120_sft1['2004-9-10':'2020-10-7']
TW150_std120_sft1 = TW150_std120_sft1['2004-9-10':'2020-10-7']
TW150_skew120_sft1 = TW150_skew120_sft1['2004-9-10':'2020-10-7']
TW150_kurt120_sft1 = TW150_kurt120_sft1['2004-9-10':'2020-10-7']
predict60D_TW150 = predict60D_TW150['2004-9-10':'2020-10-7']

# Put all X_features and y_predict in a big DataFrame
def frame_trans(df, col):
    name = 'trans_' + col
    globals()[name] = df.unstack().to_frame()
    globals()[name].index.rename('stock_id', level=0, inplace=True)
    globals()[name].rename(columns={0:col}, inplace=True)
    return

frame_trans(TW150_mean60, 'mean60')
frame_trans(TW150_std60, 'std60')
frame_trans(TW150_skew60, 'skew60')
frame_trans(TW150_kurt60, 'kurt60')
frame_trans(TW150_mean120, 'mean120')
frame_trans(TW150_std120, 'std120')
frame_trans(TW150_skew120, 'skew120')
frame_trans(TW150_kurt120, 'kurt120')
frame_trans(TW150_mean60_sft1, 'mean60_sft1')
frame_trans(TW150_std60_sft1, 'std60_sft1')
frame_trans(TW150_skew60_sft1, 'skew60_sft1')
frame_trans(TW150_kurt60_sft1, 'kurt60_sft1')
frame_trans(TW150_mean120_sft1, 'mean120_sft1')
frame_trans(TW150_std120_sft1, 'std120_sft1')
frame_trans(TW150_skew120_sft1, 'skew120_sft1')
frame_trans(TW150_kurt120_sft1, 'kurt120_sft1')
frame_trans(predict60D_TW150, 'predict60D')

# Concat them, and check there is a nan or not
df = pd.concat([trans_mean60, trans_mean60_sft1, trans_mean120, 
                 trans_mean120_sft1, trans_std60, trans_std60_sft1, 
                 trans_std120, trans_std120_sft1, trans_skew60, 
                 trans_skew60_sft1, trans_skew120, trans_skew120_sft1, 
                 trans_kurt60, trans_kurt60_sft1, trans_kurt120,
                 trans_kurt120_sft1, trans_predict60D], axis=1)

df.isnull().values.any()
df.to_pickle(inpu+'features.pkl')
#%%

'''
Step 2 - Modeling(Training, evaluation and store it)
Install any package to your python if you don't have
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras import layers
import keras
import xlwings as xw

inpu = '/Users/alex_chiang/Documents/GitHub/Stock-Investment-Robo-using-NeuralNetwork-and-MeanVarianceModel/input/'
outpu = '/Users/alex_chiang/Documents/GitHub/Stock-Investment-Robo-using-NeuralNetwork-and-MeanVarianceModel/output/'

# Read my features and other data
TWII = pd.read_csv(inpu+'TWII_CloseAdj.csv', parse_dates=True, index_col='Date')
TW150 = pd.read_csv(inpu+'TW150_CloseAdj.csv', parse_dates=True, index_col='Date')
df = pd.read_pickle(inpu+'features.pkl')
stock_id = list(TW150.columns)
stock_name = [i.replace('.TW','') for i in stock_id]

# Using for-loop to model each of stock
# This step need VERY LONG TIME!!
writer = pd.ExcelWriter(outpu+'DNN60days_return_prediction.xlsx', engine='openpyxl')
for da,i in zip(stock_id, stock_name):
    
    data = df.loc[da]
    
    # Split into training and testing data
    Data_train_X = data[:'2015-12-31'].drop('predict60D', axis=1)
    Data_train_y = data[:'2015-12-31'][['predict60D']]
    Data_test_X = data['2016-1-4':].drop('predict60D', axis=1)
    Data_test_y = data['2016-1-4':][['predict60D']]

    # Features scaling: StandardScaler or MinMaxScaler
    # You can also try to scale y_predict, but I didn't
    scaler = MinMaxScaler()
    scaler_trainX = scaler.fit(Data_train_X)

    Data_train_X_scaled = scaler_trainX.transform(Data_train_X)
    Data_train_X_scaled = pd.DataFrame(Data_train_X_scaled, 
                                            index=Data_train_X.index, 
                                            columns=Data_train_X.columns)

    Data_test_X_scaled = scaler_trainX.transform(Data_test_X)
    Data_test_X_scaled = pd.DataFrame(Data_test_X_scaled, 
                                      index=Data_test_X.index, 
                                      columns=Data_test_X.columns)

    # Modeling
    model = Sequential()
    model.add(layers.Dense(128, activation='relu',input_dim=(16)))
    model.add(layers.Dense(112, activation='relu'))
    model.add(layers.Dense(86, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(48, activation='relu'))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    
    # Change an optimizers' learning rate
    adam = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=adam, loss='mean_squared_error')
    
    # Train 300 times
    # Training data split 20% to validate it
    history = model.fit(Data_train_X_scaled, Data_train_y, 
              epochs=300, batch_size=30, validation_split=0.2, shuffle=False)
    model.save(outpu+'stock_model/'+i+'_model.h5')
    
    # Put result in a DataFrame
    true = Data_test_y
    true = true.rename(columns={'predict60D':'y_true'})
    predict = pd.DataFrame(model.predict(Data_test_X_scaled), 
                           index=Data_test_y.index, columns=Data_test_y.columns)
    predict = predict.rename(columns={'predict60D':'y_predict'})
    df_pre = pd.concat([true,predict], axis=1)

    df_pre.to_excel(writer, sheet_name=i+'_prediction', engine='openpyxl')
    writer.save()

    # Make pictures to observe result
    fig1 = plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], color='coral', label='loss')
    plt.plot(history.history['val_loss'], color='royalblue', label='val_loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(outpu+'pic1/'+i+'_loss.png')

    fig2 = plt.figure(figsize=(15,8))
    plt.plot(true, label="y_true")
    plt.plot(predict, label="y_predict", linewidth=0.8)
    plt.title('Prediction_3MonthReturn')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.grid(True)
    plt.savefig(outpu+'pic2/'+i+'_predict.png')
 
# Using xlwings to upload pictures into excel
st = ['st_'+i for i in stock_name]
workbook = xw.Book(outpu+'DNN60days_return_prediction.xlsx')  
for i,j in zip(stock_name, st):
 
    globals()[j] = workbook.sheets(i+'_prediction')
    globals()[j].pictures.add(outpu+'pic1/'+i+'_loss.png', name=i+'_loss', update=True, left=globals()[j].range('E1').left)
    globals()[j].pictures.add(outpu+'pic2/'+i+'_predict.png', name=i+'_predict', update=True, left=globals()[j].range('V1').left)

workbook.save()
workbook.close()
#%%

'''
Step 3 - Make a portfolio with Markowitz's MV model
Each stocks percentage & Number of stocks are adjustable
Default - Max percentage: 10% ; Number of stock: 10
'''

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import xlwings as xw
import scipy.optimize as sco

inpu = '/Users/alex_chiang/Documents/GitHub/Stock-Investment-Robo-using-NeuralNetwork-and-MeanVarianceModel/input/'
outpu = '/Users/alex_chiang/Documents/GitHub/Stock-Investment-Robo-using-NeuralNetwork-and-MeanVarianceModel/output/'

TWII = pd.read_csv(inpu+'TWII_CloseAdj.csv', parse_dates=True, index_col='Date')
TW150 = pd.read_csv(inpu+'TW150_CloseAdj.csv', parse_dates=True, index_col='Date')
ret_TW150 = (((TW150 - TW150.shift(1)) / TW150.shift(1)).dropna(how='all'))*240

# Read the result in excel
xls = pd.ExcelFile(outpu+'DNN60days_return_prediction.xlsx')
sheet_name = xls.sheet_names

data = pd.read_excel(outpu+'DNN60days_return_prediction.xlsx', sheet_name=None)

# Put result in a DataFrame
df_predict = pd.DataFrame([])
for i in sheet_name:
    a_sheet = data[i]
    a_sheet.index = a_sheet['Date']
    a_sheet.index = pd.to_datetime(a_sheet.index)
    a_sheet = a_sheet[['y_predict']]
    a_sheet.rename(columns={'y_predict':i.replace('_prediction','.TW')}, inplace = True)
    df_predict = pd.concat([df_predict, a_sheet], axis=1)

# Get timeline per 60 days
df_predict = df_predict.iloc[range(0, 1160, 60)]

# Markowitz's MV model
def Portfolio_volatility(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))    
    return std

def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type':'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0,0.1) ## Max percentage
    bounds = tuple(bound for asset in range(num_assets))
    
    result = sco.minimize(Portfolio_volatility, num_assets*[1/num_assets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# Calculate Portfolio in Mean-Variance Model
# 1000 dollars in the beginning
value = {'Portfolio':[1000], 'TWII':[1000]}
for i in range(len(df_predict)):
    
    # My_portfolio
    buy_stock = df_predict.iloc[i].rank(ascending=False).sort_values()[:10].index.tolist() ## Put N stocks into portfolio

    t1 = TW150.index.get_loc(df_predict.iloc[[i]].index[-1])
    t2 = ret_TW150.index.get_loc(df_predict.iloc[[i]].index[-1])
    mean_return = TW150.iloc[t1-1-240:t1-1][buy_stock].mean()
    cov_matrix = ret_TW150.iloc[t2-1-240:t2-1][buy_stock].cov()
    w = min_variance(mean_return, cov_matrix)['x']

    tarprice = TW150.iloc[t1:t1+60][buy_stock] 
    buy_price = tarprice.iloc[0] 
    sell_price = tarprice.iloc[-1]

    units = (value['Portfolio'][i] * w) / buy_price
    spread = sell_price - buy_price
    value['Portfolio'].append(value['Portfolio'][i] + (spread * units).sum())

    # TWII
    t3 = TWII.index.get_loc(df_predict.iloc[[i]].index[-1])
    V_TWII = TWII.iloc[t3:t3+60]
    TWII_buy = V_TWII.iloc[0]
    TWII_sell = V_TWII.iloc[-1]

    U_TWII = float(value['TWII'][i] / TWII_buy)
    Spread_TWII = float(TWII_sell - TWII_buy)
    value['TWII'].append(value['TWII'][i] + Spread_TWII * U_TWII)

idx = list(df_predict.index)
unique_index = pd.Index(list(TWII.index))
t4 = unique_index.get_loc(df_predict.index[-1])
idx.append(TWII.iloc[[t4+60]].index[0])

portfolio = pd.DataFrame(value, index=idx)

# Calculate IRR, Sigma, and Sharpe Ratio
# Compare with TWII
irr_p = np.round(((portfolio[['Portfolio']].iloc[-1]/portfolio[['Portfolio']].iloc[0])**(1/5)-1)[0], 4)
sigma_p = np.round((portfolio[['Portfolio']].pct_change().dropna().std()[0])*np.sqrt(4), 4)
SharpeRatio_p = round(irr_p / sigma_p, 4)

irr_tw = np.round(((portfolio[['TWII']].iloc[-1]/portfolio[['TWII']].iloc[0])**(1/5)-1)[0], 4)
sigma_tw = np.round((portfolio[['TWII']].pct_change().dropna().std()[0])*np.sqrt(4), 4)
SharpeRatio_tw = round(irr_tw / sigma_tw, 4)

# Make pictures to observe result
fig = plt.figure(figsize=(15,8))
plt.plot(portfolio)
plt.title('Portfolio_DNN60D_strategy')
plt.xlabel('Date')
plt.ylabel('Return')
plt.grid(True)
plt.savefig(outpu+'portfolio.png')
portfolio.to_excel(outpu+'Portfolio.xlsx')

# Using xlwings to upload pictures into excel
workbook = xw.Book(outpu+'Portfolio.xlsx')
sheet = workbook.sheets('Sheet1')
sheet.name = 'portfolio'

sheet.range('E1').value = ['DNN60D_strategy', 'TWII']
sheet.range('D2').value = ['IRR', irr_p, irr_tw]
sheet.range('D3').value = ['sigma', sigma_p, sigma_tw]
sheet.range('D4').value = ['sharpe ratio', SharpeRatio_p, SharpeRatio_tw]
sheet.pictures.add(outpu+'portfolio.png', name='portfolio', update=True, left=sheet.range('H1').left)

workbook.save()
workbook.close()
#%%

'''
Step 4 - Start investing!!!
Get today's stock id with percentage
'''

import pandas as pd
import numpy as np
from pandas_datareader import DataReader as dr
import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import scipy.optimize as sco
import keras

inpu = '/Users/alex_chiang/Documents/GitHub/Stock-Investment-Robo-using-NeuralNetwork-and-MeanVarianceModel/input/'
outpu = '/Users/alex_chiang/Documents/GitHub/Stock-Investment-Robo-using-NeuralNetwork-and-MeanVarianceModel/output/'
TW150 = pd.read_csv(inpu+'TW150_CloseAdj.csv', parse_dates=True, index_col='Date')
stock_id = TW150.columns.tolist()

# start: One year before
# end: today
start = datetime.datetime(2020,1,27)
end = datetime.datetime(2021,1,27)

# Waiting for about 7 minutes
# Download stock price
stock = dr(stock_id, 'yahoo', start, end)['Adj Close'] 
stock.index = pd.to_datetime(stock.index)

# Delete stock if the data is not completed
stock.drop('1216.TW', axis=1, inplace=True)

# Calculate params and upload model
def Calculate(stock, today):
    
    global ret_prediction, ret_TW150
    
    ret_TW150 = (((stock - stock.shift(1)) / stock.shift(1)).dropna(how='all'))*240
    ret60D_TW150 = ((stock - stock.shift(60)) / stock.shift(60)).dropna(how='all')
    ret120D_TW150 = ((stock - stock.shift(120)) / stock.shift(120)).dropna(how='all')

    # Input params
    TW150_mean60 = ret60D_TW150.rolling(60).mean().stack().xs(today).to_frame()
    TW150_mean60.rename(columns={0:'mean60'}, inplace=True)

    TW150_std60 = ret60D_TW150.rolling(60).std().stack().xs(today).to_frame()
    TW150_std60.rename(columns={0:'std60'}, inplace=True)

    TW150_skew60 = ret60D_TW150.rolling(60).skew().stack().xs(today).to_frame()
    TW150_skew60.rename(columns={0:'skew60'}, inplace=True)

    TW150_kurt60 = ret60D_TW150.rolling(60).kurt().stack().xs(today).to_frame()
    TW150_kurt60.rename(columns={0:'kurt60'}, inplace=True)

    TW150_mean120 = ret120D_TW150.rolling(60).mean().stack().xs(today).to_frame()
    TW150_mean120.rename(columns={0:'mean120'}, inplace=True)

    TW150_std120 = ret120D_TW150.rolling(60).std().stack().xs(today).to_frame()
    TW150_std120.rename(columns={0:'std120'}, inplace=True)

    TW150_skew120 = ret120D_TW150.rolling(60).skew().stack().xs(today).to_frame()
    TW150_skew120.rename(columns={0:'skew120'}, inplace=True)

    TW150_kurt120 = ret120D_TW150.rolling(60).kurt().stack().xs(today).to_frame()
    TW150_kurt120.rename(columns={0:'kurt120'}, inplace=True)

    TW150_mean60_sft1 = ret60D_TW150.rolling(60).mean().shift(1).stack().xs(today).to_frame()
    TW150_mean60_sft1.rename(columns={0:'mean60_sft1'}, inplace=True)

    TW150_std60_sft1 = ret60D_TW150.rolling(60).std().shift(1).stack().xs(today).to_frame()
    TW150_std60_sft1.rename(columns={0:'std60_sft1'}, inplace=True)

    TW150_skew60_sft1 = ret60D_TW150.rolling(60).skew().shift(1).stack().xs(today).to_frame()
    TW150_skew60_sft1.rename(columns={0:'skew60_sft1'}, inplace=True)

    TW150_kurt60_sft1 = ret60D_TW150.rolling(60).kurt().shift(1).stack().xs(today).to_frame()
    TW150_kurt60_sft1.rename(columns={0:'kurt60_sft1'}, inplace=True)

    TW150_mean120_sft1 = ret120D_TW150.rolling(60).mean().shift(1).stack().xs(today).to_frame()
    TW150_mean120_sft1.rename(columns={0:'mean120_sft1'}, inplace=True)

    TW150_std120_sft1 = ret120D_TW150.rolling(60).std().shift(1).stack().xs(today).to_frame()
    TW150_std120_sft1.rename(columns={0:'std120_sft1'}, inplace=True)

    TW150_skew120_sft1 = ret120D_TW150.rolling(60).skew().shift(1).stack().xs(today).to_frame()
    TW150_skew120_sft1.rename(columns={0:'skew120_sft1'}, inplace=True)

    TW150_kurt120_sft1 = ret120D_TW150.rolling(60).kurt().shift(1).stack().xs(today).to_frame()
    TW150_kurt120_sft1.rename(columns={0:'kurt120_sft1'}, inplace=True)

    # Concat them
    params = pd.concat([TW150_mean60, TW150_mean60_sft1, TW150_mean120, 
                     TW150_mean120_sft1, TW150_std60, TW150_std60_sft1, 
                     TW150_std120, TW150_std120_sft1, TW150_skew60, 
                     TW150_skew60_sft1, TW150_skew120, TW150_skew120_sft1, 
                     TW150_kurt60, TW150_kurt60_sft1, TW150_kurt120,
                     TW150_kurt120_sft1], axis=1)

    # Scaling
    s_id = params.index.tolist()
    s_scale = ['scale_'+i.replace('.TW','') for i in s_id]
    df = pd.read_pickle(inpu+'features.pkl')
    params_scale = pd.DataFrame()

    for i,j in zip(s_id, s_scale):
        data = df.loc[i]
        Data_train_X = data[:'2015-12-31'].drop('predict60D', axis=1)
        Data_train_y = data[:'2015-12-31'][['predict60D']]
        scaler = MinMaxScaler()
        scaler_trainX = scaler.fit(Data_train_X)

        globals()[j] = scaler_trainX.transform(params.loc[[i]])
        globals()[j] = pd.DataFrame(globals()[j], index=params.loc[[i]].index, columns=params.loc[[i]].columns)
        params_scale = pd.concat([params_scale, globals()[j]])

    # Upload model
    md = ['model_'+i.replace('.TW','') for i in s_id]
    md2 = [i.replace('.TW','')+'_model' for i in s_id]

    for i,j in zip(md, md2):
        globals()[i] = keras.models.load_model(outpu+'stock_model/'+j+'.h5')

    # Predict result
    pre = []
    for i,j in zip(md, s_id):
        pre.append(globals()[i].predict(params_scale.loc[[j]])[0][0])

    ret_prediction = pd.DataFrame([s_id, pre], index=['stock_id','future_return'])
    ret_prediction.columns = ret_prediction.loc['stock_id']
    ret_prediction = ret_prediction.loc[['future_return']]
    ret_prediction = ret_prediction.stack().sort_values(ascending=False).to_frame().xs('future_return',level=0)
    ret_prediction.columns = ['future_return']
    return

Calculate(stock=stock, today='2021-1-27')

# Get investing percentage of stock
def Invest_Percentage(stock, today, num, max_percent):
    
    # MV model
    def Portfolio_volatility(weights, mean_returns, cov_matrix):
        returns = np.sum(mean_returns * weights)
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))    
        return std

    def min_variance(mean_returns, cov_matrix):
        num_assets = len(mean_returns)
        args = (mean_returns, cov_matrix)
        constraints = ({'type':'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = (0,max_percent)
        bounds = tuple(bound for asset in range(num_assets))
        result = sco.minimize(Portfolio_volatility, num_assets*[1/num_assets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
        return result

    # Get portfolio
    buy_stock = ret_prediction.head(num).index.tolist()
    t1 = stock.index.get_loc(today)
    t2 = ret_TW150.index.get_loc(today)
    mean_return = TW150.iloc[t1-1-240:t1-1][buy_stock].mean()
    cov_matrix = ret_TW150.iloc[t2-1-240:t2-1][buy_stock].cov()
    w = min_variance(mean_return, cov_matrix)['x']

    globals()['Portfolio_'+today.replace('-','')] = pd.DataFrame(w, index=buy_stock)
    globals()['Portfolio_'+today.replace('-','')].rename(columns={0:'percentage'}, inplace=True)
    return globals()['Portfolio_'+today.replace('-','')]

Invest_Percentage(stock=stock, today='2021-1-27', num=10, max_percent=0.1)
#%%

'''
Step 5 - Simulate performance of 59 start day
Just a testing for each time's performance
'''

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import xlwings as xw

inpu = '/Users/alex_chiang/Documents/GitHub/Stock-Investment-Robo-using-NeuralNetwork-and-MeanVarianceModel/input/'
outpu = '/Users/alex_chiang/Documents/GitHub/Stock-Investment-Robo-using-NeuralNetwork-and-MeanVarianceModel/output/'
xls = pd.ExcelFile(outpu+'DNN60days_return_prediction.xlsx')
sheet_name = xls.sheet_names

data = pd.read_excel(outpu+'DNN60days_return_prediction.xlsx', sheet_name=None)

# Put result in a DataFrame
df_predict = pd.DataFrame([])
for i in sheet_name:
    a_sheet = data[i]
    a_sheet.index = a_sheet['Date']
    a_sheet.index = pd.to_datetime(a_sheet.index)
    a_sheet = a_sheet[['y_predict']]
    a_sheet.rename(columns={'y_predict':i.replace('_prediction','.TW')}, inplace = True)
    df_predict = pd.concat([df_predict, a_sheet], axis=1)

TWII = pd.read_csv(inpu+'TWII_CloseAdj.csv', parse_dates=True, index_col='Date')
TW150 = pd.read_csv(inpu+'TW150_CloseAdj.csv', parse_dates=True, index_col='Date')
ret_TW150 = (((TW150 - TW150.shift(1)) / TW150.shift(1)).dropna(how='all'))*240

# Mean-Variance Model
import scipy.optimize as sco

def Portfolio_volatility(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns * weights)
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))    
    return std

def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type':'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0,0.1)
    bounds = tuple(bound for asset in range(num_assets))
    
    result = sco.minimize(Portfolio_volatility, num_assets*[1/num_assets], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

# Calculate Portfolio in Mean-Variance Model
grade = ['grade_'+str(i) for i in range(1,60)]
irr = ['irr_'+str(i) for i in range(1,60)]
sigma = ['sigma_'+str(i) for i in range(1,60)]
SharpeRatio = ['SharpeRatio_'+str(i) for i in range(1,60)]

for a,b,c,d,e in zip(range(0,59), grade, irr, sigma, SharpeRatio):

    df_pre = df_predict.iloc[range(a, 1159, 60)]
    value = {'Portfolio':[1000], 'TWII':[1000]}
    for i in range(len(df_pre)):
    
        # TW_portfolio
        buy_stock = df_pre.iloc[i].rank(ascending=False).sort_values()[:10].index.tolist()

        t1 = TW150.index.get_loc(df_pre.iloc[[i]].index[-1])
        t2 = ret_TW150.index.get_loc(df_pre.iloc[[i]].index[-1])
        mean_return = TW150.iloc[t1-1-240:t1-1][buy_stock].mean()
        cov_matrix = ret_TW150.iloc[t2-1-240:t2-1][buy_stock].cov()
        w = min_variance(mean_return, cov_matrix)['x']

        tarprice = TW150.iloc[t1:t1+60][buy_stock] 
        buy_price = tarprice.iloc[0] 
        sell_price = tarprice.iloc[-1]

        units = (value['Portfolio'][i] * w) / buy_price
        spread = sell_price - buy_price
        value['Portfolio'].append(value['Portfolio'][i] + (spread * units).sum())

        # TWII
        t3 = TWII.index.get_loc(df_pre.iloc[[i]].index[-1])
        V_TWII = TWII.iloc[t3:t3+60]
        TWII_buy = V_TWII.iloc[0]
        TWII_sell = V_TWII.iloc[-1]

        U_TWII = float(value['TWII'][i] / TWII_buy)
        Spread_TWII = float(TWII_sell - TWII_buy)
        value['TWII'].append(value['TWII'][i] + Spread_TWII * U_TWII)    

    idx = list(df_pre.index)
    unique_index = pd.Index(list(TWII.index))
    t4 = unique_index.get_loc(df_pre.index[-1])
    idx.append(TWII.iloc[[t4+60]].index[0])

    globals()[b] = pd.DataFrame(value, index=idx)
    globals()[c] = np.round(((globals()[b][['Portfolio']].iloc[-1]/globals()[b][['Portfolio']].iloc[0])**(1/5)-1)[0], 4)
    globals()[d] = np.round((globals()[b][['Portfolio']].pct_change().dropna().std()[0])*np.sqrt(4), 4)
    globals()[e] = round(globals()[c] / globals()[d], 4)

# Make a DataFrame for comparation
grade = ['grade_'+str(i) for i in range(1,60)]
irr = ['irr_'+str(i) for i in range(1,60)]
sigma = ['sigma_'+str(i) for i in range(1,60)]
SharpeRatio = ['SharpeRatio_'+str(i) for i in range(1,60)]
rt_irr = []
rt_sigma = []
rt_sharperatio = []

for c,d,e in zip(irr, sigma, SharpeRatio):
    rt_irr.append(globals()[c])
    rt_sigma.append(globals()[d])
    rt_sharperatio.append(globals()[e])
    
result = pd.DataFrame([rt_irr,rt_sigma,rt_sharperatio], columns = grade).transpose()
result.columns=['irr','sigma','SharpeRatio']
result.loc['average'] = result.mean()

# Show in picture
for i in grade:
    fig = plt.figure(figsize=(15,8))
    plt.plot(globals()[i])
    plt.title('Portfolio_DNN60D_strategy')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.grid(True)
    plt.savefig(outpu+'pic_simulation/'+i+'.png')

grade.insert(0, 'result')
writer = pd.ExcelWriter(outpu+'Portfolio_simulation.xlsx')
for i in grade:
    globals()[i].to_excel(writer, sheet_name=i) 
writer.save() 

# Using xlwings to upload pictures into excel
grade = ['grade_'+str(i) for i in range(1,60)]
workbook = xw.Book(outpu+'portfolio_simulation.xlsx')

for b,c,d,e in zip(grade, irr, sigma, SharpeRatio):
    sheet = workbook.sheets(b)
    sheet.range('F1').value = ['DNN60D_strategy']
    sheet.range('E2').value = ['IRR', globals()[c]]
    sheet.range('E3').value = ['sigma', globals()[d]]
    sheet.range('E4').value = ['sharpe ratio', globals()[e]]
    sheet.pictures.add(outpu+'pic_simulation/'+b+'.png', name='portfolio', update=True, left=sheet.range('I1').left)
    workbook.save()
workbook.close()