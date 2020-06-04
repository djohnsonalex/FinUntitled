# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 14:31:29 2020

@author: djohn
"""

import requests
import datetime 
import numpy as np
import matplotlib.pyplot as plt  # To visualize
from sklearn.metrics import r2_score
#import pandas as pd

APIkey = 'brakqnfrh5rbgnjpv4r0'

ticker = 'GS' #ticker searching for proxy

#proxy candidates
ticker2 = 'KBE'
ticker3 = 'IYF'
ticker4 = 'KRE'

timelapse = 180 # in days

end_date = datetime.date.today()
start_date = datetime.date.today() + datetime.timedelta(-timelapse)

print(f'End date: {end_date}')
print(f'Start date: {start_date}')
print('')

unix_start = round((datetime.datetime.today() + datetime.timedelta(-timelapse)).timestamp())
unix_end = round(datetime.datetime.today().timestamp())

stock_history = requests.get(f'https://finnhub.io/api/v1/stock/candle?symbol={ticker}&resolution=D&from={unix_start}&to={unix_end}&token={APIkey}')
stock_history2 = requests.get(f'https://finnhub.io/api/v1/stock/candle?symbol={ticker2}&resolution=D&from={unix_start}&to={unix_end}&token={APIkey}')
stock_history3 = requests.get(f'https://finnhub.io/api/v1/stock/candle?symbol={ticker3}&resolution=D&from={unix_start}&to={unix_end}&token={APIkey}')
stock_history4 = requests.get(f'https://finnhub.io/api/v1/stock/candle?symbol={ticker4}&resolution=D&from={unix_start}&to={unix_end}&token={APIkey}')

unix_dates = stock_history.json()['t']
normal_dates = []

for i in unix_dates:
    normal_date = datetime.datetime.fromtimestamp(i).strftime('%Y-%m-%d')
    normal_dates.append(normal_date)
    
close = stock_history.json()["c"]
close2 = stock_history2.json()["c"]
close3 = stock_history3.json()["c"]
close4 = stock_history4.json()["c"]

close_return = []
close2_return = []
close3_return = []
close4_return = []

for i in range(len(close)):
    j = ((close[i]/close[i-1])-1)*100
    z = round(j,4)
    close_return.append(z)

for i in range(len(close2)):
    j = ((close2[i]/close2[i-1])-1)*100
    z = round(j,4)
    close2_return.append(z)
    
for i in range(len(close3)):
    j = ((close3[i]/close3[i-1])-1)*100
    z = round(j,4)
    close3_return.append(z)
    
for i in range(len(close4)):
    j = ((close4[i]/close4[i-1])-1)*100
    z = round(j,4)
    close4_return.append(z)

closebydate = dict(zip(normal_dates, close_return))
closebydate2 = dict(zip(normal_dates, close2_return))
closebydate3 = dict(zip(normal_dates, close3_return))
closebydate4 = dict(zip(normal_dates, close4_return))

print(f"{ticker} daily stock price returns by date:")
print(closebydate)
print('')

#print(f"{ticker2} daily stock price returns by date:")
#print(closebydate2)
#print('')

x = normal_dates
y = close_return
y_2 = close2_return
y_3 = close3_return
y_4 = close4_return

#plt.scatter(x, y)
#plt.scatter(x, y_2)
#plt.xlabel("Date")
#plt.ylabel("Stock Price")

returns_1_2 = dict(zip(close_return, close2_return))
returns_1_3 = dict(zip(close_return, close3_return))
returns_1_4 = dict(zip(close_return, close4_return))

model_1_2 = np.polyfit(y_2,y,1)
r2_1_2 = r2_score(y_2, y)
predict = np.poly1d(model_1_2)
x1_lin_reg = y_2
y2_lin_reg = predict(y_2)
#plt.scatter(y_2, y)
#plt.plot(x1_lin_reg, y2_lin_reg)

model_1_3 = np.polyfit(y_3,y,1)
r2_1_3 = r2_score(y_3, y)
predict = np.poly1d(model_1_3)
x1_lin_reg = y_3
y3_lin_reg = predict(y_3)
#plt.scatter(y_3, y)
#plt.plot(x1_lin_reg, y3_lin_reg)

model_1_4 = np.polyfit(y_4,y,1)
r2_1_4 = r2_score(y_4, y)
predict = np.poly1d(model_1_4)
x1_lin_reg = y_4
y4_lin_reg = predict(y_4)
#plt.scatter(y_4, y)
#plt.plot(x1_lin_reg, y4_lin_reg)

print(f'Regression statistics (i.e., slope and y-intercept) of {ticker} with {ticker2}: {model_1_2}. R-squared: {r2_1_2}.')
print('')
print(f'Regression statistics (i.e., slope and y-intercept) of {ticker} with {ticker3}: {model_1_3}. R-squared: {r2_1_3}.')
print('')
print(f'Regression statistics (i.e., slope and y-intercept) of {ticker} with {ticker4}: {model_1_4}. R-squared: {r2_1_4}.')
print('')

plt.scatter(normal_dates, close_return)
plt.scatter(normal_dates, close2_return)
plt.scatter(normal_dates, close3_return)
plt.scatter(normal_dates, close4_return)
plt.xlabel("Date")
plt.ylabel("Daily Price Change (%)")

reg_dict = {ticker2:[model_1_2[0], r2_1_2], ticker3: [model_1_3[0], r2_1_3], ticker4: [model_1_4[0], r2_1_4]}
print(reg_dict)
print('')

r2_threshold = .6

def proxy(dict_input):
    goodr2dict = {k:v for (k,v) in dict_input.items() if v[1] > r2_threshold}
    reverse_goodr2dict = [(v, k) for k, v in goodr2dict.items()]
    proxy = (max(reverse_goodr2dict)[1])
    return proxy

print(f'{proxy(reg_dict)} is the best proxy for {ticker}.')


