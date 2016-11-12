import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn import cross_validation
from sklearn import svm
from sklearn.linear_model import LinearRegression

import quandl
import math

import pickle

'''
df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] *100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] *100.0

df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

#print(df.head())
"""
             Adj. Close    HL_PCT  PCT_change  Adj. Volume
Date
2004-08-19   50.322842  3.712563    0.324968   44659000.0
2004-08-20   54.322689  0.710922    7.227007   22834300.0
2004-08-23   54.869377  3.729433   -1.227880   18256100.0
2004-08-24   52.597363  6.417469   -5.726357   15247300.0
2004-08-25   53.164113  1.886792    1.183658    9188600.0
"""
#print(df.tail())

forecast_col = 'Adj. Close'

df.fillna(-99999, inplace=True)
forecast_out = 10 #int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

df.dropna(inplace=True)

X = np.array( df.drop(['label'], 1) )

Y = np.array( df['label'] )

X = preprocessing.scale(X)   

#X = X[:-forecast_out+1]

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

clf = LinearRegression()
#clf = svm.SVR(kernel='ploy')

clf.fit(X_train, Y_train)

with open('stock.module', 'wb') as f:
    pickle.dump(clf, f)
'''
module = open('stock.module', 'rb')
clf = pickle.load(module)

#accuracy = clf.score(X_test, Y_test)

print(clf.predict([ 2.57846406, -0.66790081,  0.21691724, -0.8415952 ]))

#print(accuracy)





