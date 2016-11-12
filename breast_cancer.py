# dataset : https://archive.ics.uci.edu/ml/
"""import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
from sklearn.linear_model import LinearRegression
from sklearn import svm
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
#print(df.head())

#print(df.shape)
df.replace('?', np.nan, inplace=True)  # -99999
df.dropna(inplace=True)
#print(df.shape)

df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
Y = np.array(df['class'])

X_trian,X_test,Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

#clf = LinearRegression()
#clf = neighbors.KNeighborsClassifier()
clf = svm.SVC()
clf.fit(X_trian, Y_train)

accuracy = clf.score(X_test, Y_test)

print(accuracy)

sample = np.array([4,2,1,1,1,2,3,2,1])
print(sample.reshape(1, -1))   # X.reshape(-1, 1)
print(clf.predict(sample.reshape(1, -1)))
"""

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data')
#print(df.head())
#print(df.shape)
df.replace('?', np.nan, inplace=True)  # -99999
df.dropna(inplace=True)
#print(df.shape)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
Y = np.array(df['class'])

X_trian,X_test,Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_trian, Y_train)

accuracy = clf.score(X_test, Y_test)

print(accuracy)

sample = np.array([4,2,1,1,1,2,3,2,1])
sample.reshape(1, -1)
print(clf.predict(sample.reshape(1, -1)))
