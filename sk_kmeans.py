# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import KMeans
from matplotlib import pyplot
from sklearn import preprocessing
import pandas as pd

"""
x = np.array([ [1,2],[1.5,1.8],[5,8],[8,8],[1,0.6],[9,11] ])

clf = KMeans(n_clusters=2)
clf.fit(x)

centers = clf.cluster_centers_
labels = clf.labels_
print(centers)
print(labels)

for i in range(len(labels)):
    pyplot.scatter(x[i][0], x[i][1], c=('r' if labels[i] == 0 else 'b'))
pyplot.scatter(centers[:,0],centers[:,1],marker='*', s=100)

predict = [[2,1], [6,9]]
label = clf.predict(predict)
for i in range(len(label)):
    pyplot.scatter(predict[i][0], predict[i][1], c=('r' if label[i] == 0 else 'b'), marker='x')

pyplot.show()
"""

'''
数据集：titanic.xls(泰坦尼克号遇难者/幸存者名单)
<http://blog.topspeedsnail.com/wp-content/uploads/2016/11/titanic.xls>
***字段***
pclass: 社会阶层(1，精英；2，中产；3，船员)
survived: 是否幸存
name: 名字
sex: 性别
age: 年龄
sibsp: 哥哥姐姐个数
parch: 父母儿女个数
ticket: 船票号
fare: 船票价钱
cabin: 船舱
embarked
boat
body: 尸体
home.dest
******
'''

# 加载数据
df = pd.read_excel('titanic.xls')
#print(df.shape)  (1309, 14)
#print(df.head())
#print(df.tail())
"""
    pclass  survived                                            name     sex  \
0       1         1                    Allen, Miss. Elisabeth Walton  female
1       1         1                   Allison, Master. Hudson Trevor    male
2       1         0                     Allison, Miss. Helen Loraine  female
3       1         0             Allison, Mr. Hudson Joshua Creighton    male
4       1         0  Allison, Mrs. Hudson J C (Bessie Waldo Daniels)  female
    
       age  sibsp  parch  ticket      fare    cabin embarked boat   body  \
0  29.0000      0      0   24160  211.3375       B5        S    2    NaN
1   0.9167      1      2  113781  151.5500  C22 C26        S   11    NaN
2   2.0000      1      2  113781  151.5500  C22 C26        S  NaN    NaN
3  30.0000      1      2  113781  151.5500  C22 C26        S  NaN  135.0
4  25.0000      1      2  113781  151.5500  C22 C26        S  NaN    NaN
    
    home.dest
0                     St Louis, MO
1  Montreal, PQ / Chesterville, ON
2  Montreal, PQ / Chesterville, ON
3  Montreal, PQ / Chesterville, ON
4  Montreal, PQ / Chesterville, ON
"""

# 去掉无用字段
df.drop(['body','name','ticket'], 1, inplace=True)

df.convert_objects(convert_numeric=True)
df.fillna(0, inplace=True)  # 把NaN替换为0

# 把字符串映射为数字，例如{female:1, male:0}
df_map = {}  # 保存映射
cols = df.columns.values
for col in cols:
    if df[col].dtype != np.int64 and df[col].dtype != np.float64:
        temp = {}
        x = 0
        for ele in set(df[col].values.tolist()):
            if ele not in temp:
                temp[ele] = x
                x += 1

        df_map[df[col].name] = temp
        df[col] = list(map(lambda val:temp[val], df[col]))

#for key, value in df_map.iteritems():
#    print(key,value)
#print(df.head())

x = np.array(df.drop(['survived'],1 ).astype(float))
x = preprocessing.scale(x)

clf = KMeans(n_clusters=2)
clf.fit(x)


y = np.array(df['survived'])

correct = 0
for i in range(len(x)):
    predict_data = np.array(x[i].astype(float))
    predict_data = predict_data.reshape(-1, len(predict_data))
    predict = clf.predict(predict_data)
    #print(predict[0], y[i])
    if predict[0] == y[i]:
        correct+=1

print(correct*1.0/len(x))




