# -*- coding: utf-8 -*-

import numpy as np
from sklearn.cluster import MeanShift
from matplotlib import pyplot
from sklearn import preprocessing
import pandas as pd

'''
数据集：titanic.xls(泰坦尼克号遇难者/幸存者名单)
<http://blog.topspeedsnail.com/wp-content/uploads/2016/11/titanic.xls>
***字段***
pclass: 社会阶层(1，精英；2，中层；3，船员/劳苦大众)
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
目的：使用除survived字段外的数据进行means shift分组,看看能分为几组,哪个字段对生还起决定性作用
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
4       1        G 0  Allison, Mrs. Hudson J C (Bessie Waldo Daniels)  female
    
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

org_df = pd.DataFrame.copy(df)

# 去掉无用字段
df.drop(['body','name'], 1, inplace=True)

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

clf = MeanShift()
clf.fit(x)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

#print(labels)
n_cluster = len(np.unique(labels))
print(n_cluster)
#print(cluster_centers)

org_df['group'] = np.nan
for i in range(len(x)):
    org_df['group'].iloc[i] = labels[i]

survivals = {}
for i in range(n_cluster):
    temp_df = org_df[org_df['group']==float(i)]
    survival_cluster = temp_df[(temp_df['survived']==1)]
    survial = 1.0*len(survival_cluster)/len(temp_df)
    survivals[i] = survial
print(survivals)
#MeanShift自动把数据分成了三组，每组对应的生还率(有时分成4组)：
#{0: 0.37782982045277125, 1: 0.8333333333333334, 2: 0.1}
#你可以详细分析一下org_df, 看看各group的关系
#print(org_df[ org_df['group'] == 2 ])
#print(org_df[ org_df['group'] == 2 ].describe())
#org_df.to_excel('group.xls')

