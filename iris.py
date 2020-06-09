#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.datasets import load_iris # scikit-learn 의 샘플 데이터 로드
import pandas as pd # 데이터 프레임으로 변환을 위해 임포트
import numpy as np # 연산을 위해 임포트

#시각화를 위한 패키지 임포트
import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris() # sample data load

print(iris) #로드된 데이터가 속성-스타일 접근을 제공하는 딕셔너리와 번치 객체로 표현된 것을 확인
print(iris.DESCR) # Description 속성을 이용해서 데이터셋의 정보를 확인

#각 key에 저장된 value 확인
#feature
print(iris.data)
print(iris.feature_names)

# label
print(iris.target)
print(iris.target_names)

# feature_names 와 target을 레코드로 갖는 데이터프레임 생성
df = pd.DataFrame(data=iris.data,columns=iris.feature_names)
df['target']=iris.target

# 0.0, 1.0, 2.0으로 표현된 label을 문자열로 매핑
df['target'] = df['target'].map({0.:"setosa", 1:"versicolor", 2:"virginica"})
print(df)

#슬라이싱을 통해 feature와 label 분리
x_data = df.iloc[:,:-1]
y_data = df.iloc[:,[-1]]

sns.pairplot(df, x_vars=["sepal length (cm)"], y_vars=["sepal width (cm)"], hue="target", height=5)


# In[2]:


#(80:20)으로 데이터 분할
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)

#분류
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

X=iris.data
y=iris.target

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

#예측
y_pred = knn.predict(X_test)
scores = metrics.accuracy_score(y_test, y_pred)
print(scores)


# In[3]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X, y)

#0 = setosa, 1=versicolor, 2=virginica
classes = {0:'setosa',1:'versicolor',2:'virginica'}

# 아직 보지 못한 새로운 데이터를 제시해보자.
x_new = [[3,4,5,2],
         [5,4,2,2]]
y_predict = knn.predict(x_new)

print(classes[y_predict[0]])
print(classes[y_predict[1]])


# In[ ]:




