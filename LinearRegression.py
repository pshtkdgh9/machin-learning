#!/usr/bin/env python
# coding: utf-8

# In[65]:


import matplotlib.pylab as plt
from sklearn import linear_model

reg = linear_model.LinearRegression()

X = [[174],[152],[138],[128],[186]]
y = [71, 55, 46, 38, 88]
reg.fit(X,y) #학습

print(reg.predict([[165]]))
x_new= [[165]]
y_new = reg.predict(x_new)

# 학습 데이터와 y 값을 산포도로 그린다.
plt.scatter(X,y,color='black')
plt.scatter(x_new,y_new,color='r')

#학습 데이터와 y 값을 산포도로 그린다.
y_pred = reg.predict(X)

# 학습 데이터와 예측값으로 선그래프로 그린다.
#계산된 기울기와 절편을 가지는 직선이 그려진다.
plt.plot(X, y_pred, color='blue', linewidth=3)
plt.hlines(y=y_new,xmin=0,xmax=x_new, color='r',linestyle=':')
plt.vlines(x=x_new,ymin=0,ymax=y_new, color='y',linestyle=':')

plt.xlim(125,189)
plt.ylim(34,91)

plt.show()


# In[ ]:





# In[ ]:




