#!/usr/bin/env python
# coding: utf-8

# In[28]:


x = int(input("첫 번째 정수를 입력하시오:"))
y = int(input("두 번째 정수를 입력하시오:"))
z = str(input("연산자를 입력하세요: "))

if (z == '+'):
    result = x + y;
elif (z == '-'):
    result = x - y;
elif (z == '*'):
    result = x * y;
elif (z == '//' ):
    result = x//y;
else :
    result = x%y;
    
print(x,z,y, "=",result)


# In[ ]:





# In[ ]:




