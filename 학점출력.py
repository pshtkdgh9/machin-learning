#!/usr/bin/env python
# coding: utf-8

# In[43]:


a = int(input("번호 : "))
b1 = int(input("국어 점수 : "))
b2 = int(input("영어 점수 : "))
b3 = int(input("수학 점수 : "))
b4 = int(input("물리 점수 : "))

r = float((b1+b2+b3+b4)/4.0);
s = b1+b2+b3+b4;
if r >= 90 :
    score = 'A'
elif r >= 80:
    score = 'B'
elif r >= 70:
    score = 'C'
elif r >= 60:
    score = 'D'
else :
    score = 'F'
print("============================================")
print("번호  국어  영어  수학  물리  총점  평균 학점")
print("=============================================")
print("%2.d"%a,"%5.d"%b1, "%5.d"%b2,  "%5.d"%b3,   "%5.d"%b4,"%6.d"%s,   "%7.2f"%r, score)


# In[ ]:





# In[ ]:




