#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as skl


# In[5]:


data=pd.read_csv("D:\\DATA SCINCE\\ML\\prediction analasis\\Linea Regression\\Simple_Linrar_Regression\\salary.csv")


# In[6]:


data.head()


# In[16]:


data.shape


# In[7]:


x = data.iloc[:,:-1].values
y = data.iloc[:,1].values


# In[8]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split (x,y,test_size = 1/3, random_state = 0)


# In[9]:


from sklearn.linear_model import LinearRegression
simplelinearRegression=LinearRegression()
simplelinearRegression.fit(x_train,y_train)


# In[14]:


y_predict = simplelinearRegression.predict(x_test)
y_predict


# In[11]:


plt.scatter(x_train, y_train,color = 'red')
plt.plot(x_train, simplelinearRegression.predict(x_train))
plt.show()


# In[12]:


plt.scatter(x_test,y_test, color='red')
plt.plot(x_train, simplelinearRegression.predict(x_train),color='blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# In[ ]:





# In[ ]:




