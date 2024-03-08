#!/usr/bin/env python
# coding: utf-8

# <h1> <center> Supervised Machine Learning (Regression)
# 

# <h3> <center> SALARY AND EXPERINCE DATA SET 

# <h4> IMPORTING LIBRARIES 

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting


# <h4> sklearn package for machine learning in python

# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# In[23]:


df = pd.read_csv(r"C:\Users\owner\Downloads\salary_data (1).csv")


# <h3> EDA 

# In[24]:


df.head()


# In[25]:


df.shape


# In[26]:


df.info()


# In[27]:


df.columns


# In[28]:


df.describe()


# In[29]:


df.corr()


# In[32]:


df.corr(),'\n'


# <h3> LINEAR REGRESSION 

# In[34]:


X = df.iloc[:, [0]].values # inputs YearsExperience
y = df.iloc[:, 1].values # outputs Salary


# In[35]:


# visualise initial data set
fig1, ax1 = plt.subplots()
ax1.scatter(X, y, color = 'blue')
fig1.tight_layout()
fig1.savefig('LR_initial_plot.png')


# <h3> SPLITTING DATASET INTO TRAIN AND TEST 

# In[36]:


# split the data into training and test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3,
random_state = 0)


# In[37]:


# fit the linear least-squares regression line to the training data:
regr = LinearRegression()
regr.fit(X_train, y_train)


# In[41]:


# visualise training set results
fig2, ax2 = plt.subplots()
ax2.scatter(X_train, y_train, color = 'red')
ax2.plot(X_train, regr.predict(X_train), color = 'blue')
ax2.set_title('Salary vs Experience (Train set)')
ax2.set_xlabel('Years of Experience')
ax2.set_ylabel('Salary')
fig2.tight_layout()
fig2.savefig('LR_train_plot.png')


# In[39]:


# visualise test set results
fig3, ax3 = plt.subplots()
ax3.scatter(X_test, y_test, color = 'red')
ax3.plot(X_test, regr.predict(X_test), color = 'blue')
ax3.set_title('Salary vs Experience (Test set)')
ax3.set_xlabel('Years of Experience')
ax3.set_ylabel('Salary')
fig3.tight_layout()
fig3.savefig('LR_test_plot.png')


# <h3> TESTING MODEL 

# In[42]:


# The coefficients
print('Coefficients: ', regr.coef_)

# The intercept
print('Intercept: ', regr.intercept_)

# The mean squared error
print('Mean squared error: %.8f'
% mean_squared_error(y_test, regr.predict(X_test)))

# The R^2 value:
print('Coefficient of determination: %.2f'
% r2_score(y_test, regr.predict(X_test)))


# <h3> PREDICTION 

# In[43]:


print('Predict single value: ', regr.predict(np.array([[6]])))


# In[44]:


print('Predict single value: ', regr.predict(np.array([[10]])))


# In[45]:


print('Predict single value: ', regr.predict(np.array([[15]])))


# In[ ]:




