
# coding: utf-8

# ## Boston House price prediction

# ### Loading the modules and the dataset

# In[1]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
from sklearn.datasets import load_boston
boston_data=load_boston()


# In[2]:


boston_data.data.shape


# ### Understanding the dataset

# In[3]:


print(boston_data.keys())


# In[5]:


X = boston_data.data
Y = boston_data.target


# In[16]:


boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston.columns


# In[7]:


X.shape


# In[8]:


Y.shape


# In[9]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[10]:


lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)


# In[11]:


lin_model.coef_


# ### Looking at the coefficient values, the least influencing variable is NOX and RM is the highest influencing variable on the price of a house in Boston

# In[12]:


lin_model.intercept_

