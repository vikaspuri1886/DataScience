#!/usr/bin/env python
# coding: utf-8

# ## Tasks:
# 
# #### Exploratory Data Analysis:
# 1. Handling Missing Values
# 2. Check the distribution of target variable
# 3. Categorical and numerical features analysis
# 4. Analysis of categorical features w.r.t target feature y
# 
# ### Preprocessing
# 1. Remove unnecessary features
# 2. Encoding of categorical features
# 3. Check the correlation
# 
# ### Model training
# 1. Design Model

# ### Import necessary libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


# In[2]:


pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', None)


# In[3]:


original_dataset = pd.read_csv('data/train.csv')


# In[4]:


dataset = original_dataset.copy() # Make copy original dataset


# In[5]:


dataset.head()


# In[6]:


dataset.info()    # Total number of columns are 378 including Id and target variables


# In[7]:


dataset.shape


# In[8]:


dataset.drop('ID', axis=1, inplace=True)   # Drop Id column as there is not use of it


# In[9]:


dataset.head()


# In[10]:


dataset.describe() # Only will display numerical features


# ### Exploratory data analysis
# 
# ##### 1. Check the missing values

# In[11]:


dataset.isna().sum()  ## That's nice, we don't have any null entry


# ##### 2. Check the distribution of target variable

# In[12]:


dataset.head()


# In[13]:


sns.boxplot(x='y', data=dataset)
plt.xlabel('Target variable - y')


# In[14]:


sns.histplot(data=dataset,x='y', bins=50, color='r', discrete=True)
plt.xlabel('Target variable - y')


# ##### From above, we conclude that maximum distribution of target lies with in 75 to 140,  beyond that are outliers

# #### 3. Get the categorical and numerical features

# In[15]:


dataset.info() # We see from here that we have 1 float feature, 368 as int and 8 as object


# In[16]:


# Get the list of categorical features
feature_categorical = [feature for feature in dataset.columns if dataset[feature].dtype == 'O']
feature_categorical


# In[17]:


# Get the list of numerical features
feature_numerical = [feature for feature in dataset.columns if dataset[feature].dtype != 'O']
feature_numerical


# In[18]:


# Number of unique categories
for feature in feature_categorical:
    print('Feature {} has {} unique categories'.format(feature, len(np.unique(dataset[feature]))))


# #### 4. Analaysis of categorical features w.r.t target variable 'y'

# In[19]:


fig,ax = plt.subplots(len(feature_categorical), figsize=(30,50))
for i in range(len(feature_categorical)):
    sns.boxplot(x=feature_categorical[i], y='y', data=dataset, ax=ax[i])


# ### Preprocessing 
# 
# #### 1. Remove columns with only one unique value

# In[20]:


columns_with_one_unique_values = [feature for feature in dataset.columns if len(np.unique(dataset[feature])) == 1]
columns_with_one_unique_values


# In[21]:


# As there are only one repeatitive value, so we can remove these columns which is not useful in modeling
dataset.drop(columns=columns_with_one_unique_values, axis=1, inplace=True)


# In[22]:


dataset.head() # Columns are removed


# #### 2. Encoding

# In[23]:


feature_categorical


# In[24]:


dataset[feature_categorical] # Its seems variable are ordinal, so let us apply LabelEncoding over here


# In[25]:


# Apply label encoding
for feature in feature_categorical:
    label_encoder = LabelEncoder()
    dataset[feature] = label_encoder.fit_transform(dataset[feature])


# In[26]:


dataset.head()


# #### 3. Check correlation

# In[27]:


dataset.corr()


# ### 3. Modellin

# In[28]:


x = dataset.drop('y', axis=1)
y = dataset['y']


# In[33]:


train_size = int(x.shape[0] * 80/100)


# In[30]:


x_train = x[0:train_size]
y_train = y[0:train_size]

x_valid = x[train_size:] 
y_valid = y[train_size:]


# In[32]:


x_train.shape, x_valid.shape


# In[ ]:


from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_absolute_error,mean_squared_error


# In[ ]:


# Linerar Regression model
linear_regression_model=LinearRegression()
linear_regression_model.fit(x_train,y_train)
y_predict=linear_regression_model.predict(x_valid)

print(mean_absolute_error(y_valid, y_predict))
print(mean_squared_error(y_valid, y_predict))
print(np.sqrt(mean_squared_error(y_valid, y_predict)))


# In[ ]:


# Lasso model, from result we can see Lasso is far better than Linear regression
lasso_model = Lasso()
lasso_model.fit(x_train, y_train)
y_predict=lasso_model.predict(x_valid)
print(mean_absolute_error(y_valid, y_predict))
print(mean_squared_error(y_valid, y_predict))
print(np.sqrt(mean_squared_error(y_valid, y_predict)))


# In[ ]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet, Ridge


# In[ ]:


# Elastic net combination of Lasso(L1) and Ridge(L2), We can see from below results Lasso is better
elastic_net_model = ElasticNet()
elastic_net_model.fit(x_train, y_train)
y_predict=elastic_net_model.predict(x_valid)
print(mean_absolute_error(y_valid, y_predict))
print(mean_squared_error(y_valid, y_predict))
print(np.sqrt(mean_squared_error(y_valid, y_predict)))


# In[ ]:


# Ridge(L2), We can see from below results Lasso is better, Finally we can see Ridge is better than all
ridfe_net_model = Ridge()
ridfe_net_model.fit(x_train, y_train)
y_predict=ridfe_net_model.predict(x_valid)
print(mean_absolute_error(y_valid, y_predict))
print(mean_squared_error(y_valid, y_predict))
print(np.sqrt(mean_squared_error(y_valid, y_predict)))


# In[ ]:




