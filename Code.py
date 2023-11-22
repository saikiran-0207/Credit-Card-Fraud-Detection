#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec


# In[8]:


data = pd.read_csv("data1.csv")


# In[10]:


print(data.head())


# In[11]:


print(data.isnull().sum())


# In[12]:


print(data.type.value_counts())


# In[13]:


correlation = data.corr()
print(correlation["isFraud"].sort_values(ascending=False))


# In[14]:


data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
print(data.head())


# In[15]:


from sklearn.model_selection import train_test_split
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])


# In[16]:


from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))


# In[17]:


features = np.array([[4, 9000.60, 9000.60, 0.0]])
print(model.predict(features))


# In[ ]:




