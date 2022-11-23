#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import Libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
#the warnings module
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#import dataset

df_voi = pd.read_csv('Data/voice-classification.csv')
df_voi.shape


# In[3]:


df_voi.head()


# In[4]:


df_voi.info


# In[5]:


df_voi.describe()


# In[6]:


#check for missing values 0 state there are no missing values in the data

df_voi.isnull().sum()


# In[7]:


#get shape of data to record the total number of labels
#Record the total number of males
#Record the total number of females

print('Shape of Data:',df_voi.shape)
print('Total number of labels:{}'.format(df_voi.shape[0]))
print('Nurber of male: {}'.format(df_voi[df_voi.label=="male"].shape[0]))
print('Number of female: {}'.format(df_voi[df_voi.label=='female'].shape[0]))


# In[8]:


#Etract the traget variable within X
#Solve a classfication problem the labels
#Converted to numerical form

X = df_voi.iloc[:,:-1]
print(df_voi.shape)
print(X.shape)


# In[9]:


#Use the sklearn dot preprocessing library
#Reference to the previouis demo, use label encoder to extract

from sklearn.preprocessing import LabelEncoder


# In[10]:


#Encode the categorical labels, so that male gets encoded to one and femail get encoded to zero
#Extract the value of the last column with in y, and initialise the label encoder
#Making the model efficient, scale the training data between zero and one. 

y = df_voi.iloc[:,-1]
gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y)
y


# In[11]:


#import standard scaler from SK learn dot pre processing
#Initialise the standard scalar
#Fit the scalar over the training data 
#Use transform fuction to apply the transformation on x


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)


# In[12]:


#Import the train test split funciton within the SK learn dot model selection module
#Splitting the data between training and testing sets, that training to test ratio is 70% or 30%

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=100)


# In[13]:


#Train a support vector machine classifier
#Call the SVC model from SK learn and fit the model to the training data
#Fit the model to the training data
#Import the classification report and confusion matrix classes with the SK learn dot metrics library

from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix


# In[14]:


#Initialise the SVC model
#Fit the SVC model on the training sets 
#Perform prediction on the testing set
#Get predictions from the model and print accuracy

svc_model = SVC()
svc_model.fit(X_train,y_train)
y_pred = svc_model.predict(X_test)
                           


# In[15]:


#Get predictions from the model and print accuracy

print('Accuracy Score')
print(metrics.accuracy_score(y_test,y_pred)) 


# In[16]:


#Create a confusion matrix and classification report with respect to y
#Uderscore test and y underscore pred. 

print(confusion_matrix(y_test,y_pred))


# In[17]:


#Tune the parameters
#The data is small good to optimise the results using the same grid search

from sklearn.model_selection import GridSearchCV


# In[18]:


#Create a dictionary called param grid and fill out parameters for C and gamma
#

param_grid = {'C': [0.1,1,10,100], 'gamma':[1,0.1,0.01,0.001]}


# In[19]:


#Initialise the grid search CV model that refit is equal to true and 
#Verbose is equal to the parameter refit in estimator using the best 
#Found parameters on the whole data set
#Fit the grid search model on the training sets

grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)


# In[20]:


#Take grid model and create predictions using the test set

grid_predictions = grid.predict(X_test)


# In[21]:


#Calculate the accuracy score for the same

print('Accuracy Score:')
print(metrics.accuracy_score(y_test,grid_predictions))


# In[22]:


#Create classification reports and confusion matrices for them 
#with respect to y underscore test and grid underscore predictions

print(confusion_matrix(y_test,grid_predictions))


# In[23]:


#Successfully built a vocie classification model

print(classification_report(y_test,grid_predictions))


# In[ ]:





# In[ ]:




