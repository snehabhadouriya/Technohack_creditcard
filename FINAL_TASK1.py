#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.svm import SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import math
from pylab import rcParams
rcParams['figure.figsize']= 14,8
RANDOM_SEED=42
LABELS=["Normal","Fraud"]
from sklearn.preprocessing import StandardScaler


# In[6]:


dataset=pd.read_csv('creditcard.csv',sep=',')
dataset.head()


# In[7]:


dataset.info()


# In[8]:


x=dataset.iloc[:,1:30].values
y=dataset.iloc[:,30].values


# In[9]:


print("input range: ",x.shape)
print("output range",y.shape)


# In[10]:


print("Class labels: \n",y)


# In[11]:


dataset.isnull().values.any()


# In[12]:


set_class=pd.value_counts(data['Class'],sort=True)
set_class.plot(kind='bar',rot=0)
plt.title("Transaction class Distribution")
plt.xticks(range(2),LABELS)
plt.xlabel("Class")
plt.ylabel("Frequency")


# In[14]:


## get the fraud and the normal dataset
fraud_data = dataset[dataset['Class']==1]
normal_data = dataset[dataset['Class']==0]


# In[15]:


print(fraud_data.shape,normal_data.shape)


# In[16]:


## we need to analyze more amount of information from the transaction data
#how different are the amount of money used in different transaction classes?
fraud_data.Amount.describe()


# In[17]:


normal_data.Amount.describe()


# In[18]:


## correlation
import seaborn as sns
# get correlation of each features in datasets
corrmat=dataset.corr()
top_corr_features= corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g= sns.heatmap(dataset[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# In[24]:


xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.25,random_state=0)


# In[25]:


print("xtrain.shape: ",xtrain.shape)
print("xtest.shape: ",xtest.shape)
print("ytrain.shape: ",ytrain.shape)
print("ytest.shape: ",ytest.shape)


# In[26]:


stdsc=StandardScaler()
xtrain=stdsc.fit_transform(xtrain)
xtest=stdsc.transform(xtest)


# In[27]:


print("Training Set after Standardised: \n",xtrain[0])


# In[30]:


dt_classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
dt_classifier.fit(xtrain,ytrain)


# In[32]:


y_pred_decision_tree = dt_classifier.predict(xtest)


# In[33]:


print("y_pred_decision_tree: \n",y_pred_decision_tree)


# In[34]:


com_decision = confusion_matrix(ytest,y_pred_decision_tree)
print("confusion Matrix: \n", com_decision)


# In[37]:


Accuracy_Model = ((com_decision[0][0]+ com_decision[1][1])/ com_decision.sum())*100
print("Accuracy_Decison : ",Accuracy_Model)
Error_rate_Model = ((com_decision[0][1]+ com_decision[1][0])/ com_decision.sum())*100
print("Error_rate_decison: ",Error_rate_Model)
specificity_Model = (com_decision[1][1]/(com_decision[1][1]+com_decision[0][1]))*100
print("Specificity_Decison: ",specificity_Model)
Sensitivity_Model = (com_decision[0][0]/(com_decision[0][0]+com_decision[1][0]))*100
print("Sensitivity_Decison: ",Sensitivity_Model)


# In[38]:


svc_classifier = SVC(kernel = 'rbf',random_state=0)
svc_classifier.fit(xtrain,ytrain)


# In[39]:


y_pred2 = svc_classifier.predict(xtest)


# In[40]:


print("y_pred_randomforest: \n",y_pred2)


# In[41]:


cm2 = confusion_matrix(ytest,y_pred2)
print("Confusion Matrix: \n",cm2)


# In[43]:


Accuracy_Model = ((cm2[0][0]+cm2[1][1])/cm2.sum())*100
print("Accuracy_svc : ",Accuracy_Model)
Error_rate_Model = ((cm2[0][1]+cm2[1][0])/cm2.sum())*100
print("Error_rate_svc: ",Error_rate_Model)
specificity_Model = (cm2[1][1]/(cm2[1][1]+cm2[0][1]))*100
print("specificity_svc: ",specificity_Model)
sensitivity_Model = (cm2[0][0]/(cm2[0][0]+cm2[1][0]))*100
print("sensitivity_svc: ",sensitivity_Model )


# In[ ]:




