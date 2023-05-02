#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import pickle


# In[2]:


dataset = pd.read_csv('Crop_recommendation.csv')


# In[3]:


dataset


# In[4]:


X = dataset.drop('label', axis=1)
y = dataset['label']


# In[5]:


dataset.describe()


# In[6]:


dataset['temperature'].hist()


# In[7]:


dataset['label'].value_counts().plot(kind = 'barh')


# In[8]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[9]:


dataset['N'].hist()


# In[10]:


sc = StandardScaler()
X_train = sc.fit_transform(X_train) 
X_test = sc.transform(X_test)


# In[11]:


np.histogram(X_train[:,0],bins = 10) 


# In[12]:


svm_classifier = SVC(kernel='linear', random_state=42)
svm_classifier.fit(X_train, y_train)


# In[13]:


# import pickle


# In[14]:


# Pkl_Filename = "svm_pred.pkl"  

# with open(Pkl_Filename, 'wb') as file:  
#     pickle.dump(svm_classifier, file)
    


# In[15]:


# with open(Pkl_Filename, 'rb') as file:  
#     Pickled_SVM_Model = pickle.load(file)

# Pickled_SVM_Model


# In[16]:


# # Use the Reloaded Model to 
# # Calculate the accuracy score and predict target values

# # Calculate the Score 
# score = Pickled_SVM_Model.score(X_test, y_test)  
# # Print the Score
# print("Test score: {0:.2f} %".format(100 * score))  

# # Predict the Labels using the reloaded Model
# y_pred = Pickled_SVM_Model.predict(X_test)  

# y_pred


# In[17]:


# Import Joblib Module from Scikit Learn
import joblib


# In[18]:


joblib_file = "joblib_SVM_Model.pkl"  
joblib.dump(svm_classifier, joblib_file)


# In[19]:


joblib_SVM_model = joblib.load(joblib_file)


joblib_SVM_model


# In[20]:


# Use the Reloaded Joblib Model to 
# Calculate the accuracy score and predict target values

# Calculate the Score 
score = joblib_SVM_model.score(X_test, y_test)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  

# Predict the Labels using the reloaded Model
y_pred = joblib_SVM_model.predict(X_test)  

y_pred


# In[21]:


# y_pred = loaded_svm_classifier.predict(X_test)


# In[22]:


print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nAccuracy Score:")
print(accuracy_score(y_test, y_pred))


# In[ ]:




