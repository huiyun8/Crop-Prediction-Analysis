#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv('/Users/pauviramontes/Downloads/Crop_recommendation 3.csv')


# In[2]:


df.head(10)


# In[3]:


df['label'] = df['label'].astype(str)


# In[4]:


df['label'].value_counts()


# In[5]:


df['label'].nunique()


# In[6]:


df['label'] = df['label'].str.strip()


# In[7]:


df['label'] = df['label'].astype(str).str.lower()


# Check number of unique classes in target variable
print("Unique values in target variable:", df['label'].unique())
print("Number of classes:", len(df['label'].unique()))

# Check class distribution
print(df['label'].value_counts())


# In[8]:


X = df[['N', 'P', 'K', 'temperature', 'humidity','ph', 'rainfall']]


# In[9]:


y = df['label']


# In[10]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[11]:


from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[12]:


import numpy as np

unique_values = np.unique(df['label'])
num_classes = len(unique_values)

print("Unique values in target variable:", unique_values)
print("Number of classes:", num_classes)


# In[13]:


y_pred = logreg.predict(X_test)


# In[14]:


print(y_pred)


# In[15]:


import pandas as pd

# create a DataFrame with the test data and the predicted values
results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# display the first 10 rows of the results
print(results.head(10))


# In[16]:


accuracy = logreg.score(X_test, y_test)
print("The model's accuracy is:", accuracy)


# In[17]:


from sklearn.metrics import f1_score

# Assuming y_test and y_pred are the true labels and predicted labels respectively
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"F1 score: {f1:.2f}")


# In[18]:


from sklearn.metrics import classification_report

y_pred = logreg.predict(X_test)
print(classification_report(y_test, y_pred))


# In[19]:


from sklearn.metrics import confusion_matrix

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(cm)


# In[20]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix with colors
sns.heatmap(cm, annot=True, cmap='Blues')


# In[21]:


import pickle

# Save model to file
Pkl_Filename = "LR.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(logreg, file)


# In[22]:


# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:  
    Pickled_LG_Model = pickle.load(file)

Pickled_LG_Model


# In[23]:


# Use the Reloaded Model to 
# Calculate the accuracy score and predict target values

# Calculate the Score 
score = Pickled_LG_Model.score(X_test, y_test)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  


# In[24]:


print(y_pred)


# In[ ]:




