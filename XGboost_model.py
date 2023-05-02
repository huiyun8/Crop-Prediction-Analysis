#!/usr/bin/env python
# coding: utf-8

# ## XGBoost (eXtreme Gradient Boosting) 
# **XGBoost is a powerful machine learning algorithm used in crop prediction systems to recommend the most suitable crop based on factors such as soil properties, weather conditions, and nutrient requirements. It learns patterns from historical data and helps make informed decisions for optimal crop growth.**

# In[1]:


# Import necessary libraries
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load the dataset
data = pd.read_csv('Crop_recommendation.csv')


# ## Data Wrangling

# In[3]:


# Check for null values
print(data.isnull().sum())


# We checked there are no null values in our dataset, so we don't need to impute or drop any columns.

# In[4]:


# Rename Columns
data = data.rename(columns={'N': 'Nitrogen', 'P': 'Phosphorus', 'K': 'Potassium', 
                            'temperature': 'Temperature', 'humidity': 'Humidity', 'ph': 'pH', 
                            'rainfall': 'Rainfall', 'label': 'Crop'})


# In[5]:


# Capitalizing the first letter of column name
data.columns = data.columns.str.capitalize()


# ## Data Exploration

# In[6]:


# checking the top 5 rows
data.head()


# In[7]:


# checking bottom rows
data.tail()


# In[8]:


# checking the data types
data.dtypes


# ## Data Visualization

# In[9]:


# Pairplot to visualize relationships between features
sns.pairplot(data, hue='Crop')
plt.show()


# In[10]:


data.corr()


# In[11]:


# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()


# ## Model Training and Evaluation

# In[12]:


# Encode the labels as integers
label_encoder = LabelEncoder()
data['Crop'] = label_encoder.fit_transform(data['Crop'])


# In[13]:


# Create a list of crop names
crop_list = label_encoder.classes_
print('Crop List:', crop_list)


# In[14]:


# Split dataset into features and target
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# In[15]:


# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[16]:


# Create the DMatrix object
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test)


# In[17]:


# Define the hyperparameters
params = {
    'objective': 'multi:softmax', # Specify the objective function
    'num_class': len(set(y_train)), # Specify the number of classes
    'eta': 0.1, # Learning rate
    'max_depth': 6, # Maximum depth of the tree (increase to reduce overfitting)
    'verbosity': 0, # Set the level of information printed during training
    'subsample': 0.7, # Subsample ratio of the training instances (increase to reduce overfitting)
    'colsample_bytree': 0.7, # Subsample ratio of columns when constructing each tree (increase to reduce overfitting)
    'alpha': 1, # L1 regularization parameter (Lasso)
    'lambda': 1, # L2 regularization parameter (Ridge)
}


# In[18]:


# Train the model
num_rounds = 50
model = xgb.train(params, dtrain, num_rounds)


# In[19]:


# Predict the test results
y_pred = model.predict(dtest)


# ## Model Evaluation and Performance Metrics

# In[20]:


# Calculate the accuracy, confusion matrix, and classification report
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)


# In[21]:


#checking the accuracy of the model
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_mat)
print("Classification Report:\n", classification_rep)


# In[22]:


#Create a heatmap of the confusion matrix
sns.heatmap(confusion_mat, annot=True, cmap='coolwarm')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[23]:


#Plot the feature importances
fig, ax = plt.subplots(figsize=(10,6))
xgb.plot_importance(model, ax=ax)
plt.title('Feature Importances')
plt.show()


# ## Model Serialization and Deserialization

# **Pickle approach**

# In[24]:


import pickle


# In[25]:


# Save the model to a pickle file
with open('crop_recommendation_model.pkl', 'wb') as file:
    pickle.dump(model, file)


# In[26]:


# Load the model from the pickle file
with open('crop_recommendation_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


# In[27]:


# Use the loaded model to make predictions
dtest = xgb.DMatrix(X_test)
y_pred_loaded = loaded_model.predict(dtest)


# In[28]:


# Check if the predictions made by the loaded model are the same as the original model
assert (y_pred == y_pred_loaded).all(), "Loaded model predictions do not match original model predictions"


# In[ ]:




