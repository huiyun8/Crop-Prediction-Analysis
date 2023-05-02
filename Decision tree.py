import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier


# Load the dataset
data = pd.read_csv('Crop_recommendation.csv')
X = data.drop('label' ,axis =1)
X.head()

#
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['label'] = le.fit_transform(data['label'])
data.head()

#
y = data['label']
y.head()


model = []
accuracy = []
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)
                                                 
                                                 
                                                 
DT = DecisionTreeClassifier().fit(X_train, y_train)

predict = DT.predict(X_test)
DT_accuracy = DT.score(X_test,y_test)
DT_accuracy
DT_train_accuracy = DT.score(X_train, y_train)
DT_train_accuracy

#Let's visualize the import features which are taken into consideration by decision trees.
plt.figure(figsize=(10,4), dpi=80)
c_features = len(X_train.columns)
plt.barh(range(c_features), DT.feature_importances_)
plt.xlabel("Feature importance")
plt.ylabel("Feature name")
plt.yticks(np.arange(c_features), X_train.columns)
plt.show()
roc_auc = roc_auc_score(y_test, DT.predict_proba(X_test), multi_class = "ovo")
roc_auc
# Calculate the accuracy, confusion matrix, and classification report
accuracy = accuracy_score(y_test, predict)
confusion_mat = confusion_matrix(y_test, predict)
classification_rep = classification_report(y_test, predict)
#checking the accuracy of the model
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_mat)
print("Classification Report:\n", classification_rep)
#Create a heatmap of the confusion matrix
sns.heatmap(confusion_mat, annot=True, cmap='coolwarm')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


import pickle

Pkl_Filename = "Pickle_DecisionTree_Model.pkl"  
with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(DT, file)
# Load the Model back from file
with open(Pkl_Filename, 'rb') as file:  
    Pickle_DecisionTree_Model = pickle.load(file)
Pickle_DecisionTree_Model
# Use the Reloaded Model to 
# Calculate the accuracy score and predict target values
# Calculate the Score 
score = Pickle_DecisionTree_Model.score(X_test, y_test)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  
# Predict the Labels using the reloaded Model
Ypredict = Pickle_DecisionTree_Model.predict(X_test)  
Ypredict


