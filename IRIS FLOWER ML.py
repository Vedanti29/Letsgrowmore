#!/usr/bin/env python
# coding: utf-8

# # LGMVIP - Data Science Intern FEB23

# ## Importing required

# In[1]:


# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score, plot_confusion_matrix, classification_report,confusion_matrix


# ## Loading data

# In[5]:


data = pd.read_csv("E:\letsgrowmemore\Iris.csv")
data.head()


# In[6]:


data.shape


# In[7]:


# Dataset Columns
data.columns


# In[8]:


# Dataset Summary
data.info()


# In[9]:


# Check Statistical Summary
data.describe()


# In[10]:


# Checking Null Values
data.isnull().sum()


# In[11]:


data['Species'].unique()


# In[12]:


# Checking columns count of "Species"
data['Species'].value_counts()


# ## Exploratory data Analysis

# In[13]:


sns.set(rc = {'figure.figsize':(14,6)})
sns.pairplot(data, hue='Species',palette="husl")


# In[14]:



fig,(ax1,ax2)=plt.subplots(ncols=2,figsize=(16,5))
sns.scatterplot(x='SepalLengthCm',y='PetalLengthCm',data=data,hue='Species',ax=ax1,s=300,marker='o')
sns.scatterplot(x='SepalWidthCm',y='PetalWidthCm',data=data,hue='Species',ax=ax2,s=300,marker='o')


# In[16]:


sns.distplot(x=data["PetalWidthCm"], kde=True, color='BLUE');


# In[20]:


sns.distplot(x=data["PetalLengthCm"], kde=True, color='yellow');


# In[21]:



# Pie plot to show the overall types of Iris classifications
colors = ['YELLOW','ORANGE','RED']
data['Species'].value_counts().plot(kind = 'pie',  autopct = '%1.1f%%', shadow = True,colors=colors, explode = [0.08,0.08,0.08])


# In[22]:


## Heat Map for Data
sns.heatmap(data.corr(), annot=True, cmap="BuPu")


# ## Extracting dependent and independent features

# In[23]:



features = ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
X = data.loc[:, features].values   #defining the feature matrix
y = data.Species


# ## Splitting the dataset into training and test sets

# In[24]:


#Splitting the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33,random_state=0)


# ## Model training and predictions

# In[25]:


#Defining the decision tree classifier and fitting the training set
dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
DecisionTreeClassifier()


# ## Prediction on test data

# In[26]:


#Prediction on test data
y_pred = dtree.predict(X_test)
y_pred


# ## Checking the accuracy of the model

# In[27]:


#Checking the accuracy of the model
score=accuracy_score(y_test,y_pred)
print("Accuracy:",score)


# ## Plotting confusion matrix

# In[28]:



def report(model):
    preds=model.predict(X_test)
    print(classification_report(preds,y_test))
    plot_confusion_matrix(model,X_test,y_test,cmap='BuPu',colorbar=True)


# In[29]:


print('Decision Tree Classifier')
report(dtree)
print(f'Accuracy: {round(score*100,2)}%') 


# ## visulalizing the decision tree

# In[30]:


#Visualizing the decision tree
from sklearn import tree
feature_name =  ['sepal length(cm)','sepal width(cm)','petal length(cm)','petal width(cm)']
class_name= data.Species.unique()
plt.figure(figsize=(15,10))
tree.plot_tree(dtree, filled = True, feature_names = feature_name, class_names= class_name)


# ## HyperParameter Tunning

# In[31]:


params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4], 'max_depth':[4,5,6,7]}


# In[32]:


grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=101), params, verbose=1, cv=3)
grid_search_cv.fit(X_train, y_train)


# In[33]:


grid_search_cv.best_params_


# In[34]:



grid_search_cv.best_score_


# ## Buliding tree using the best parameters

# In[35]:


model=DecisionTreeClassifier(max_depth= 4, max_leaf_nodes= 4, min_samples_split= 2)
model.fit(X_train, y_train)


# In[36]:


pred=model.predict(X_test)


# In[37]:


#Checking the accuracy of the model
score=accuracy_score(y_test,pred)
print("Accuracy of Model:",score)


# In[38]:


print('Decision Tree Classifier')
report(model)
print(f'Accuracy: {round(score*100,2)}%')


# In[39]:


#Visualizing the decision tree
from sklearn import tree
feature_name =  ['sepal length(cm)','sepal width(cm)','petal length(cm)','petal width(cm)']
class_name= data.Species.unique()
plt.figure(figsize=(8,6))
tree.plot_tree(model, filled = True, feature_names = feature_name, class_names= class_name)


# In[40]:


input_data=(6.0,3.6,2.6,1.2)

 #changing the input data to a numpy array 
input_data_as_nparray = np.asarray(input_data)

# reshape the data as we are predicting the label for only one instance 
input_data_reshaped = input_data_as_nparray.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print("The category is",prediction) 


# In[41]:



#Testing for New points except from Dataset

Test_point = [[5.4,3.0,1.5,-1.5],
             [6.5,2.8,1.5,1.3],
             [5.0,3.6,2.6,1.2],
             [5.1,3.3,0.5,1.6],
             [6.0,5.1,1.6,1.1],
             [5.0,3.6,2.6,2.2]]

print(model.predict(Test_point))


# ## Successfully Tested

# # THANKYOU
