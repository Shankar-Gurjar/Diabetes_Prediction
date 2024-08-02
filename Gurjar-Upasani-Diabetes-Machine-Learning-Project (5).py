#!/usr/bin/env python
# coding: utf-8

# In[63]:


#importing useful liberaries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import accuracy_score


# In[64]:


#Reading the data from CSV file
df = pd.read_csv("health care diabetes_New.csv")


# In[ ]:





# In[65]:


#Used head function to see the overview of the data
df.head()


# In[66]:


#used columns method to understand the columns available in dataset
df.columns


# ## Column Description
# 
# Pregnancies = 'the condition or period of being pregnant.'
# 
# Glucose = 'type of sugar in the blood and is the major source of energy for the body's cells.'
# 
# BloodPressure = 'the force of blood pushing against the walls of your arteries as your heart pumps blood through your body'
# 
# SkinThickness = 'Thick skin is thicker due to it containing an extra layer in the epidermis, called the stratum lucidum'
# 
# Insulin = 'a hormone produced in the pancreas by the islets of Langerhans, which regulates the amount of glucose in the blood. The lack of insulin causes a form of diabetes.'
# 
# BMI = 'Body mass index (BMI) is a ratio of a person's weight in kilograms to the square of their height in meters.'
# 
# DiabetesPedigreeFunction = 'The 'DiabetesPedigreeFunction' is a function that scores the probability of diabetes based on family history, with a realistic range of 0.08 to 2.42'
# 
# Age. = 'The length of time that a person has lived'
# 
# Outcome = Diabetes, Yes or No

# In[67]:


df.info()


# In[68]:


df.isnull().sum()


# In[69]:


#used dtype function to understand the datatypes of the columsn
df.dtypes.to_frame()


# In[70]:


#used describe function to see mathematical properties of the dataset
df.describe()


# # Exploratory Data Analysis (EDA)

# In[71]:


#BloodPressure column analysis
df.BloodPressure.plot(kind='hist')


# In[ ]:





# In[72]:


df.BloodPressure.value_counts(normalize = True).to_frame().head()


# In[73]:


#Insulin column analysis
df.Insulin.plot(kind ='hist')


# In[74]:


df.Insulin.value_counts(normalize = True).to_frame().head()


# In[75]:


# Insulin median with 0
df.Insulin.median()


# In[76]:


# Insulin median without 0
Insulin_median = df[df['Insulin'] !=0]['Insulin'].median()
Insulin_median


# ### Replaced  all rows with NA or 0 with Median. We also tried Mean but distribution was right skewed so found median more useful approach for our use case.

# In[77]:


df['Insulin'] = df['Insulin'].apply(lambda x: Insulin_median if x==0 else x)
df.head()


# In[ ]:





# In[78]:


select_col = ['Glucose','BloodPressure','SkinThickness','BMI' ]

for i in select_col:
    median = df[df[i]!=0][i].median()
    df[i] = df[i].apply(lambda x: median if x==0 else x)
df.head()


# In[79]:


df.Outcome.value_counts(normalize = True).plot(kind ='bar')


# In[80]:


#Output variable analysis.
df.Outcome.value_counts(normalize = True)


# In[ ]:





# In[81]:


# Calculate correlation matrix
corr_matrix = df.corr()
# Plotting heatmap with 'flare' colormap
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='flare', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Numeric Variables')
plt.show()


# In[82]:


# Create pair scatter plot using seaborn
plt.figure(figsize=(12, 6))
sns.pairplot(df)
plt.show()


# In[83]:


# Import necessary libraries
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardizing the features before PCA
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Perform PCA
pca = PCA(n_components=2)  # You can change the number of components
principal_components = pca.fit_transform(x_scaled)

# Create a DataFrame with the principal components
principal_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])

# Concatenate the principal components with the target variable
final_df = pd.concat([principal_df, df[['Outcome']]], axis=1)

# Visualize the principal components
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Principal Component 1', y='Principal Component 2', hue='Outcome', data=final_df, palette='viridis', alpha=0.7)
plt.title('2 Component PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_
print(f'Explained variance by each component: {explained_variance}')
print(f'Total explained variance: {explained_variance.sum()}')

# Plotting the explained variance ratio
plt.figure(figsize=(8, 5))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6, align='center', label='individual explained variance')
plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', label='cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.title('Explained Variance Ratio of Principal Components')
plt.show()


# In[84]:


# Creating input and output datasets.
x = df.drop('Outcome',axis= 1)
y = df['Outcome']


# In[85]:


#Scaling the dataset
from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler()
transform_x = scaler.fit_transform(x)
transform_x


# In[86]:


#splitting the data for training & testing
from sklearn.model_selection import train_test_split


# In[87]:


x_train,x_test,y_train,y_test = train_test_split(transform_x,y,test_size=.25, random_state=42,stratify=y)


# ### DecisionTreeClassifier

# In[88]:


#Imported DecisionTreeClassifier from sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# In[89]:


# Train a decision tree model with max depth of 7
clf = DecisionTreeClassifier(max_depth=7, random_state=42)
clf.fit(x_train, y_train)

clf.score(x_train,y_train)
clf.score(x_test,y_test)
y_pred = clf.predict(x_test)


# In[90]:


# Convert feature names to a list
feature_names = x.columns.tolist()
class_names = clf.classes_.astype(str).tolist()

# Visualize the decision tree
plt.figure(figsize=(20,10))
tree.plot_tree(clf, feature_names=feature_names, class_names=class_names, filled=True)
plt.show()


# In[91]:


# List to store accuracies
train_accuracies = []
test_accuracies = []
depths = range(1, 21) 

for depth in depths:
    # Train a decision tree model with specified max depth
    clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
    clf.fit(x_train, y_train)
    
    # Predict on training and test sets
    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Plot accuracies
plt.figure(figsize=(5, 5))
plt.plot(depths, train_accuracies, label='Training Accuracy')
plt.plot(depths, test_accuracies, label='Test Accuracy')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Accuracy vs. Depth')
plt.legend()
plt.grid(True)
plt.show()


# In[92]:


#Used classification report for evaluation of model performance
from sklearn.metrics import classification_report as cr
print(cr(y_test,y_pred))


# In[93]:


#confusion matrix
from sklearn.metrics import confusion_matrix as cm
print(cm(y_test,y_pred))


# ### RandomForest ML Model implementation

# In[94]:


# RandomForestClassifier model 
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_depth=16, random_state=42)
# fit the model on traing data
rf.fit(x_train,y_train)
rf.score(x_train,y_train)
rf.score(x_test,y_test)
#prediction on test data
y_pred = rf.predict(x_test)


# In[95]:


# Lists to store accuracies
train_accuracies = []
test_accuracies = []
depths = range(1, 21)  # Vary depth from 1 to 20

for depth in depths:
    # Train a Random Forest model with specified max depth
    rf = RandomForestClassifier(max_depth=depth, n_estimators=100, random_state=42)
    rf.fit(x_train, y_train)
    
    # Predict on training and test sets
    y_train_pred = rf.predict(x_train)
    y_test_pred = rf.predict(x_test)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

# Plot accuracies
plt.figure(figsize=(5, 5))
plt.plot(depths, train_accuracies, label='Training Accuracy')
plt.plot(depths, test_accuracies, label='Test Accuracy')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.title('Random Forest Accuracy vs. Depth')
plt.legend()
plt.grid(True)
plt.show()


# In[96]:


#classification report to evaluate the model
from sklearn.metrics import classification_report as cr
print(cr(y_test,y_pred))


# In[97]:


#confusion matrix
from sklearn.metrics import confusion_matrix as cm
print(cm(y_test,y_pred))


# In[98]:


# Get feature importances
importances = rf.feature_importances_
feature_names = x.columns

# Sort feature importances in descending order
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for i in range(x.shape[1]):
    print(f"{i + 1}. feature {indices[i]} ({importances[indices[i]]})")

# Plot the feature importances
plt.figure(figsize=(5, 7))
plt.title("Feature Importances")
plt.bar(range(x.shape[1]), importances[indices], align='center')
plt.xticks(range(x.shape[1]), [feature_names[i] for i in indices], rotation=90)
plt.xlim([-1, x.shape[1]])
plt.show()


# In[ ]:




