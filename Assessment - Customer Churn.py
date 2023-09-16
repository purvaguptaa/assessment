#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing all the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
import os
from textwrap import wrap
import missingno as msno
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from chart_studio.plotly import plot, iplot
from plotly.offline import iplot
import plotly.express as px
from sklearn.preprocessing import LabelEncoder 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
# evaluations
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

# ignore warnings
import warnings
warnings.filterwarnings('ignore')

# to display the total number columns present in the dataset
pd.set_option('display.max_columns', None)
from sklearn.model_selection import cross_val_score


# In[2]:


# Loading given dataset as DataFrame
df = pd.read_excel("/Users/purvagupta/Downloads/customer_churn_large_dataset.xlsx")
print("\nColumn names and data types:")
print(df.dtypes)


# In[3]:


# Calculating basic summary statistics for numeric features
(df.describe())


# In[4]:


# Knowing Total number of records
df.shape


# In[5]:


# Finding out any duplicate rows
df.duplicated().sum()


# In[6]:


# Is there any empty data in training dataset?
df.isnull().sum()/df.shape[0]


# In[7]:


# Visually checking any missing fields in the provided dataset
msno.matrix(df);


# In[8]:


# Checking for unique values
df.describe(include=object).T


# In[9]:


# What are the features available and what are their data type?
df.info()


# In[10]:


# Checking for any outliers
# Replace 'numeric_feature' with the name of the numeric feature you want to analyze for outliers
numeric_feature = 'Subscription_Length_Months'

# Calculate the IQR (Interquartile Range)
Q1 = df[numeric_feature].quantile(0.25)
Q3 = df[numeric_feature].quantile(0.75)
IQR = Q3 - Q1

# Define the lower and upper bounds for outliers detection
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Detect outliers
outliers = df[(df[numeric_feature] < lower_bound) | (df[numeric_feature] > upper_bound)]

# Print the number of detected outliers and display the outliers
print("Number of outliers detected:", len(outliers))
print("Outliers:")
print(outliers)


# In[11]:


numeric_feature1 = 'Monthly_Bill'

# Calculate the IQR (Interquartile Range)
Q11 = df[numeric_feature1].quantile(0.25)
Q33 = df[numeric_feature1].quantile(0.75)
IQR1 = Q33 - Q11

# Define the lower and upper bounds for outliers detection
lower_bound1 = Q11 - 1.5 * IQR1
upper_bound1 = Q33 + 1.5 * IQR1

# Detect outliers
outliers1 = df[(df[numeric_feature1] < lower_bound1) | (df[numeric_feature1] > upper_bound1)]

# Print the number of detected outliers and display the outliers
print("Number of outliers detected:", len(outliers1))
print("Outliers:")
print(outliers1)


# In[12]:


#Performing categorical encoding
categorical_feature = 'Location'

# Performing label encoding
label_encoder = LabelEncoder()
df[categorical_feature + '_encoded'] = label_encoder.fit_transform(df[categorical_feature])

# Printing the encoded dataset
print(df.head())


# In[13]:


categorical_feature1 = 'Gender'


label_encoder1 = LabelEncoder()
df[categorical_feature1 + '_encoded'] = label_encoder1.fit_transform(df[categorical_feature1])

print(df.head())


# In[14]:


columns_to_drop = ['CustomerID', 'Name', 'Location','Gender']
df = df.drop(columns=columns_to_drop)
print(df.head())


# In[15]:


# Defining feature columns (X) and target variable column (y)
X = df[['Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB']]  # Replace with your actual feature columns
y = df['Churn']  # Replace with your actual target variable column

# Splitting the data into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Printing the sizes of the resulting sets
print("Training set size:", X_train.shape[0])
print("Testing set size:", X_test.shape[0])


# In[16]:


numeric_columns = df.select_dtypes(include=['number'])
correlation_matrix = numeric_columns.corr()
# Selecting numeric columns for correlation analysis
numeric_columns = df.select_dtypes(include=['number'])

# Calculating the correlation matrix
correlation_matrix = numeric_columns.corr()

# Creating a heatmap to visualize correlations
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()


# In[19]:


fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])

fig.add_trace(go.Pie(labels=df['Gender_encoded'].unique(), values=df['Gender_encoded'].value_counts(), name='Gender', 
                     marker_colors=['gold', 'mediumturquoise']), 1, 1)
fig.add_trace(go.Pie(labels=df['Churn'].unique(), values=df['Churn'].value_counts(), name='Churn', 
                     marker_colors=['darkorange', 'lightgreen']), 1, 2)

fig.update_traces(hole=0.5, textfont_size=20, marker=dict(line=dict(color='black', width=2)))

fig.update_layout(
    title_text='<b>Gender and Churn Distributions<b>', 
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Gender', x=0.19, y=0.5, font_size=20, showarrow=False),
                 dict(text='Churn', x=0.8, y=0.5, font_size=20, showarrow=False)])
iplot(fig)


# In[20]:


# Initializing the Logistic Regression model
model = LogisticRegression()

# Training the model on the training data
model.fit(X_train, y_train)

# Making predictions on the test data
y_pred = model.predict(X_test)

# Evaluating the model's performance
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion)
print("Classification Report:\n", classification_rep)


# In[21]:


#Initializing the Decision Tree model
model2 = DecisionTreeClassifier(random_state=42)

# Training the model on the training data
model2.fit(X_train, y_train)

# Making predictions on the test data
y_pred2 = model2.predict(X_test)

# Evaluating the model's performance
accuracy2 = accuracy_score(y_test, y_pred2)
confusion2 = confusion_matrix(y_test, y_pred2)
classification_rep2 = classification_report(y_test, y_pred2)

# Prining model performance metrics
print("Accuracy:", accuracy2)
print("Confusion Matrix:\n", confusion2)
print("Classification Report:\n", classification_rep2)


# In[22]:


#Initializing the Random Forest model
model3 = RandomForestClassifier(n_estimators=100, random_state=42)

# Training the model on the training data
model3.fit(X_train, y_train)

# Making predictions on the test data
y_pred3 = model3.predict(X_test)

# Evaluating the model's performance
accuracy3 = accuracy_score(y_test, y_pred3)
confusion3 = confusion_matrix(y_test, y_pred3)
classification_rep3 = classification_report(y_test, y_pred3)

# Prining model performance metrics
print("Accuracy:", accuracy3)
print("Confusion Matrix:\n", confusion3)

print("Classification Report:\n", classification_rep3)


# In[24]:


#Performing hyperparameter tuning for Logistic Regression

params_LR = {'C': list(np.arange(1,12)), 'penalty': ['l2', 'elasticnet', 'none'], 'class_weight': ['balanced','None']}
grid_LR = RandomizedSearchCV(model, param_distributions=params_LR, cv=5, n_jobs=-1, n_iter=20, random_state=42, return_train_score=True)
grid_LR.fit(X_train, y_train)
print('Best parameters:', grid_LR.best_estimator_)


# In[26]:


#Performing Cross Validation

CV = StratifiedKFold(n_splits=10, random_state=0, shuffle=True)

LR = LogisticRegression(random_state = 42, penalty= 'l2', class_weight= 'balanced', C=6)
cross_val_LR_Acc = cross_val_score(LR, X_train, y_train, cv = CV, scoring = 'accuracy') 
cross_val_LR_f1 = cross_val_score(LR, X_train, y_train, cv = CV, scoring = 'f1')
cross_val_LR_AUC = cross_val_score(LR, X_train, y_train, cv = CV, scoring = 'roc_auc')


# In[29]:


#Finding which feature carries more importance

RF_I = RandomForestClassifier(n_estimators=70, random_state=42)
RF_I.fit(X, y)
d = {'Features': X_train.columns, 'Feature Importance': RF_I.feature_importances_}
df = pd.DataFrame(d)
df_sorted = df.sort_values(by='Feature Importance', ascending = True)
df_sorted
df_sorted.style.background_gradient(cmap='Blues')


# In[30]:


fig = px.bar(x=df_sorted['Feature Importance'], y=df_sorted['Features'], color_continuous_scale=px.colors.sequential.Blues,
             title='<b>Feature Importance Based on Random Forest<b>', text_auto='.4f', color=df_sorted['Feature Importance'])

fig.update_traces(marker=dict(line=dict(color='black', width=2)))
fig.update_layout({'yaxis': {'title':'Features'}, 'xaxis': {'title':'Feature Importance'}})

iplot(fig)


# In[ ]:




