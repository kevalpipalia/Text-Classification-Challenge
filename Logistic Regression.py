#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Importing custom utility functions
from utilities.data_loader import load_modeling_data, load_testing_data, prepare_kaggle_submission
from utilities.text_cleaner import advanced_data_cleaning

# Importing modeling utilities
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectPercentile
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# In[2]:


# Loading Raw training and testing data
train_data, train_labels = load_modeling_data()
test_data = load_testing_data()


# In[3]:


le = LabelEncoder()
train_labels['target'] = le.fit_transform(train_labels['target'].values)


# In[4]:


# Splitting data for validation
# Using 20% data for validation and keeping random_state 8 for consistency in stated results in report.
X_train, X_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=0.2, random_state = 8)


# # Experiment 0: Making Baseline

# In[5]:


# Initializing Tf-Idf vectorizer instance
print('-'*175+'Logistic Regression with TFIDF'+'-'*175)
vectorizer = TfidfVectorizer()


# In[6]:


# Fitting and training the bag of words
X_train_vectorizer = vectorizer.fit_transform(X_train['text'])
X_val_vectorizer = vectorizer.transform(X_val['text'])


# In[7]:


print("shape of the bag of words matrix: ",X_train_vectorizer.shape)


# In[8]:


# Initializing naive bayes classifier
lr_clf_0 = LogisticRegression()


# In[9]:


# Training the classifier with default parameters
lr_clf_0.fit(X_train_vectorizer, y_train['target'].values)


# In[10]:


# Pridicting from the validation set
y_pred_val = lr_clf_0.predict(X_val_vectorizer)


# In[11]:


# Printing the results
print('Accuracy score: ', accuracy_score(y_val['target'].values, y_pred_val))
print('Confusion Matrix: ')
print(confusion_matrix(y_val['target'].values, y_pred_val))
print('Classification Report: ')
print(classification_report(y_val['target'].values, y_pred_val))


# In[ ]:





# # Experiment 1: Testing Text Cleaning Improvements

# In[12]:


print('-'*175+'Logistic Regression with advance text cleaning'+'-'*175)


# In[13]:


# making copy of dataframe and applying stemming to each text documents
X_train_clean = X_train.copy()
X_val_clean = X_val.copy()
X_train_clean['text'] = X_train_clean['text'].apply(advanced_data_cleaning)
X_val_clean['text'] = X_val_clean['text'].apply(advanced_data_cleaning)


# In[14]:


X_train_clean


# In[15]:


vectorizer = TfidfVectorizer()
# Fitting and training the bag of words
X_train_vectorizer = vectorizer.fit_transform(X_train_clean['text'])
X_val_vectorizer = vectorizer.transform(X_val_clean['text'])


# In[16]:


lr_clf_1 = LogisticRegression()
lr_clf_1.fit(X_train_vectorizer, y_train['target'].values)
y_val_pred = lr_clf_1.predict(X_val_vectorizer)


# In[17]:


# Printing the results
print('Accuracy score: ', accuracy_score(y_val['target'].values, y_val_pred))
print('Confusion Matrix: ')
print(confusion_matrix(y_val['target'].values, y_val_pred))
print('Classification Report: ')
print(classification_report(y_val['target'].values, y_val_pred))


# In[ ]:





# # Experiment 2: hyper parameter tuning 

# In[18]:


# from sklearn.model_selection import GridSearchCV


# In[19]:


print('-'*175+'Logistic Regression Hyper parameter tuning'+'-'*175)


# In[20]:


X_train_clean = X_train.copy()
X_val_clean = X_val.copy()
X_train_clean['text'] = X_train_clean['text'].apply(advanced_data_cleaning)
X_val_clean['text'] = X_val_clean['text'].apply(advanced_data_cleaning)


# In[21]:


pipe = Pipeline([('vec', TfidfVectorizer()), ('logreg', LogisticRegression())])


# In[22]:


# defining search grid
grid = {
#     'vec__ngram_range': [(1,1),(1,2),(1,3)],
    'vec__max_features': [10000, 50000, 120000, 250000],
    'logreg__tol': [1e-4, 1e-3],
    'logreg__solver': ['saga', 'liblinear', 'lbfgs'],
    'logreg__C': [0.5, 0.75, 1.0 , 2.0],
    'logreg__max_iter': [100, 150, 200, 300]
}


# In[23]:


# Initializing bayesian search
logreg_clf_2 = BayesSearchCV(pipe, grid, scoring='accuracy', n_iter=25)


# In[ ]:


# Training for best hyperparameters
_ = logreg_clf_2.fit(X_train_clean['text'].values, y_train['target'].values)


# In[ ]:


# printing the best found parameters
print("Best found hyperparameters are: ")
print(logreg_clf_2.best_params_)


# In[22]:


# Initializing Tf-Idf vectorizer instance
print('-'*175+'Logistic Regression with Best parameters'+'-'*175)
vectorizer = TfidfVectorizer(max_features=250000, ngram_range=(1,3))


# In[23]:


# Fitting and training the bag of words
X_train_vectorizer = vectorizer.fit_transform(X_train_clean['text'])
X_val_vectorizer = vectorizer.transform(X_val_clean['text'])


# In[24]:


print("shape of the bag of words matrix: ",X_train_vectorizer.shape)


# In[25]:


# Initializing naive bayes classifier
lr_clf_3 = LogisticRegression(max_iter=300, solver='saga', C= 0.5, tol=0.00022294400779122961)


# In[26]:


# Training the classifier with default parameters
lr_clf_3.fit(X_train_vectorizer, y_train['target'].values)


# In[27]:


# Pridicting from the validation set
y_pred_val = lr_clf_3.predict(X_val_vectorizer)


# In[28]:


# Printing the results
print('Accuracy score: ', accuracy_score(y_val['target'].values, y_pred_val))
print('Confusion Matrix: ')
print(confusion_matrix(y_val['target'].values, y_pred_val))
print('Classification Report: ')
print(classification_report(y_val['target'].values, y_pred_val))


# In[30]:


import joblib
# save the model to disk
filename = 'models/logreg+tfidf.sav'
joblib.dump(lr_clf_3, filename)


# In[ ]:





# In[ ]:





# # SMOTE - DATA IMBALANCE

# In[34]:


from imblearn.over_sampling import SMOTE
# Generating synthetic samples for underrepresented class
smote = SMOTE(random_state=8, n_jobs=-1)
X_SMOTE, y_SMOTE = smote.fit_resample(X_train_vectorizer, y_train['target'])


# In[35]:


lr_clf_4 = LogisticRegression(max_iter=300, solver='saga', C= 0.5, tol=0.00022294400779122961)


# In[36]:


lr_clf_4.fit(X_SMOTE,y_SMOTE)


# In[37]:


y_pred_val = lr_clf_3.predict(X_val_vectorizer)


# In[38]:


# Printing the results
print('Accuracy score: ', accuracy_score(y_val['target'].values, y_pred_val))
print('Confusion Matrix: ')
print(confusion_matrix(y_val['target'].values, y_pred_val))
print('Classification Report: ')
print(classification_report(y_val['target'].values, y_pred_val))


# In[39]:


import joblib
# save the model to disk
filename = 'models/logreg+tfidf+smote.sav'
joblib.dump(lr_clf_4, filename)


# # Finalizing model for Kaggle Submission

# In[ ]:


print("Logistic Regression experiments completed, starting retraining on full training data with best model for kaggle submission..")


# In[ ]:


X_train_final = train_data.copy()
X_test_final = test_data.copy()
X_train_final['text'] = X_train_final['text'].apply(advanced_data_cleaning)
X_test_final['text'] = X_test_final['text'].apply(advanced_data_cleaning)


# In[ ]:


vectorizer = TfidfVectorizer(max_features=250000, ngram_range=(1,3))
X_train_final_vec = vectorizer.fit_transform(X_train_final['text'])
X_test_final_vec = vectorizer.transform(X_test_final['text'])


# In[ ]:


logreg_final = LogisticRegression(C=0.5, max_iter=300, solver='saga',
                   tol=0.00022294400779122961)
logreg_final.fit(X_train_final_vec, train_labels['target'].values)
y_test_pred = logreg_final.predict(X_test_final_vec)


# In[ ]:


prepare_kaggle_submission(y_test_pred, 'final-logreg-advance-clean-tfidf-13-hp.csv')


# In[ ]:





# In[ ]:


# Kaggle public score: 


# In[ ]:




