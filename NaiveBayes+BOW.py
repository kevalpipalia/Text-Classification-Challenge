#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import pandas as pd
from sklearn.naive_bayes import MultinomialNB, ComplementNB

# Importing custom utility functions
from utilities.data_loader import load_modeling_data, load_testing_data, prepare_kaggle_submission
from utilities.text_cleaner import advanced_data_cleaning

# Importing modeling utilities
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
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


# # Experiment 1: Making Baseline

# In[5]:


# Initializing Bag of Words instance (CountVectorizer)
print('-'*175+'Baseline Naive Bayes'+'-'*175)
bow = CountVectorizer()


# In[6]:


# Fitting and training the bag of words
X_train_bow = bow.fit_transform(X_train['text'])
X_val_bow = bow.transform(X_val['text'])


# In[7]:


print("shape of the bag of words matrix: ",X_train_bow.shape)


# In[8]:


# Initializing naive bayes classifier
nb_clf_1 = MultinomialNB()


# In[9]:


# Training the classifier with default parameters
nb_clf_1.fit(X_train_bow, y_train['target'].values)


# In[10]:


# Pridicting from the validation set
y_pred_val = nb_clf_1.predict(X_val_bow)


# In[11]:


# Printing the results
print('Accuracy score: ', accuracy_score(y_val['target'].values, y_pred_val))
print('Confusion Matrix: ')
print(confusion_matrix(y_val['target'].values, y_pred_val))
print('Classification Report: ')
print(classification_report(y_val['target'].values, y_pred_val))


# In[ ]:





# # Experiment 2: hyper parameter tuning 

# In[12]:


print('-'*175+'Naive Bayes Hyper parameter tuning'+'-'*175)


# In[13]:


# defining search grid
grid = {
    'alpha': [0, 0.25, 1, 2, 3, 5, 10]
}


# In[14]:


# Initializing bayesian search
nb_clf_2 = BayesSearchCV(MultinomialNB(), grid, n_iter= 7)


# In[15]:


# Training for best hyperparameters
_ = nb_clf_2.fit(X_train_bow, y_train['target'].values)


# In[16]:


# printing the best found parameters
print("Best found hyperparameters are: ")
print(nb_clf_2.best_params_)


# In[17]:


nb_clf_2 = MultinomialNB(alpha=2.0)
nb_clf_2.fit(X_train_bow, y_train['target'].values)
y_val_pred = nb_clf_2.predict(X_val_bow)


# In[18]:


# Printing the results
print('Accuracy score: ', accuracy_score(y_val['target'].values, y_val_pred))
print('Confusion Matrix: ')
print(confusion_matrix(y_val['target'].values, y_val_pred))
print('Classification Report: ')
print(classification_report(y_val['target'].values, y_val_pred))


# In[ ]:





# # Experiment 3: Stemming

# In[19]:


print('-'*175+'Naive Bayes with stemming'+'-'*175)


# In[20]:


# defining stemming function
def stemmer(text):
    porter = PorterStemmer()
    ls = [porter.stem(word) for word in text.split()]
    return ' '.join(ls)


# In[21]:


# making copy of dataframe and applying stemming to each text documents
X_train_stem = X_train.copy()
X_val_stem = X_val.copy()
X_train_stem['text'] = X_train_stem['text'].apply(stemmer)
X_val_stem['text'] = X_val_stem['text'].apply(stemmer)


# In[22]:


bow = CountVectorizer()
X_train_bow = bow.fit_transform(X_train_stem['text'])
X_val_bow = bow.transform(X_val_stem['text'])


# In[23]:


nb_clf_3 = MultinomialNB(alpha=2.0)
nb_clf_3.fit(X_train_bow, y_train['target'].values)
y_val_pred = nb_clf_3.predict(X_val_bow)


# In[24]:


# Printing the results
print('Accuracy score: ', accuracy_score(y_val['target'].values, y_val_pred))
print('Confusion Matrix: ')
print(confusion_matrix(y_val['target'].values, y_val_pred))
print('Classification Report: ')
print(classification_report(y_val['target'].values, y_val_pred))


# In[ ]:





# In[ ]:





# # Experiment 4: Lemmatizing

# In[25]:


print('-'*175+'Naive Bayes with lemmatizing'+'-'*175)


# In[26]:


# defining lemmatizing function
def lemmatizer(text):
    wordnet = WordNetLemmatizer()
    ls = [wordnet.lemmatize(word) for word in text.split()]
    return ' '.join(ls)


# In[27]:


# making copy of dataframe and applying lemmatizing to each text documents
X_train_lemmatize = X_train.copy()
X_val_lemmatize = X_val.copy()
X_train_lemmatize['text'] = X_train_lemmatize['text'].apply(lemmatizer)
X_val_lemmatize['text'] = X_val_lemmatize['text'].apply(lemmatizer)


# In[28]:


bow = CountVectorizer()
X_train_bow = bow.fit_transform(X_train_lemmatize['text'])
X_val_bow = bow.transform(X_val_lemmatize['text'])


# In[29]:


nb_clf_4 = MultinomialNB(alpha=2.0)
nb_clf_4.fit(X_train_bow, y_train['target'].values)
y_val_pred = nb_clf_4.predict(X_val_bow)


# In[30]:


# Printing the results
print('Accuracy score: ', accuracy_score(y_val['target'].values, y_val_pred))
print('Confusion Matrix: ')
print(confusion_matrix(y_val['target'].values, y_val_pred))
print('Classification Report: ')
print(classification_report(y_val['target'].values, y_val_pred))


# In[ ]:





# # Experiment 5: Removing stopwords

# In[31]:


print('-'*175+'Naive Bayes with stop words removal'+'-'*175)


# In[32]:


# defining stopwords remover function
def remove_stopwords(text):
    text = text.lower()
    ls = [word for word in text.split() if word not in stop_words]
    return ' '.join(ls)


# In[33]:


# making copy of dataframe and applying lemmatizing to each text documents
X_train_stop = X_train.copy()
X_val_stop = X_val.copy()
X_train_stop['text'] = X_train_stop['text'].apply(remove_stopwords)
X_val_stop['text'] = X_val_stop['text'].apply(remove_stopwords)


# In[34]:


bow = CountVectorizer()
X_train_bow = bow.fit_transform(X_train_stop['text'])
X_val_bow = bow.transform(X_val_stop['text'])


# In[35]:


nb_clf_5 = MultinomialNB(alpha=2.0)
nb_clf_5.fit(X_train_bow, y_train['target'].values)
y_val_pred = nb_clf_5.predict(X_val_bow)


# In[36]:


# Printing the results
print('Accuracy score: ', accuracy_score(y_val['target'].values, y_val_pred))
print('Confusion Matrix: ')
print(confusion_matrix(y_val['target'].values, y_val_pred))
print('Classification Report: ')
print(classification_report(y_val['target'].values, y_val_pred))


# In[ ]:





# # Experiment 6: Advanced Text Cleaning

# In[37]:


print('-'*175+'Naive Bayes with advanced text cleaning'+'-'*175)


# In[38]:


# making copy of dataframe and applying lemmatizing to each text documents
X_train_clean = X_train.copy()
X_val_clean = X_val.copy()
X_train_clean['text'] = X_train_clean['text'].apply(advanced_data_cleaning)
X_val_clean['text'] = X_val_clean['text'].apply(advanced_data_cleaning)


# In[39]:


bow = CountVectorizer()
X_train_bow = bow.fit_transform(X_train_clean['text'])
X_val_bow = bow.transform(X_val_clean['text'])


# In[40]:


nb_clf_6 = MultinomialNB(alpha=2.0)
nb_clf_6.fit(X_train_bow, y_train['target'].values)
y_val_pred = nb_clf_6.predict(X_val_bow)


# In[41]:


# Printing the results
print('Accuracy score: ', accuracy_score(y_val['target'].values, y_val_pred))
print('Confusion Matrix: ')
print(confusion_matrix(y_val['target'].values, y_val_pred))
print('Classification Report: ')
print(classification_report(y_val['target'].values, y_val_pred))


# In[ ]:


# save the model to disk
filename = 'models/naive_bayes+BOW.sav'
joblib.dump(nb_clf_6, filename)


# # Finalizing model for Kaggle Submission

# In[42]:


print("Naive Bayes experiments completed, starting retraining on full training data with best model for kaggle submission..")


# In[43]:


X_train_final = train_data.copy()
X_test_final = test_data.copy()
X_train_final['text'] = X_train_final['text'].apply(advanced_data_cleaning)
X_test_final['text'] = X_test_final['text'].apply(advanced_data_cleaning)


# In[44]:


bow = CountVectorizer()
X_train_bow = bow.fit_transform(X_train_final['text'])
X_test_bow = bow.transform(X_test_final['text'])


# In[45]:


nb_clf_6 = MultinomialNB(alpha=3.0)
nb_clf_6.fit(X_train_bow, train_labels['target'].values)
y_test_pred = nb_clf_6.predict(X_test_bow)


# In[46]:


prepare_kaggle_submission(y_test_pred, 'final-naive-bayes-advance-clean-bow-hp.csv')


# In[47]:



X_train_bow = X_train_bow.astype('float32')
X_val_bow = X_val_bow.astype('float32')


# In[ ]:





# In[ ]:




