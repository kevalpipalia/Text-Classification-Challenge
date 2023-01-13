#!/usr/bin/env python
# coding: utf-8

# In[57]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# In[30]:


from utilities.data_loader import load_modeling_data, load_testing_data
from utilities.text_cleaner import advanced_data_cleaning


# In[31]:


train_data, train_labels = load_modeling_data()


# In[32]:


dict_vals = {'negative': 0, 'neutral': 1, 'positive': 2}
train_labels['target'] = train_labels['target'].apply(lambda x: dict_vals[x])


# In[33]:


test_data = load_testing_data()


# In[34]:


X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_labels, test_size=0.2, random_state=10)


# In[35]:


X_train['text'] = X_train['text'].apply(advanced_data_cleaning)


# In[36]:


X_valid['text'] = X_valid['text'].apply(advanced_data_cleaning)


# In[37]:


from sklearn.pipeline import Pipeline

text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MultinomialNB())])


# In[38]:


text_clf = text_clf.fit(X_train['text'], y_train['target'].values)
y_pred = text_clf.predict(X_valid['text'])


# In[39]:


print("Accuracy: ", accuracy_score(y_valid['target'].values, y_pred))


# In[12]:


from sklearn.model_selection import GridSearchCV
parameters = {'tfidf__max_features': [50000, 80000, 120000],
              'tfidf__ngram_range': [(1, 1), (1, 2),(1,3)],
               'tfidf__use_idf': (True, False),
               'clf__alpha': (1e-3, 1e-2)}


# In[ ]:


bs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
bs_clf = bs_clf.fit(X_train['text'], y_train['target'].values)


# In[ ]:


print(bs_clf.cv_results_)


# In[ ]:





# In[41]:


tfidf = TfidfVectorizer(max_features=250000,ngram_range=(1, 3))
X_train_vec = tfidf.fit_transform(X_train['text'])
X_valid_vec = tfidf.transform(X_valid['text'])


# In[42]:


nb_clf = MultinomialNB()
nb_clf.fit(X_train_vec, y_train['target'].values)
y_pred = nb_clf.predict(X_valid_vec)


# In[43]:


print("Accuracy: ", accuracy_score(y_valid['target'].values, y_pred))


# In[44]:


print(classification_report(y_valid['target'].values, y_pred))


# In[45]:


import joblib
# save the model to disk
filename = 'models/naive_bayes+tfidf.sav'
joblib.dump(nb_clf, filename)


# In[46]:


lr_clf = LogisticRegression()
lr_clf.fit(X_train_vec, y_train['target'].values)
y_pred = lr_clf.predict(X_valid_vec)


# In[47]:


print("Accuracy: ", accuracy_score(y_valid['target'].values, y_pred))


# In[48]:


print(classification_report(y_valid['target'].values, y_pred))


# In[ ]:





# In[49]:


svc_clf = CalibratedClassifierCV(base_estimator=LinearSVC(penalty='l2', dual=False), cv=3)
svc_clf.fit(X_train_vec, y_train['target'].values)
y_pred = svc_clf.predict(X_valid_vec)


# In[50]:


print("Accuracy: ", accuracy_score(y_valid['target'].values, y_pred))


# In[51]:


print(classification_report(y_valid['target'].values, y_pred))


# In[52]:


# save the model to disk
filename = 'models/svc+tfidf.sav'
joblib.dump(svc_clf, filename)


# In[ ]:





# In[ ]:


get_ipython().run_cell_magic('time', '', "xgb = XGBClassifier()\nxgb.fit(X_train_vec, y_train['target'].values)\ny_pred = xgb.predict(X_valid_vec)")


# In[54]:


print("Accuracy: ", accuracy_score(y_valid['target'].values, y_pred))


# In[55]:


print(classification_report(y_valid['target'].values, y_pred))


# In[ ]:





# In[58]:


# complement Naive Bayes
comp_nb = ComplementNB(alpha=3.0)
comp_nb.fit(X_train_vec, y_train['target'].values)
y_pred = comp_nb.predict(X_valid_vec)


# In[59]:


print("Accuracy: ", accuracy_score(y_valid['target'].values, y_pred))


# In[60]:


print(classification_report(y_valid['target'].values, y_pred))


# In[61]:


# save the model to disk
filename = 'models/complement_nb.sav'
joblib.dump(comp_nb, filename)


# In[ ]:





# In[ ]:


# Final model


# In[ ]:


# Retraining for Kaggle Submission

# final_training_data = train_data.copy()

# final_training_labels = train_labels.copy()

# final_test_data = test_data.copy()

# vectorizer = TfidfVectorizer(max_features=120000, ngram_range=(1, 3))
# final_training_data = vectorizer.fit_transform(final_training_data['text'])

# final_test_data = vectorizer.transform(final_test_data['text'])

# lr_clf = LogisticRegression()
# lr_clf.fit(final_training_data, final_training_labels['target'])
# final_test_pred = lr_clf.predict(final_test_data) 

# final_test_pred = pd.DataFrame(final_test_pred).reset_index()
# final_test_pred.columns = ['id', 'target']
# final_test_pred.to_csv('kaggle-submissions/logreg-tfidf-advcln.csv',index=False)


# In[ ]:




