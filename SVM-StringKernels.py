#!/usr/bin/env python
# coding: utf-8

# In[1]:


# For String Kernels we will be using custom trained Worrd2Vec Embeddings.


# In[2]:


from utilities.data_loader import load_modeling_data, load_testing_data
from utilities.text_cleaner import advanced_data_cleaning
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# In[3]:


train_data, train_labels = load_modeling_data()


# In[4]:


X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_labels, test_size=0.2,random_state=10)


# In[5]:


le = LabelEncoder()
y_train['target'] = le.fit_transform(y_train['target'])
y_valid['target'] = le.transform(y_valid['target'])


# In[6]:


length = [len(i) for i in train_data['text']]


# In[7]:


print("The Average sentence length is", np.mean(length))
print("The Standard Deviation is", round(np.std(length)))
print(np.percentile(length, 95))


# In[8]:


# Creating Word Embeddings
NB_WORDS = 10000  # Parameter indicating the number of words we'll put in the dictionary


# In[9]:


full_df = pd.concat([X_train['text'], X_valid['text']], axis = 0)


# In[10]:


corpus=full_df.values
corpus = [x.split() for x in corpus]


# In[11]:


print("Model Training Started...")
model = Word2Vec(sentences=corpus, vector_size=10, window=1, min_count=1, workers=4)


# In[12]:


print("Total number of unique words loaded in Model : ", len(model.wv))
# model.wv.most_similar('laptop', topn=10)
model.save("word2vec-training.model")


# In[13]:


tk = Tokenizer(num_words=NB_WORDS,lower=True,split=" ")
tk.fit_on_texts(full_df)


# In[14]:


X_train_values = X_train['text'].values
X_valid_values = X_valid['text'].values


# In[15]:


train_embedding = []
for item in X_train_values:
    words = item.split()
    emb = []
    for word in words:
        if word in model.wv:
            emb.append(sum(abs(model.wv[word]))/len(model.wv[word]))
        else:
            emb.append(0)
    train_embedding.append(emb)
    
valid_embedding = []
for item in X_valid_values:
    words = item.split()
    emb = []
    for word in words:
        if word in model.wv:
            emb.append(sum(abs(model.wv[word]))/len(model.wv[word]))
        else:
            emb.append(0)
    valid_embedding.append(emb)


# In[16]:


X_train_seq_trunc = pad_sequences(train_embedding, maxlen=10)


# In[17]:


X_valid_seq_trunc = pad_sequences(valid_embedding, maxlen=10)


# In[18]:


svc_clf = CalibratedClassifierCV(base_estimator=LinearSVC(penalty='l2', dual=False), cv=3)
svc_clf.fit(X_train_seq_trunc, y_train['target'].values)
y_pred = svc_clf.predict(X_valid_seq_trunc)


# In[19]:


print(accuracy_score(y_valid['target'].values, y_pred))


# In[ ]:




