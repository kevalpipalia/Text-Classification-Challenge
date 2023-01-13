#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utilities.data_loader import load_modeling_data, load_testing_data
from utilities.text_cleaner import advanced_data_cleaning


# In[30]:


from keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
# Packages for modeling
from keras import models
from keras import layers
from keras import regularizers


# In[3]:


train_data, train_labels = load_modeling_data()


# In[4]:


train_data['text'] = train_data['text'].apply(advanced_data_cleaning)
# dict_val = {'negative': 0, 'neutral': 1, 'positive': 2}
# train_labels['target'] = train_labels['target'].apply(lambda x: dict_val[x])


# In[5]:


X_train, X_valid, y_train, y_valid = train_test_split(train_data, train_labels, test_size=0.2,random_state=10)


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


from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


# In[10]:


tk = Tokenizer(num_words=NB_WORDS,lower=True,split=" ")


# In[11]:


tk.fit_on_texts(X_train['text'])


# In[12]:


X_train_seq = tk.texts_to_sequences(X_train['text'])


# In[13]:


X_valid_seq = tk.texts_to_sequences(X_valid['text'])


# In[14]:


seq_lengths = X_train['text'].apply(lambda x: len(x.split(' ')))
seq_lengths.describe()


# In[15]:


np.percentile(seq_lengths, 95)


# In[16]:


X_train_seq_trunc = pad_sequences(X_train_seq, maxlen=30)
X_valid_seq_trunc = pad_sequences(X_valid_seq, maxlen=30)


# In[17]:


from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical


# In[18]:


le = LabelEncoder()
y_train_le = le.fit_transform(y_train['target'])
y_valid_le = le.transform(y_valid['target'])
y_train_oh = to_categorical(y_train_le)
y_valid_oh = to_categorical(y_valid_le)


# In[33]:


vocab_size = len(tk.word_index) + 1


# In[34]:


emb_model = models.Sequential()
emb_model.add(Embedding(vocab_size, 64, input_length=30))
emb_model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(None, 1))))
emb_model.add(Dropout(0.2))
emb_model.add(Bidirectional(LSTM(32)))
emb_model.add(Dropout(0.2))
emb_model.add(Dense(64, activation='relu'))
emb_model.add(Dropout(0.1))
emb_model.add(Dense(3, activation='softmax'))
emb_model.summary()


# In[36]:


def deep_model(model, X_train, y_train, X_valid, y_valid):
    '''
    Function to train a multi-class model. The number of epochs and 
    batch_size are set by the constants at the top of the
    notebook. 
    
    Parameters:
        model : model with the chosen architecture
        X_train : training features
        y_train : training target
        X_valid : validation features
        Y_valid : validation target
    Output:
        model training history
    '''
    model.compile(optimizer='adam'
                  , loss='categorical_crossentropy'
                  , metrics=['accuracy'])
    
    history = model.fit(X_train
                       , y_train
                       , epochs=NB_START_EPOCHS
                       , batch_size=BATCH_SIZE
                       , validation_data=(X_valid, y_valid))
    return history


# In[37]:


NB_START_EPOCHS = 10
BATCH_SIZE = 512


# In[38]:


emb_history = deep_model(emb_model, X_train_seq_trunc, y_train_oh, X_valid_seq_trunc, y_valid_oh)
# emb_history.history['val_acc'][-1]


# In[39]:


emb_history.history['val_accuracy'][-1]


# In[43]:


import seaborn as sns
sns.set_theme(context="notebook", palette=("deep"))


# In[47]:


def eval_metric(history, metric_name):
    '''
    Function to evaluate a trained model on a chosen metric. 
    Training and validation metric are plotted in a
    line chart for each epoch.
    
    Parameters:
        history : model training history
        metric_name : loss or accuracy
    Output:
        line chart with epochs of x-axis and metric on
        y-axis
    '''
    metric = history.history[metric_name]
    val_metric = history.history['val_' + metric_name]

    e = range(1, NB_START_EPOCHS + 1)

    plt.plot(e, metric, 'bo', label='Train ' + metric_name)
    plt.plot(e, val_metric, 'b', label='Validation ' + metric_name)
    plt.title('Training '+metric_name+' vs validation '+metric_name)
    plt.ylabel(metric_name)
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


# In[48]:


eval_metric(emb_history, 'accuracy')


# In[49]:


eval_metric(emb_history, 'loss')


# In[ ]:


plt.plot(history_embedding.history['accuracy'],c='b',label='train accuracy')
plt.plot(history_embedding.history['val_accuracy'],c='r',label='validation accuracy')
plt.legend(loc='lower right')
plt.show()


# In[ ]:





# In[ ]:


get_ipython().system('pip install wordcloud')


# In[48]:


# Importing required libraries
import nltk
import pandas as pd
from nltk.corpus import stopwords
from textblob import Word
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import wordcloud
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt


# In[ ]:


common_words=''
for i in X_train['text']:
    i = str(i)
    tokens = i.split()
    common_words += " ".join(tokens)+" "
wordcloud = wordcloud.WordCloud().generate(common_words)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

