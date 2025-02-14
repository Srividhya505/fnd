#!/usr/bin/env python
# coding: utf-8

# # ML Model Building and Deployment Using Flask

# ## Fake News Detection

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv("fake_news.csv")
data.head()


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.isna().sum()


# In[6]:


data = data.drop(['id'], axis=1)


# In[7]:


# fill missing values with empty string
data = data.fillna('')


# In[8]:


data['content'] = data['author']+' '+ data['title']+' '+data['text']


# In[9]:


data = data.drop(['title','author', 'text'], axis=1)


# In[10]:


data.head()


# ## Data Pre-processing

# In[11]:


# Convert to lowercase
data['content'] = data['content'].apply(lambda x: " ".join(x.lower() for x in x.split()))


# In[12]:


# Remove punctuation
data['content'] = data['content'].str.replace('[^\w\s]','')


# In[13]:


#import nltk
#nltk.download('stopwords')


# In[14]:


# Remove stop words
from nltk.corpus import stopwords
stop = stopwords.words('english')
data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))


# In[15]:


#!pip install textblob


# In[16]:


# Do lemmatization
from nltk.stem import WordNetLemmatizer
from textblob import Word
data['content'] = data['content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
data['content'].head()


# In[17]:


#separating the data and label
X = data[['content']]
y = data['label']


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


# splitting into training and testing data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=45, stratify=y)


# In[20]:


#validate the shape of train and test dataset
print (X_train.shape)
print (y_train.shape)
print (X_test.shape)
print (y_test.shape)


# In[21]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[22]:


tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(data['content'])
xtrain_tfidf = tfidf_vect.transform(X_train['content'])
xtest_tfidf = tfidf_vect.transform(X_test['content'])


# # Model Building

# ## 1. Passive Aggressive Classifier
# 
# Passive-Aggressive algorithms are generally used for large-scale learning. It is one of the few ```online-learning algorithms```. In online machine learning algorithms, the input data comes in sequential order and the machine learning model is updated step-by-step, as opposed to batch learning, where the entire training dataset is used at once. 

# In[23]:


from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn import metrics
pclf = PassiveAggressiveClassifier()
pclf.fit(xtrain_tfidf, y_train)
predictions = pclf.predict(xtest_tfidf)
print(metrics.classification_report(y_test, predictions))


# In[24]:


print(metrics.confusion_matrix(y_test,predictions)) 


# ## 2. MLP Classifier

# In[25]:


from sklearn.neural_network import MLPClassifier
mlpclf = MLPClassifier(hidden_layer_sizes=(256,64,16),
                       activation = 'relu', 
                       solver = 'adam')
mlpclf.fit(xtrain_tfidf, y_train)
predictions = mlpclf.predict(xtest_tfidf)
print(metrics.classification_report(y_test, predictions))


# In[26]:


print(metrics.confusion_matrix(y_test,predictions)) 


# In[27]:


import pickle
# Save trained model to file
pickle.dump(mlpclf, open("fakenews1.pkl", "wb"))


# In[36]:


def fake_news_det(news):
    input_data = [news]
    vectorized_input_data = tfidf_vect.transform(input_data)
    prediction = pclf.predict(vectorized_input_data)
    print(prediction)


# In[37]:


fake_news_det('U.S. Secretary of State John F. Kerry said Monday that he will stop in Paris later this week, amid criticism that no top American officials attended Sundayâ€™s unity march against terrorism.')


# In[38]:


fake_news_det("""Go to Article 
President Barack Obama has been campaigning hard for the woman who is supposedly going to extend his legacy four more years. The only problem with stumping for Hillary Clinton, however, is sheâ€™s not exactly a candidate easy to get too enthused about.  """)


# In[1]:


Deploy_Directory
    - templates 
        - index.html
    - app.py
    - data file


# In[ ]:


python app.py

