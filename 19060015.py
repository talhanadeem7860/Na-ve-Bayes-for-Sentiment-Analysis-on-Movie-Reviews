#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import os
import numpy as np
import pandas as pd


# In[2]:


#reads and opens the training set of positive and negative files
path1 = 'train/pos/'
path2 = 'train/neg/'
pos_file = os.listdir(path1)
pos_file = [open(path1+f, 'r',encoding = 'utf8').read() for f in pos_file]
neg_file = os.listdir(path2)
neg_file = [open(path2+f, 'r',encoding = 'utf8').read() for f in neg_file]


# In[3]:


#accessing stopwords
my_file =open("stop_words.txt")
content = my_file.read()
content = re.sub(r'[^(a-zA-Z)\s]','',content)
stop_words = content.split("\n")
my_file.close()


# In[4]:


rev=[]
for p in pos_file:
    #creates a list in which the first element is the review while second is the respective label
    rev.append( (p, 1) )
    df =  pd.DataFrame(rev,columns=['Review','Label'])


# In[5]:


def clean_data(data):
    
    data = re.sub('[()]', '', data)    #removes paranthesis
        
    data = re.sub('[^ a-zA-Z0-9]', '', data) #removes special characters
    
    data = re.sub('[0-9]', '', data)  #removes numbers
     
    data = data.lower() #converts the review into lower case
   
    return data


# In[6]:


df['Review'] = df['Review'].apply(clean_data)


# In[7]:


df.head()


# In[8]:


def data_tokenize(tokenized_text):
    return tokenized_text.split()      #tokenizes the data


# In[9]:


df['Review']=df['Review'].apply(data_tokenize)


# In[10]:


df.head()


# In[11]:


def Remove_Stop_Words(data):
    filtered_sentence = [w for w in data if not w in stop_words]
    return filtered_sentence


# In[12]:


df['Review']=df['Review'].apply(Remove_Stop_Words)


# In[13]:


df.head()


# In[14]:


rev_neg=[]
for p in neg_file:
    rev_neg.append( (p, 0) )
    df_neg =  pd.DataFrame(rev_neg,columns=['Review','Label'])


# In[15]:


df_neg.head()


# In[16]:


df_neg['Review'] = df_neg['Review'].apply(clean_data)


# In[17]:


df_neg.head()


# In[18]:


df_neg['Review'] = df_neg['Review'].apply(data_tokenize)


# In[19]:


df_neg.head()


# In[20]:


df_neg['Review'] = df_neg['Review'].apply(Remove_Stop_Words)


# In[21]:


df_neg.head()


# In[22]:


merged_train = pd.concat([df, df_neg], ignore_index=True) #combines the whole training data


# In[23]:


merged_train.head


# In[24]:


#opening and reading testing files
path1 = 'test/pos/'
path2 = 'test/neg/'
test_pos = os.listdir(path1)
test_pos = [open(path1+f, 'r',encoding = 'utf8').read() for f in test_pos]
test_neg = os.listdir(path2)
test_neg = [open(path2+f, 'r',encoding = 'utf8').read() for f in test_neg]


# In[25]:


pos_test = []
for p in test_pos:
    pos_test.append( (p, 1) )
    df_pos =  pd.DataFrame(pos_test,columns=['Review','Label'])


# In[26]:


df_pos['Review'] = df_pos['Review'].apply(clean_data)


# In[27]:


df_pos.head()


# In[28]:


df_pos['Review'] = df_pos['Review'].apply(data_tokenize)


# In[29]:


df_pos.head()


# In[30]:


df_pos['Review'] = df_pos['Review'].apply(Remove_Stop_Words)


# In[31]:


df_pos.head()


# In[32]:


neg_test = []
for p in test_neg:
    neg_test.append( (p, 0) )
    df_test =  pd.DataFrame(neg_test,columns=['Review','Label'])


# In[33]:


df_test['Review'] = df_test['Review'].apply(clean_data)


# In[34]:


df_test.head()


# In[35]:


df_test['Review'] = df_test['Review'].apply(data_tokenize)


# In[36]:


df_test.head()


# In[37]:


df_test['Review'] = df_test['Review'].apply(Remove_Stop_Words)


# In[38]:


df_test.head()


# In[39]:


merged_test = pd.concat([df_pos, df_test], ignore_index=True) #merges the whole of testing data


# In[40]:


merged_test.head


# In[41]:


#Creates the training and testing datasets along with respective labels
X_train = merged_train['Review'].values
Y_train = merged_train['Label'].values
X_test = merged_test['Review'].values
Y_test = merged_test['Label'].values


# In[42]:


from collections import defaultdict,Counter
import math


# In[43]:


def unique_label(Y):
    return np.unique(Y)


# In[44]:


def TrainNB(X, Y):                       #{} represents dictionary data
    Ndoc = {}         
    log_priors = {}
    word_count = {}
    vocab = set()
    Class = unique_label(Y) #gives unique elemtents present in labels
    data_grouped = dict()
    
    for c in Class:
        data_grouped[c] = X[np.where(Y == c)]     
    
    #calculation of prior prob and word counts
    for c, data in data_grouped.items():
        
        Ndoc[c] = len(data)
        log_priors[c] = math.log(Ndoc[c] / len(X))
        word_count[c] = defaultdict(lambda: 0)
        
        for syntax in data:      
            
            for word, count in Counter(syntax).items():
                
                if word not in vocab:
                    
                    vocab.add(word)      #vocabulary
                word_count[c][word] = word_count[c][word] + count #gives the final word count
    return log_priors,vocab,word_count,Ndoc


# In[45]:


log_priors,vocab,word_count,Ndoc = TrainNB(X_train,Y_train)


# In[46]:


def predictNB(X):   #prediction function
    out = []
    Class = unique_label(Y_train)
    for syntax in X:
        class_results = {c:log_priors[c] for c in Class}
        words = set(syntax)
        for word in words:
            if word not in vocab: continue
            for c in Class:
                log_w_c = Laplace_smoothing(word, c)  #laplace smoothing applied
                class_results[c] = class_results[c] + log_w_c
                func = class_results.get
        out.append(max(class_results, key=func))
    return out


# In[47]:


#Add1 smoothing
def Laplace_smoothing(word, syntax_class):
    v = len(vocab)
    numerator = word_count[syntax_class][word] + 1
    denominator = Ndoc[syntax_class] + v
    smooth_out = math.log(numerator / denominator)
    return smooth_out


# In[48]:


Y_predict = predictNB(X_test)
Y_predict = np.array(Y_predict)


# In[49]:


def Evaluation_func(Y_actual,Y_predicted):
    confusion_matrix = pd.crosstab(Y_actual, Y_predicted,rownames=['Actual'], colnames=['Predicted']) #forms confusion matrix
    print('The confusion matrix is \n\n',confusion_matrix,'\n')
    accuracy =100* (confusion_matrix[0][0]+confusion_matrix[1][1])/((confusion_matrix[0][0]+confusion_matrix[1][1])+(confusion_matrix[0][1]+confusion_matrix[1][0]))
    print('Accuracy of testing data is:',accuracy,'%')


# In[50]:


Evaluation_func(Y_test,Y_predict)


# In[51]:


#Part2


# In[52]:


#creates corpus for training and testing data along with labels
rev=[]
rev_neg=[]
pos_test = []
neg_test = []


for p in pos_file:
    
    rev.append( (p, 1) )
    corpus =  pd.DataFrame(rev,columns=['Review','Label'])

for p in neg_file:
    rev_neg.append( (p, 0) )
    corpus_neg =  pd.DataFrame(rev_neg,columns=['Review','Label'])
    

for p in test_pos:
    pos_test.append( (p, 1) )
    corpus_test_pos =  pd.DataFrame(pos_test,columns=['Review','Label'])
    
    
for p in test_neg:
    neg_test.append( (p, 0) )
    corpus_test_neg =  pd.DataFrame(neg_test,columns=['Review','Label'])


# In[53]:


corpus_train = pd.concat([corpus, corpus_neg], ignore_index=True) #merges the whole of training corpus
corpus_test = pd.concat([corpus_test_pos,corpus_test_neg],ignore_index=True) #merges the whole testing corpus


# In[54]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[55]:


Xtr = corpus_train['Review'].values
Ytr = corpus_train['Label'].values
Xte = corpus_test['Review'].values
Yte = corpus_test['Label'].values


# In[56]:


#Feature extraction
vectorizer = CountVectorizer(stop_words = stop_words)
Xtrain = vectorizer.fit_transform(Xtr)
Xtest = vectorizer.transform(Xte)


# In[57]:


#Training and prediction
MB = MultinomialNB()
MB.fit(Xtrain,Ytr)
Ypred = MB.predict(Xtest)


# In[58]:


accu_score = 100*accuracy_score(Yte, Ypred)
print('The accuracy is',accu_score,'%')


# In[59]:


confusion_mat = confusion_matrix(Yte, Ypred)
print('The confusion matrix is\n',confusion_mat)


# In[ ]:




