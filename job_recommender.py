#!/usr/bin/env python
# coding: utf-8

# In[16]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


# In[3]:


df=pd.read_csv('Skill_Map.csv')
df.columns = ['job', 'skill']
df.head()


# In[5]:


len(df) ###  900 Jobs


# In[22]:


s=input("Explain your skillset:\n")


# In[23]:


all_skills = df['skill'].values
all_jobs=df['job'].unique().tolist()


# In[24]:


def find_related_jobs(s):
    doc = s
    skill_match_score = []
    dictionary_list = []
#Vectorizing the data and finding Cosine Similarity
    for j in range(len(all_skills)):
        final_score = 0
        tfidf_vectorizer = TfidfVectorizer(ngram_range = (1,3))
        train_set = [doc , all_skills[j]]
        tfidf_train = tfidf_vectorizer.fit_transform(train_set)
        score = cosine_similarity(tfidf_train[0:1],tfidf_train)
        skill_match_score.append("%.2f" % score[0][1])
    dictionary = dict(zip(all_jobs, skill_match_score))
    dictionary_sorted_by_value = sorted(dictionary.items(), key=lambda kv: kv[1], reverse = True)
# Top 5 recommended jobs are stored
    for k in range(5):
        dictionary_list.append(dictionary_sorted_by_value[k])
#most recommended is returned
    return (dictionary_list[0][0])


# In[25]:


find_related_jobs(s)


# In[ ]:




