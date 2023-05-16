# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 18:38:05 2023

@author: Big data lab project
"""


import pandas as pd
import numpy as np


import string

import spacy

import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel


# libraries for visualization

import matplotlib.pyplot as plt

from wordcloud import WordCloud
#---------------------------------------

data= pd.read_csv(r"Data_Review.csv")

data.info()
data.head(5)

sentiment=data['sentiment'].replace({0:'Negative review',1:"Positive review"})
sentiment.value_counts().to_frame()
sentiment.value_counts().plot(kind="bar",color='green')
#-----------------------------------------------------------------------------
print('Length of Data is ' ,len(data))
##---------------------------------------------------

from cleantext import clean
clean(data['review'], no_emoji=True)
#------------------------------------------------------------------------------
def clean_text(text ): 
    delete_dict = {sp_character: '' for sp_character in string.punctuation} 
    delete_dict[' '] = ' ' 
    table = str.maketrans(delete_dict)
    text1 = text.translate(table)
    #print('cleaned:'+text1)
    textArr= text1.split()
    text2 = ' '.join([w for w in textArr if ( not w.isdigit() and  ( not w.isdigit() and len(w)>3))]) 
    
    return text2.lower()


data['review'] = data['review'].apply(clean_text)

#------------------------------------------------------------------------------
#Let us pre-process the data
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
# function to remove stopwords
def remove_stopwords(text):
    textArr = text.split(' ')
    rem_text = " ".join([i for i in textArr if i not in stop_words])
    return rem_text
#------------------------------------------------------------------------------

df = data

# remove stopwords from the text
df['review']=df['review'].apply(remove_stopwords)



#------------------------------------------------------------------------------

# WORD CLOUD


text = " ".join(review for review in df['review'])
wordcloud = WordCloud(background_color="white").generate(text)
wordcloud.words_
# plot the WordCloud image                      
plt.figure(figsize = (10, 10), facecolor = None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad = 0)
plt.show()
#------------------------------------------------------------------------------
#lemmatization
nlp = spacy.load('en_core_web_md')

def lemmatization(texts,allowed_postags=['NOUN', 'ADJ']): 
       output = []
       for sent in texts:
             doc = nlp(sent) 
             output.append([token.lemma_ for token in doc if token.pos_ in allowed_postags ])
       return output
   
text_list=df['review'].tolist()
print(text_list[1])
tokenized_reviews = lemmatization(text_list)
print(tokenized_reviews[1])


import pickle

# saving
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenized_reviews, handle, protocol=pickle.HIGHEST_PROTOCOL)

# loading
#with open('tokenizer.pickle', 'rb') as handle:
 #   tokenizer = pickle.load(handle)
#------------------------------------------------------------------------------

#Create vocabulary dictionary and document term matrix

dictionary = corpora.Dictionary(tokenized_reviews)
#dictionary.save('dictionary.txt')
len(dictionary)
print(dictionary)

#from gensim.utils import simple_preprocess
#from smart_open import smart_open
#import os
#dict_STF = corpora.Dictionary( simple_preprocess(line, deacc =True) for line in open(r'dictionary.txt', encoding='unicode_escape'))
#dict_STF = corpora.Dictionary(tokenizer)
#print(dict_STF.token2id)



########################################################
doc_term_matrix = [dictionary.doc2bow(rev) for rev in tokenized_reviews]

#gensim.corpora.MmCorpus.serialize('corpus.mm', doc_term_matrix)

for doc in doc_term_matrix :
   print([[dictionary[id], freq] for id, freq in doc])


#from gensim.corpora.mmcorpus import MmCorpus
#from gensim.test.utils import datapath

#corpus = MmCorpus(datapath('test_mmcorpus_with_index.mm'))
#for document in corpus:
 #   pass
#for doc in corpus :
 #  print([[dictionary[id], freq] for id, freq in doc])

#-----------------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfconvert = TfidfVectorizer(analyzer=clean_text,ngram_range=(1,3)).fit(df['review'])
len(tfidfconvert.vocabulary_)
x_transformed = tfidfconvert.transform(df['review'])


from sklearn.cluster import KMeans
Sum_of_squared_distance =[]
K = range(1,60)
for k in K:
 	km = KMeans(n_clusters=k)
 	km = km.fit(x_transformed )
 	Sum_of_squared_distance.append(km.inertia_)
     
plt.plot(K,Sum_of_squared_distance,'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distance')
plt.title('Elbow method for optimal k')
plt.show()

#------------------------------------------------------------------------------

# Gap Statistic for K means
def optimalK(data, nrefs=3, maxClusters=15):
    """
    Calculates KMeans optimal K using Gap Statistic 
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount':[], 'gap':[]})
    for gap_index, k in enumerate(range(1, maxClusters)):
# Holder for reference dispersion results
        refDisps = np.zeros(nrefs)
# For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)
            
            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)
            
            refDisp = km.inertia_
            refDisps[i] = refDisp
# Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)
        
        origDisp = km.inertia_
# Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
# Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap
        
        resultsdf = resultsdf.append({'clusterCount':k, 'gap':gap}, ignore_index=True)
    return (gaps.argmax() + 1, resultsdf)
score_g, df2 = optimalK(x_transformed , nrefs=5, maxClusters=50)
plt.plot(df2['clusterCount'], df2['gap'], linestyle='--', marker='o', color='b');
plt.xlabel('K');
plt.ylabel('Gap Statistic');
plt.title('Gap Statistic vs. K');

#----------------------------------------------------------------------------
#Final Model 

# Creating the object for LDA model using gensim library
LDA = gensim.models.ldamodel.LdaModel

# Build LDA model
lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=6, random_state=100,
                chunksize=1000, passes=50,iterations=100)


lda_model.print_topics()

for i in range (6):  
    wp = lda_model.show_topic(i)
    topic_keywords = ", ".join([word for word, prop in wp])
    print("Topic:",i,"keywords are")
    print(topic_keywords )
    
    

print('\nPerplexity: ', lda_model.log_perplexity(doc_term_matrix,total_docs=2413))  # a measure of how good the model is. lower the better.

# Compute Coherence Score

coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized_reviews, dictionary=dictionary , coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

from matplotlib.pyplot import cm
colors = iter(cm.rainbow(np.linspace(0,1,6)))
for i in range(6):
    temp = pd.DataFrame(lda_model.show_topic(i))
    temp= temp.rename(columns={0:'keyword',1:'probability'})
    temp= temp.set_index('keyword')
    temp.plot(kind='barh',title="Topic %d"%i,color=next(colors))
##########################################
import pyLDAvis
import pyLDAvis.gensim as gensimvis
vis_data = gensimvis.prepare(lda_model, doc_term_matrix,dictionary)
vis_data
pyLDAvis.display(vis_data)  
pyLDAvis.save_html(vis_data, 'topic6'+'.html')
############################################
def format_topics_sentences(ldamodel=None, corpus=doc_term_matrix, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(lda_model[doc_term_matrix]):
        row = row_list[0] if lda_model.per_word_topics else row_list            
        #print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = lda_model.show_topic(topic_num)
                #print(wp)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    # Add original text to the end of the output
    sen = pd.Series(data['sentiment'])
    review  = pd.Series(data['review'])
    sent_topics_df = pd.concat([sent_topics_df,sen,review], axis=1)

    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model, corpus=doc_term_matrix, texts=df)
df_topic_sents_keywords.head(5)
df_topic_sents_keywords.to_csv('data_topic6.csv', index=False)
####################################################################################
#-----------------------------------------------------------------------------
# Number of article for each topic
gt= df_topic_sents_keywords['Dominant_Topic'].value_counts()
g = []
g =gt.to_frame()
g = g.rename(columns={'Dominant_Topic':'Number of review'})
g.plot(kind='bar',color='orange',xlabel = "Topic")
g.sort_index(ascending=True)
#------------------------------------------------------------------------------
#DataFrame by group

#DataFrame by group

df_topic = df_topic_sents_keywords.groupby('Dominant_Topic')
df_topic.describe()
#Topic 0

df_t0= df_topic.get_group(0)

sentiment0=df_t0['sentiment'].replace({0:'Negative review',1:"Positive review"})
sentiment0.value_counts().to_frame()
sentiment0.value_counts().plot(kind="bar",color='blue',title = "Topic 0")

#########################
df_t1= df_topic.get_group(1)

sentiment1=df_t1['sentiment'].replace({0:'Negative review',1:"Positive review"})
sentiment1.value_counts().to_frame()
sentiment1.value_counts().plot(kind="bar",color='blue',title = "Topic 1")


#########################################

df_t2= df_topic.get_group(2)

sentiment2=df_t2['sentiment'].replace({0:'Negative review',1:"Positive review"})
sentiment2.value_counts().to_frame()
sentiment2.value_counts().plot(kind="bar",color='blue',title = "Topic 2")
#########################################

df_t3= df_topic.get_group(3)

sentiment3=df_t3['sentiment'].replace({0:'Negative review',1:"Positive review"})
sentiment3.value_counts().to_frame()
sentiment3.value_counts().plot(kind="bar",color='blue',title = "Topic 3")
#################################################

df_t4= df_topic.get_group(4)

sentiment4=df_t4['sentiment'].replace({0:'Negative review',1:"Positive review"})
sentiment4.value_counts().to_frame()
sentiment4.value_counts().plot(kind="bar",color='blue',title = "Topic 4")


#########################################

df_t5= df_topic.get_group(5)

sentiment5=df_t5['sentiment'].replace({0:'Negative review',1:"Positive review"})
sentiment5.value_counts().to_frame()
sentiment5.value_counts().plot(kind="bar",color='blue',title = "Topic 5")

#######################








#create a pickle file
import pickle
pickle_out =open('Topicmodelk=6.pkl','wb')
pickle.dump(lda_model,pickle_out)
pickle_out.close()










