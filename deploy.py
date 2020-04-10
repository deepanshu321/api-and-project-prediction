# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 00:56:52 2020

@author: Deepanshu.pal
"""

from flask import Flask, request, jsonify, render_template
import pickle
import string

app = Flask(__name__)
import numpy as np
import pandas as pd
import string
import nltk
import os
java_path = "C:/Program Files/Java/jdk1.8.0_231/bin/java.exe"
os.environ['JAVAHOME'] = java_path
from nltk.tag import StanfordPOSTagger
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
def newone(inpt):
     data=pd.read_csv('combinedproject.csv')
     jar ='C:/Users/Deepanshu.pal/Documents/deepanshu/stanford-postagger-2018-10-16/stanford-postagger.jar'
     model ='C:/Users/Deepanshu.pal/Documents/deepanshu/stanford-postagger-2018-10-16/models/english-left3words-distsim.tagger'
     pos_tagger = StanfordPOSTagger(model, jar,encoding='utf8')
     data['bag'] = data.apply(lambda row:nltk.word_tokenize( row["Description"] ),axis=1)
     stop_words = set(stopwords.words('english')) 
     data['bag_remove_stopwords'] =data.apply(lambda ro: [w for w in ro['bag'] if not w in stop_words],axis=1) 
     
            
     def text_process(mess):
         nopunc = [char for char in mess if char not in string.punctuation]
            
         nopunc = ''.join(nopunc)
         stem_data=[ps.stem(w) for w in nopunc.split()]
         return [word for word in stem_data if word.lower() not in stopwords.words('english')]
            
     ps = PorterStemmer()   
     data['bag_stemming'] =data.apply(lambda ro: [ps.stem(w) for w in ro['bag_remove_stopwords'] ],axis=1)
            
     data['Description'].head(5).apply(text_process)
     bow_transformer = CountVectorizer(analyzer=text_process).fit(data['Description'])
     messages_bow = bow_transformer.transform(data['Description'])
     tfidf_transformer = TfidfTransformer().fit(messages_bow)
     messages_tfidf = tfidf_transformer.transform(messages_bow)
            
     new_project_description=inpt
     new_project_bow = bow_transformer.transform([new_project_description])
     new_project_vector = tfidf_transformer.transform(new_project_bow)
     result_array=cosine_similarity(new_project_vector, messages_tfidf)
     similar_score_array=result_array.flatten()
     data['similar_score']=similar_score_array
     return data.sort_values(["similar_score"], axis=0, 
                                 ascending=False)
            

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    
    final_features = request.form.values()
    data = newone(final_features)

    
    r=pd.DataFrame()
    r['Title']=data['Title']
    r['APIsUsed (a.k.a. npm dependencies used)']=data['APIsUsed (a.k.a. npm dependencies used)']
    output=np.array(r)

    return render_template('index.html', data=output)


