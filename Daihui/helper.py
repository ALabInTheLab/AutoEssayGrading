import numpy as np
import pandas as pd

import sys, time

from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from nltk.tokenize import RegexpTokenizer

class Helper:
    def __init__(self, set_num):
        self.readData(set_num)

    def sent_vectorizer_average(self, sent, model, deminsion):
        sent_vec = np.zeros(deminsion)
        for w in sent:
            sent_vec = np.add(sent_vec, model[w])

        return sent_vec / np.sqrt(sent_vec.dot(sent_vec))


    def sent_vectorizer_concatenate(self, sent, model, deminsion):
        sent_vec = np.empty([0, deminsion])
        for w in sent:
            sent_vec = np.append(sent_vec, model[w].reshape(1,-1), axis=0)

        return sent_vec

    def readData(self, set_num):
        # training data
        print("Reading from: ../data/training_set_rel3.tsv ")
        df = pd.read_csv('../data/training_set_rel3.tsv', sep='\t', header = 0, encoding = "ISO-8859-1")

        tokenizer = RegexpTokenizer(r'\w+')
        for i in range(df.shape[0]):
            df.at[i,'essay'] = tokenizer.tokenize(df['essay'][i])
        
        dfs = []
        #sentences = []
        for i in range(8):
            dfs.append(df.loc[df['essay_set'] == i+1])
            #sentences.append(dfs[i]['essay'].values)
        
        self.sentences = dfs[int(set_num)]['essay'].values
        self.labels = dfs[int(set_num)]['domain1_score'].values
    
    def getDoc2Vec():
        documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentences)]
        model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
        
    def trainWord2Vec(self):
        # training model
        self.model = Word2Vec(self.sentences, min_count=1)
         
        # get vector data
        self.X = self.model[self.model.wv.vocab]

        print('Word vector shape:', self.X.shape)

        vocab = self.model.wv.vocab.keys()
        #print(X)
        #print(vocab)
    
    def getAverage(self):
        self.trainWord2Vec()
        V = np.empty((0,X.shape[1]))
        for sentence in self.sentences:
            V = np.append(V, self.sent_vectorizer_average(sentence, self.model, self.X.shape[1]).reshape(1,-1), axis=0)

        print("Data size: ", V.shape)

        return V, self.labels

    def getPadding3D(self):
        self.trainWord2Vec()
        V = []
        for sentence in self.sentences:
            V.append(self.sent_vectorizer_concatenate(sentence, self.model, self.X.shape[1]))
        print("Number of instances: ", len(V))

        maxLength = 0
        for i in range(len(V)):
            temp = V[i].shape[0]
            if temp > maxLength:
                maxLength = temp
        print("Max essay length:", maxLength)

        V_padding = np.empty((0,maxLength,self.X.shape[1]))
        print(V_padding.shape)
        for i in range(len(V)):
            if V[i].shape[0] < maxLength:
                padding = maxLength - V[i].shape[0]
                V[i] = np.append(V[i], np.zeros((padding, self.X.shape[1])), axis=0)

            V_padding = np.append(V_padding, V[i].reshape((1, maxLength, self.X.shape[1])), axis=0)
            print(V_padding.shape)

        return V_padding, maxLength, self.labels

    def getPadding2D(self):
        self.trainWord2Vec()
        V = []
        for sentence in self.sentences:
            V.append(self.sent_vectorizer_concatenate(sentence, self.model, self.X.shape[1]))
        print("Number of instances: ", len(V))

        maxLength = 0
        for i in range(len(V)):
            temp = V[i].shape[0]
            if temp > maxLength:
                maxLength = temp
        print("Max essay length:", maxLength)

        V_padding = np.empty((0, self.X.shape[1]))
        print(V_padding.shape)
        for i in range(len(V)):
            print(i)
            if V[i].shape[0] < maxLength:
                padding = maxLength - V[i].shape[0]
                V[i] = np.append(V[i], np.zeros((padding, self.X.shape[1])), axis=0)

            V_padding = np.append(V_padding, V[i].reshape((maxLength, self.X.shape[1])), axis=0)

        return V_padding, maxLength, self.labels