import numpy as np
import pandas as pd

import sys, time

from gensim.models import Word2Vec
# from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from nltk.tokenize import RegexpTokenizer

class Helper:
    def __init__(self, set_num, file_name):
        self.readData(set_num, file_name)
        # self.readData(set_num)

    def sent_vectorizer_average(self, sent, model, deminsion):
        sent_vec = np.zeros(deminsion)
        for w in sent:
            sent_vec = np.add(sent_vec, model[w])

        return sent_vec / np.sqrt(sent_vec.dot(sent_vec))


    def sent_vectorizer_concatenate(self, sent, model, dimension):
        # sent_vec = np.empty([0, deminsion])
        # for w in sent:
        #     sent_vec = np.append(sent_vec, model[w].reshape(1,-1), axis=0)
        #
        # return sent_vec

        num_w = len(sent)
        sent_vec = np.zeros([num_w, dimension])
        for idx, w in enumerate(sent):
            sent_vec[idx] = model[w].reshape(1, -1)

        return sent_vec


    def readData(self, set_num, file_name='../data/training_set_rel3.tsv'):
        # training data
        print("Reading from:" + file_name)
        df = pd.read_csv(file_name, sep='\t', header = 0, encoding = "ISO-8859-1")

        tokenizer = RegexpTokenizer(r'\w+')
        for i in range(df.shape[0]):
            df.at[i,'essay'] = tokenizer.tokenize(df['essay'][i])
            # if i > 1782:
            #     dbstop = 1
        
        dfs = []
        #sentences = []
        for i in range(8):
            dfs.append(df.loc[df['essay_set'] == i+1])
            #sentences.append(dfs[i]['essay'].values)
        
        self.sentences = dfs[int(set_num)]['essay'].values
        self.labels = dfs[int(set_num)]['domain1_score'].values
    
    # def getDoc2Vec(self):
    #     documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(sentences)]
    #     model = Doc2Vec(documents, vector_size=5, window=2, min_count=1, workers=4)
        
    def trainWord2Vec(self):
        # training model
        self.model = Word2Vec(self.sentences, min_count=1)
         
        # get vector data
        self.X = self.model[self.model.wv.vocab]

        print('Word vector shape:', self.X.shape)

        vocab = self.model.wv.vocab.keys()
        #print(X)
        #print(vocab)
    
    # def getAverage(self):
    #     self.trainWord2Vec()
    #     V = np.empty((0, X.shape[1]))
    #     for sentence in self.sentences:
    #         V = np.append(V, self.sent_vectorizer_average(sentence, self.model, self.X.shape[1]).reshape(1,-1), axis=0)
    #
    #     print("Data size: ", V.shape)
    #
    #     return V, self.labels

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

        # padding from
        # V_padding = np.empty((0,maxLength,self.X.shape[1])) # X is the vocabulary

        V_padding = np.zeros((len(V), maxLength, self.X.shape[1]))

        # time consuming!
        print(V_padding.shape)
        for idx in range(len(V)):
            # if V[idx].shape[0] < maxLength:
            #     padding = maxLength - V[idx].shape[0]
            #     V[idx] = np.append(np.zeros((padding, self.X.shape[1])), V[idx], axis=0)

            # V_padding = np.append(V_padding, V[art].reshape((1, maxLength, self.X.shape[1])), axis=0)
            V_padding[idx, :V[idx].shape[0]] = V[idx]

            print(V_padding.shape)

        return V_padding, maxLength, self.labels

    def get_embed(self):
        self.trainWord2Vec()
        essays = []
        for sentence in self.sentences:
            essays.append(self.sent_vectorizer_concatenate(sentence, self.model, self.X.shape[1]))
        print("Number of instances: ", len(essays))

        maxLength = 0
        for i in range(len(essays)):
            temp = essays[i].shape[0]
            if temp > maxLength:
                maxLength = temp
        print("Max essay length:", maxLength)

        # # padding from
        # V_padding = np.empty((0,maxLength,self.X.shape[1])) # X is the vocabulary, X.shape[1] is the embedding size
        #
        # # time consuming!
        # print(V_padding.shape)
        # for art in range(len(V)):
        #     if V[art].shape[0] < maxLength:
        #         padding = maxLength - V[art].shape[0]
        #         V[art] = np.append(np.zeros((padding, self.X.shape[1])), V[art], axis=0)
        #
        #     V_padding = np.append(V_padding, V[art].reshape((1, maxLength, self.X.shape[1])), axis=0)
        #
        #     print(V_padding.shape)

        return essays, maxLength, self.labels



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
                V[i] = np.append(np.zeros((padding, self.X.shape[1])), V[i], axis=0)

            V_padding = np.append(V_padding, V[i].reshape((maxLength, self.X.shape[1])), axis=0)

        return V_padding, maxLength, self.labels
