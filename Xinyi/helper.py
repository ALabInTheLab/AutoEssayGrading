import numpy as np
import pandas as pd

import sys

from gensim.models import Word2Vec
from gensim.models import Doc2Vec
 
from nltk.cluster import KMeansClusterer
from nltk.tokenize import RegexpTokenizer

class Helper:
    def __init__(self):
        return None

    def sent_vectorizer_average(self, sent, model, deminsion):
        sent_vec = np.zeros(deminsion)
        numw = 0
        for w in sent:
            try:
                sent_vec = np.add(sent_vec, model[w])
                numw+=1
            except:
                pass
        return sent_vec / np.sqrt(sent_vec.dot(sent_vec))


    def sent_vectorizer_concatenate(self, sent, model, deminsion):
        sent_vec = np.empty([0, deminsion])
        for w in sent:
            sent_vec = np.append(sent_vec, model[w].reshape(1,-1), axis=0)

        return sent_vec

    def readData(self, set_num):
        # training data
        df = pd.read_csv('../data/training_set_rel3.tsv', sep='\t', header = 0, encoding = "ISO-8859-1")


        tokenizer = RegexpTokenizer(r'\w+')
        for i in range(df.shape[0]):
            df.at[i,'essay'] = tokenizer.tokenize(df['essay'][i])
        
        dfs = []
        #sentences = []
        for i in range(8):
            dfs.append(df.loc[df['essay_set'] == i+1])
            #sentences.append(dfs[i]['essay'].values)
        
        sentences = dfs[int(set_num)]['essay'].values
        labels = dfs[int(set_num)]['domain1_score'].values
        
        # training model
        model = Word2Vec(sentences, min_count=1)
         
        # get vector data
        X = model[model.wv.vocab]

        print('Word vector shape:',X.shape)

        vocab = model.wv.vocab.keys()
        #print(X)
        #print(vocab)
         
        #V = np.empty((0,X.shape[1]))
        #V = np.empty((0,0,X.shape[1]))
        V = []
        for sentence in sentences:
            V.append(self.sent_vectorizer_concatenate(sentence, model, X.shape[1]))

        print("Number of instances: ", len(V) )
        print("V: ", V[0].shape )
        return V, labels

    # if __name__ == '__main__':
    #     readData(1)