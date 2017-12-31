# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 12:07:04 2017

@author: vgupta
"""

from sklearn.feature_extraction.text import CountVectorizer 
import re
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk import word_tokenize
import numpy as np
from nltk.tag.stanford import StanfordNERTagger
from collections import Counter, OrderedDict
import gensim
from nltk import RegexpTokenizer
from gensim.models.phrases import Phraser
from gensim.models import Phrases
from sklearn.feature_extraction.text import CountVectorizer     



def nlp_clean(data):
   new_data = []
   tokenizer = RegexpTokenizer(r'\w+')
   stopword_set = set(stopwords.words('english'))
   for d in data:
      new_str = d.lower()
      dlist = tokenizer.tokenize(new_str)
      dlist = list(set(dlist).difference(stopword_set))
      new_data.append(dlist)
   return new_data
    
class LabeledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list
    def __iter__(self):

        for idx, doc in enumerate(self.doc_list):
            #print(type(self.labels_list[idx]))
            yield (gensim.models.doc2vec.LabeledSentence(doc,self.labels_list[idx]))

class QueryGenerator:
    def __init__(self,Name):
        self.docName = Name
        
    def splitSentence(self):
        sent_tokenize_list = sent_tokenize(self.docName)
        return sent_tokenize_list
        
    def constructNGramsBOW(self):
        
        text = self.docName
        stopsWords = set(stopwords.words('english'))
        text = re.sub("[^a-zA-Z]", " ", text.lower()) 
        text = re.sub("\s\s+", " ", text)
        #bigram = Phraser.load('mymodel/bigram_phraser_wikien2017')
        #trigram = Phraser.load('mymodel/trigram_phraser_wikien2017')
        #sent_tokenize_list = sent_tokenize(text)
        with open("data/embed.vocab") as f:
            vocab_list = map(str.strip, f.readlines())
        #for line in sent_tokenize_list:
        #    sent = word_tokenize(line)
        #    line = trigram[bigram[sent]]
        #    line = [w for w in line if not w in stopsWords ]
        with open("data/embed.vocab") as f:
            vocab_list = map(str.strip, f.readlines())
            vocab_dict = {w: k for k, w in enumerate(vocab_list)}
        
        vectorizer = CountVectorizer(ngram_range=(1,3))
        analyzer = vectorizer.build_analyzer()
        sett = analyzer(text)

        sett = [token.replace(" ","_") for token in sett]
        BOWNgram = [token for token in sett if token in vocab_dict.keys()]
        BOWNgram = Counter(BOWNgram)
        return OrderedDict(BOWNgram)
        
    def NER(self):
        tags =[]
    
        mainTag=[]
        sentences = self.splitSentence()
        # loading this again and again CAUSES TIME DELAY
        st = StanfordNERTagger('/home/IAIS/vgupta/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz','/home/IAIS/vgupta/stanford-ner/stanford-ner.jar',encoding='utf-8')
        #st = StanfordNERTagger('/home/IAIS/vgupta/stanford-ner/classifiers/english.muc.7class.distsim.crf.ser.gz','/home/IAIS/vgupta/stanford-ner/stanford-ner.jar',encoding='utf-8') 
        for sent in sentences:
            sent = re.sub("[^a-zA-Z]", " ", sent) 
            sent = word_tokenize(sent)
            tags.extend(st.tag(sent))
        
        for a in tags:
            if a[1]!='O':
                mainTag.append(a)
        #print(set(mainTag))
        return set(mainTag)
             
        
    def keywords(self):
        """Get the top 10 keywords and their frequency scores ignores blacklisted
        words in stopwords, counts the number of occurrences of each word, and
        sorts them in reverse natural order (so descending) by number of
        occurrences.
        """
        
        NUM_KEYWORDS = 10
        text = self.docName
        # of words before removing blacklist words
        if text:
            num_words = len(text)
            text = re.sub("[^a-zA-Z]", " ", text)
            stopsWords = set(stopwords.words('english'))

            text = [x for x in text.lower().split() if x not in stopsWords]
            freq = {}
            for word in text:
                if word in freq:
                    freq[word] += 1
                else:
                    freq[word] = 1

            min_size = min(NUM_KEYWORDS, len(freq))
            keywords = sorted(freq.items(),key=lambda x: (x[1], x[0]),reverse=True)
            keywords = keywords[:min_size]
            keywords = dict((x, y) for x, y in keywords)

            for k in keywords:
                articleScore = keywords[k] * 1.0 / max(num_words, 1)
                keywords[k] = articleScore * 1.5 + 1

            return OrderedDict(keywords)
        else:
            return dict()
        
    def doc2vec(self,link):
       
        #fname ='doc2vec.model'
        #docLabels = [link]
        data = [self.docName]            
        data = nlp_clean(data)
        #it = LabeledLineSentence(data, docLabels)
        #corpus_count = len(docLabels)
        #model = gensim.models.doc2vec.Doc2Vec.load(fname)
        #model.build_vocab(it)
        #model.intersect_word2vec_format('vectors-phrase.bin', binary=True)
        #model.train(it, total_examples=corpus_count, epochs=model.iter)
        #model.save('doc2vec.model')

        d2v_model = gensim.models.doc2vec.Doc2Vec.load('doc2vec.model')
        #docvec = d2v_model.docvecs[link]
        #docvec = d2v_model.docvecs.infer_vector(link)
        docvec = d2v_model.infer_vector(doc_words=data[0], steps=20, alpha=0.025)
        return docvec

    def CentroidOfDocument(self, BOWnGrams,EmdWords):
        dim = 200
        
        #print("centroid")
        BOW = [x for x in BOWnGrams]
        W = np.memmap("data/embed.dat", dtype=np.double, mode="r", shape= ((EmdWords, dim)))
        with open("data/embed.vocab") as f:
            vocab_list = map(str.strip, f.readlines())
        vocab_dict = {w: k for k, w in enumerate(vocab_list)}
        
        numberWords = len(BOW)
        VecMatrixWords =np.zeros((numberWords, dim))
        for j,t in enumerate(BOW):
            VecMatrixWords[j] = (W[[vocab_dict[t]]])
        centroid =np.zeros(dim)
        for i in range(numberWords):
            centroid = centroid+ VecMatrixWords[i]
        centroid = centroid/numberWords
        return list(centroid)
