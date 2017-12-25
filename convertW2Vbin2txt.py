import sys
import os
import numpy as np
from gensim import models

from gensim.models.keyedvectors import KeyedVectors
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

#model = word2vec.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
model.init_sims(replace=True)
if not os.path.exists("data/embed.dat"):
    print("Caching word embeddings in memmapped format...")
    fp = np.memmap("data/embed.dat", dtype=np.double, mode='w+', shape=model.syn0norm.shape)
    fp[:] = model.syn0norm[:]
if not os.path.exists("data/vocab.dat"):
    with open("data/embed.vocab", "w") as f:
        for _, w in sorted((voc.index, word) for word, voc in model.vocab.items()):
            print(w, file=f)
    del fp, model
W = np.memmap("data/embed.dat", dtype=np.double, mode="r", shape=(3000000, 300))
with open("data/embed.vocab") as f:
    vocab_list = map(str.strip, f.readlines())

