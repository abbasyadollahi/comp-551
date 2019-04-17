import pickle
import numpy as np
import pandas as pd
import os.path as op
from keras import backend as K
from keras.models import Model
from keras.layers.core import Activation
from keras.layers import Input, Embedding, Flatten, dot
from keras.optimizers import Adam
from sklearn.feature_extraction.text import CountVectorizer

def build_nbsvm(dtm_train, labels, num_words):
    x = []
    nwds = []
    maxlen = 2000
    for row in dtm_train:
        seq = []
        indices = (row.indices + 1).astype(np.int64)
        np.append(nwds, len(indices))
        data = (row.data).astype(np.int64)
        count_dict = dict(zip(indices, data))
        for k,v in count_dict.items():
            seq.extend([k]*v)
        num_words = len(seq)
        nwds.append(num_words)
        # pad up to maxlen with 0 else truncate down to maxlen
        seq = np.pad(seq, (maxlen-num_words, 0), mode='constant') if num_words < maxlen else seq[-maxlen:]
        x.append(seq)
    nwds = np.array(nwds)
    print(f'Sequence stats: avg: {nwds.mean()}, max: {nwds.max()}, min: {nwds.min()}')
    x_train = np.array(x)

    # def pr(dtm, y, y_i):
    #     return ((np.array(x)[y==y_i]).sum(0)+1) / ((y==y_i).sum()+1)

    nbratios = np.log(pr(dtm_train, labels, 1) / pr(dtm_train, labels, 0))
    nbratios = np.squeeze(np.asarray(nbratios))

    return get_model(num_words, maxlen, nbratios), x_train

def pr(dtm, y, y_i):
    p = dtm[y==y_i].sum(0)
    return (p+1) / ((y==y_i).sum()+1)

def get_model(num_words, maxlen, nbratios=None):
    embedding_matrix = np.zeros((num_words, 1))
    for i in range(1, num_words): # skip 0, the padding value
        embedding_matrix[i] = nbratios[i-1] if nbratios is not None else 1

    inp = Input(shape=(maxlen,))
    r = Embedding(num_words, 1, input_length=maxlen, weights=[embedding_matrix], trainable=False)(inp)
    x = Embedding(num_words, 1, input_length=maxlen, embeddings_initializer='glorot_normal')(inp)
    x = dot([r,x], axes=1)
    x = Flatten()(x)
    x = Activation('sigmoid')(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
    return model
