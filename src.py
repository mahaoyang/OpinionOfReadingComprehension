# coding=gbk
import pickle
import random
import numpy as np
from model import model
from data import load_vectors, train

# with open('pasg.txt', 'r') as f:
#     pasg = f.readline()
#     print('pasg length : %s' % len(pasg))

# with open('index2word.pick', 'rb') as f:
#     index2word = pickle.load(f)
#     print(len(index2word))
# with open('passages.pick', 'rb') as f:
#     pasg = pickle.load(f)
# with open('query.pick', 'rb') as f:
#     query = pickle.load(f)
# with open('alts.pick', 'rb') as f:
#     alts = pickle.load(f)
#
# with open('answer.pick', 'rb') as f:
#     answer = pickle.load(f)

size = 1000


def dgen(batch_size=2):
    while 1:
        p, q, a, an = [], [], [], []
        for i in range(batch_size):
            r = random.randint(0, size)
            train_ = train(r)
            p.append(train_[1].tolist())
            q.append(train_[2].tolist())

            # a_t = []
            # for ii in train_[0]:
            #     for iii in ii:
            #         a_t.extend(iii)
            # a.append(a_t)

            a_t = []
            for ii in train_[0].tolist():
                a_t.extend(ii)
            a.append(a_t)
            an.append(train_[3].tolist())
        p = np.array(p)
        q = np.array(q)
        a = np.array(a)
        an = np.array(an)
        print(p.shape, q.shape, a.shape, an.shape)
        yield ([p, q, a], [a, an])


model = model()
model.fit_generator(dgen(), steps_per_epoch=100, epochs=30, validation_data=dgen(), validation_steps=20)
model.save('oporc_1.h5')
