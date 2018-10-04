# coding=gbk
import pickle
import random
import numpy as np
from model import model
from data import load_vectors

with open('pasg.txt', 'r') as f:
    pasg = f.readlines()
    print('pasg length : %s' % len(pasg))
with open('query.pick', 'rb') as f:
    query = pickle.load(f)
with open('alts.pick', 'rb') as f:
    alts = pickle.load(f)

with open('answer.pick', 'rb') as f:
    answer = pickle.load(f)


def dgen(batch_size=32):
    while 1:
        p, q, a, an = [], [], [], []
        for i in range(batch_size):
            r = random.randint(0, len(pasg))
            p.append(eval(pasg[r]))
            q.append(query[r])
            for i in alts[r]:
                a.extend(i)
            an.append(answer[r])
        p = np.asarray(p, dtype='float32')
        q = np.array(p)
        a = np.array(a)
        a = np.array(an)
        yield [p, q, a], [a, an]


model = model()
model.fit_generator(dgen(), steps_per_epoch=100, epochs=30, validation_data=dgen(), validation_steps=20)
model.save('oporc_1.h5')
