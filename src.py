# coding=gbk
import pickle
import random
import numpy as np
from model import model
from data import load_vectors


def dgen(batch_size=32):
    with open('pasg.txt', 'r') as f:
        pasg = f.readlines()
    with open('query.pick', 'rb') as f:
        query = pickle.load(f)
    with open('alts.pick', 'rb') as f:
        alts = pickle.load(f)

    with open('answer.pick', 'rb') as f:
        answer = pickle.load(f)

    with open('index2word.pick', 'rb') as f:
        index2word = pickle.load(f)

    w2v = load_vectors()
    while 1:
        p, q, a, an = [], [], [], []
        for i in range(batch_size):
            r = random.randint(0, len(pasg))
            p.append(pasg[r])
            q.append(query[r])
            for i in alts[r]:
                a.extend(i)
            an.append(answer[r])

        yield [p, q, a], [a, an]


model = model()
model.fit_generator(dgen(), steps_per_epoch=100, epochs=30, validation_data=dgen(), validation_steps=20)
model.save('oporc_1.h5')
