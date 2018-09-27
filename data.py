import json
import random

import jieba
import numpy as np

random.seed(123)
np.random.seed(123)

PATH = 'C:/Users/99263/Downloads/oqmrc/'
TRAIN = 'ai_challenger_oqmrc_trainingset.json'
TEST = 'ai_challenger_oqmrc_validationset.json'
VALIODATION = 'ai_challenger_oqmrc_validationset.json'

with open(PATH + TRAIN, 'r', encoding='utf-8') as f:
    train = f.readlines()
    # train = np.array(train)
    # np.random.shuffle(train)
    # train = train.tolist()
with open(PATH + TEST, 'r', encoding='utf-8') as f:
    test = f.readlines()
    # test = np.array(test)
    # np.random.shuffle(test)
    # test = test.tolist()
with open(PATH + VALIODATION, 'r', encoding='utf-8') as f:
    validation = f.readlines()
    # validation = np.array(validation)
    # np.random.shuffle(validation)
    # validation = validation.tolist()

a = 0
b = 0
c = 0
d = 0
for i in train:
    i = json.loads(i)
    alternatives = ['<wd>'.join(jieba.cut(ii, cut_all=True, HMM=False)) for ii in i.get('alternatives').split('|')]
    passage = '<wd>'.join(jieba.cut(i.get('passage'), cut_all=True, HMM=False)).replace('<wd><wd><wd>', '<wd>').replace(
        '<wd><wd>', '')
    query = '<wd>'.join(jieba.cut(i.get('query'), cut_all=True, HMM=False)).replace('<wd><wd><wd>', '<wd>')
    a = a if max([len(i.split('<wd>')) for i in alternatives]) <= a else max(
        [len(i.split('<wd>')) for i in alternatives])
    b = b if len(passage.split('<wd>')) <= b else len(passage.split('<wd>'))
    d += len(passage.split('<wd>'))
    if len(passage.split('<wd>')) == 13452:
        print(passage)
    c = c if len(query.split('<wd>')) <= c else len(query.split('<wd>'))
print(a, b, c, d / len(train))
