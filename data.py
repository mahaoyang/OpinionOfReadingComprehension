import io
import json
import pickle
import random

import jieba
import numpy as np
from keras.preprocessing import text, sequence
from textrank4zh import TextRank4Sentence

random.seed(123)
np.random.seed(123)

PATH = 'C:/Users/99263/Downloads/oqmrc/'
TRAIN = 'ai_challenger_oqmrc_trainingset.json'
TEST = 'ai_challenger_oqmrc_validationset.json'
VALIODATION = 'ai_challenger_oqmrc_validationset.json'

MAX_ALT_NUM = 3
MAX_AN_LENGTH = 60
MAX_TRI_AN_LENGTH = 97
MAX_QUES_LENGTH = 30
MAX_PASSAGE_LENGTH = 300
MAX_WORD_INDEX = 30000


def shuf():
    with open(PATH + TRAIN, 'r', encoding='utf-8') as f:
        train = f.readlines()
        train = np.array(train)
        np.random.shuffle(train)
        train = train.tolist()
    with open('train_shuffle', 'wb') as f:
        pickle.dump(train, f)
    with open(PATH + TEST, 'r', encoding='utf-8') as f:
        test = f.readlines()
        test = np.array(test)
        np.random.shuffle(test)
        test = test.tolist()
    with open('test_shuffle', 'wb') as f:
        pickle.dump(test, f)
    with open(PATH + VALIODATION, 'r', encoding='utf-8') as f:
        validation = f.readlines()
        validation = np.array(validation)
        np.random.shuffle(validation)
        validation = validation.tolist()
    with open('val_shuffle', 'wb') as f:
        pickle.dump(validation, f)


def tr():
    all_words = {'\n'}
    alternatives = []
    passages = []
    querys = []
    answers = []
    train = pickle.load(open('train_shuffle', 'rb'))
    n = 0
    for i in train:
        n += 1
        if n > 1001:
            break
        i = json.loads(i)
        alternative = [' '.join(jieba.cut(ii, cut_all=True, HMM=False)) for ii in i.get('alternatives').split('|')]
        alternatives.append(alternative)
        passage = ' '.join(jieba.cut(i.get('passage').replace(' ', ''), cut_all=True,
                                     HMM=False)).replace('   ', ' ， ').replace('  ', ' 。 ')

        if len(passage.split(' ')) > 300:
            trs = TextRank4Sentence()
            trs.analyze(text=i.get('passage').replace(' ', ''), lower=True, source='all_filters')
            passages.append(' '.join(
                jieba.cut('。'.join([i.sentence for i in trs.get_key_sentences(1)])[0:300], cut_all=True,
                          HMM=False)).replace('   ', ' ， ').replace('  ', ' 。 '))
        else:
            passages.append(passage)

        query = ' '.join(jieba.cut(i.get('query').replace(' ', ''), cut_all=True,
                                   HMM=False)).replace('   ', ' ').replace('  ', '')
        querys.append(query)

        answer = ' '.join(jieba.cut(i.get('answer').replace(' ', ''), cut_all=True,
                                    HMM=False)).replace('   ', ' ').replace('  ', '')
        answers.append(answer)

        for ii in alternative:
            ii = set(ii.split(' '))
            all_words |= ii
        all_words |= set(passage.split(' ')) | set(query.split(' '))

    token = text.Tokenizer()
    token.fit_on_texts(all_words)

    with open('token.pick', 'wb') as f:
        pickle.dump([token, alternatives, passages, querys, answers], f)

    # index2word = token.index_word
    # word2index = token.word_index

    # alternatives_idx = []
    # for i in alternatives:
    #     alternatives_idx.append([sequence.pad_sequences(token.texts_to_sequences(i), maxlen=MAX_TRI_AN_LENGTH)])
    #
    # passages_idx = sequence.pad_sequences(token.texts_to_sequences(passages), maxlen=MAX_PASSAGE_LENGTH)
    #
    # querys_idx = sequence.pad_sequences(token.texts_to_sequences(querys), maxlen=MAX_QUES_LENGTH)
    #
    # answer_idx = sequence.pad_sequences(token.texts_to_sequences(answers), maxlen=MAX_AN_LENGTH)
    #
    # with open('index2word.pick', 'wb') as f:
    #     pickle.dump(index2word, f)
    #
    # with open('word2index.pick', 'wb') as f:
    #     pickle.dump(word2index, f)
    #
    # with open('alts.pick', 'wb') as f:
    #     pickle.dump(np.array(alternatives_idx), f)
    #
    # with open('query.pick', 'wb') as f:
    #     pickle.dump(np.array(querys_idx), f)
    #
    # with open('answer.pick', 'wb') as f:
    #     pickle.dump(np.array(answer_idx), f)
    #
    # with open('passages.pick', 'wb') as f:
    #     pickle.dump(np.array(passages_idx), f)
    #
    # # for iii in passages_idx:
    # #     with open('pasg.txt', 'a+') as f:
    # #         f.write('%s\n' % iii.tolist())
    # print('passages_idx : %s' % len(passages_idx))


def load_vectors(fname='cc.zh.300.vec'):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    data = dict()
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.asarray(tokens[1:], dtype='float32')
    return data


def ebd_matrix(embeddings_index, index2word, EMBEDDING_DIM):
    embedding_matrix = np.zeros((len(index2word) + 1, EMBEDDING_DIM))
    for i, word in index2word.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


# ii = np.asarray([w2v[index2word[iii]] for iii in pasg[r]], dtype='float32')
# p.append(ii)
#
# ii = np.asarray([w2v[index2word[iii]] for iii in query[r]], dtype='float32')
# q.append(ii)
#
# for iii in alts[r]:
#     ii = np.asarray([w2v[index2word[iiii]] for iiii in iii], dtype='float32')
#     a.append(ii)
#
# ii = np.asarray([w2v[index2word[iii]] for iii in answer[r]], dtype='float32')
# an.append(ii)

train = pickle.load(open('token.pick', 'rb'))
token = train[0]
alternatives = train[1]
passages = train[2]
querys = train[3]
answers = train[4]


def train(index):
    alternatives_idx = \
        sequence.pad_sequences(token.texts_to_sequences(alternatives[index]), maxlen=MAX_TRI_AN_LENGTH)

    passages_idx = sequence.pad_sequences(token.texts_to_sequences([passages[index]]), maxlen=MAX_PASSAGE_LENGTH)[0]

    querys_idx = sequence.pad_sequences(token.texts_to_sequences([querys[index]]), maxlen=MAX_QUES_LENGTH)[0]

    answer_idx = sequence.pad_sequences(token.texts_to_sequences([answers[index]]), maxlen=MAX_AN_LENGTH)[0]

    return alternatives_idx, passages_idx, querys_idx, answer_idx


if __name__ == '__main__':
    # tr()
    train(2)
