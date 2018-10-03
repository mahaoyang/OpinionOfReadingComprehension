import os
import pickle

import numpy as np

from keras import utils
from keras import layers
from keras import models
from keras import optimizers
from keras import applications

from attention import Position_Embedding, Attention
from data import load_vectors, ebd_matrix

MAX_ALT_NUM = 3
MAX_AN_LENGTH = 60
MAX_TRI_AN_LENGTH = 97
MAX_QUES_LENGTH = 30
MAX_PASSAGE_LENGTH = 300
MAX_WORD_INDEX = 30000

GLOVE_DIR = 'C:/Users/99263/Downloads/oqmrc'

with open('index2word.pick', 'rb') as f:
    index2word = pickle.load(f)
word_vector = load_vectors()
embedding_matrix = ebd_matrix(embeddings_index=word_vector, index2word=index2word, EMBEDDING_DIM=300)


def model():
    passage_input = layers.Input(shape=(MAX_PASSAGE_LENGTH,), dtype='int32')
    passage = layers.Embedding(MAX_WORD_INDEX + 1,
                               300,
                               weights=[embedding_matrix],
                               input_length=MAX_PASSAGE_LENGTH,
                               trainable=False)(passage_input)
    passage = Position_Embedding()(passage)
    question_input = layers.Input(shape=(MAX_QUES_LENGTH,))
    question = layers.Embedding(MAX_WORD_INDEX + 1,
                                300,
                                weights=[embedding_matrix],
                                input_length=MAX_PASSAGE_LENGTH,
                                trainable=False)(question_input)
    question = Position_Embedding()(question)
    alternatives_input = layers.Input(shape=(MAX_TRI_AN_LENGTH,))
    alternatives = layers.Embedding(MAX_WORD_INDEX + 1,
                                    300,
                                    weights=[embedding_matrix],
                                    input_length=MAX_PASSAGE_LENGTH,
                                    trainable=False)(alternatives_input)

    p_encoder = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(passage)
    q_encoder = layers.Bidirectional(layers.LSTM(256, return_sequences=True))(question)

    a_decoder = Attention(8, 16)([p_encoder, q_encoder, alternatives])
    a_decoder = layers.Flatten()(a_decoder)
    a_decoder = layers.Concatenate()([a_decoder, alternatives_input])

    output_alt = layers.Dense(MAX_ALT_NUM)(a_decoder)
    output = layers.Dense(MAX_ALT_NUM, activation='softmax')(a_decoder)

    rc_model = models.Model(inputs=[passage_input, question_input, alternatives_input], output=[output_alt, output])
    opti = optimizers.Adam(lr=1e-1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    rc_model.compile(optimizer=opti, loss="categorical_crossentropy", metrics=["accuracy"])

    rc_model.summary()
    return rc_model


model()
