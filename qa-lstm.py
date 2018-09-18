# -*- coding: utf-8 -*-
from __future__ import division, print_function
from gensim.models import KeyedVectors
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Merge, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.cross_validation import train_test_split
import numpy as np
import os

import kaggle

MODEL_DIR = "data/qa-lstm.json"
WORD2VEC_BIN = "data/corpusWord2Vec.bin"
WORD2VEC_EMBED_SIZE = 300
filepath = 'data/model_lstm.h5'
QA_TRAIN_FILE = "data/train_data2.txt"

QA_EMBED_SIZE = 64
BATCH_SIZE = 32
NBR_EPOCHS = 1

## extract data   asc

print("Loading and formatting data...")
qapairs = kaggle.get_question_answer_pairs(QA_TRAIN_FILE)

question_maxlen = max([len(qapair[0]) for qapair in qapairs])
answer_maxlen = max([len(qapair[1]) for qapair in qapairs])
seq_maxlen = max([question_maxlen, answer_maxlen])


word2idx = kaggle.build_vocab([], qapairs, [])
vocab_size = len(word2idx) + 1 # include mask character 0
print(vocab_size)
Xq, Xa, Y = kaggle.vectorize_qapairs(qapairs, word2idx, seq_maxlen)  ##长度变成一样

Xqtrain, Xqtest, Xatrain, Xatest, Ytrain, Ytest = \
    train_test_split(Xq, Xa, Y, test_size=0.3, random_state=42)
print(Xqtrain.shape, Xqtest.shape, Xatrain.shape, Xatest.shape,
      Ytrain.shape, Ytest.shape)

# get embeddings from word2vec
# see https://github.com/fchollet/keras/issues/853
print("Loading Word2Vec model and generating embedding matrix...")
word2vec = KeyedVectors.load_word2vec_format(WORD2VEC_BIN, binary=True)
embedding_weights = np.zeros((vocab_size, WORD2VEC_EMBED_SIZE))##87254
for word, index in word2idx.items():
    try:
        embedding_weights[index, :] = word2vec[word]
    except KeyError:
        pass  # keep as zero (not ideal, but what else can we do?)
del word2vec
del word2idx

print("Building model...")
qenc = Sequential()
qenc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,input_length=seq_maxlen,
                   weights=[embedding_weights], mask_zero=True))
qenc.add(LSTM(QA_EMBED_SIZE,  return_sequences=False))
qenc.add(Dropout(0.3))
## Dropout的意思就是训练和预测时随机减少特征个数，即去掉输入数据中的某些维度，用于防止过拟合。
#通过设置Dropout中的参数p，在训练和预测模型的时候，每次更新都会丢掉（总数*p）个特征，以达到防止过拟合的目的
#Fatten  ： 把多维输入转换为1维输入，名字很形象，就是把输入给压平了。
aenc = Sequential()
aenc.add(Embedding(output_dim=WORD2VEC_EMBED_SIZE, input_dim=vocab_size,input_length=seq_maxlen,
                   weights=[embedding_weights], mask_zero=True))
aenc.add(LSTM(QA_EMBED_SIZE,  return_sequences=False))
aenc.add(Dropout(0.3))

model = Sequential()
model.add(Merge([qenc, aenc], mode="sum"))  
#  把layers(or containers) list合并为一个层，用以下三种模式中的一种：sum，mul或 concat。
model.add(Dense(2, activation="softmax"))  ##Dense类(标准的一维全连接层)

model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy"])

print("Training...")
checkpoint = ModelCheckpoint(filepath,verbose=1, save_best_only=True)
model.fit([Xqtrain, Xatrain], Ytrain, batch_size=BATCH_SIZE,
          nb_epoch=NBR_EPOCHS, validation_split=0.1,
          callbacks=[checkpoint])

print("Evaluation...")
loss, acc = model.evaluate([Xqtrain, Xatrain], Ytrain, batch_size=BATCH_SIZE)
print("Train loss/accuracy final model = %.4f, %.4f" % (loss, acc))


model.save_weights(filepath)
with open(MODEL_DIR, "w",encoding='utf-8') as fjson:
    fjson.write(model.to_json())

model.load_weights(filepath)
loss, acc = model.evaluate([Xqtest, Xatest], Ytest, batch_size=BATCH_SIZE)
print("Test loss/accuracy best model = %.4f, %.4f" % (loss, acc))
