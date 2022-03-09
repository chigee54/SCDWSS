import glob
import os
import json
import tensorflow as tf
import numpy as np
import csv
from bert4keras.backend import keras, set_gelu, K
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Dropout, Dense, Lambda
from sklearn.metrics import f1_score

set_gelu('tanh')  # 切换gelu版本

maxlen = 128
batch_size = 9
config_path = '../english_uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../english_uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../english_uncased_L-12_H-768_A-12/vocab.txt'


train_path = 'data_analysis/split_small.txt'
validation_path = 'data_analysis/split_small_val.txt'

allkind = set()
data = open(train_path, 'r', encoding='utf-8')
for line in data:
    allkind.add(int(line.strip().split('\t')[0]))
allkind = list(allkind)
print(len(allkind))


def load_data(filename):
    """
    加载数据
    单条格式：(标签id, 句子sentence)
    """
    D = []

    with open(filename, 'r', encoding='utf-8') as f:
        for l in f:
            label, sentence = l.strip().split('\t')
            label_index = allkind.index(int(label))
            D.append((sentence, int(label_index)))
    return D


train_data = load_data(train_path)
valid_data = load_data(validation_path)


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (sentence, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(sentence, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    # with_pool=True,
    return_keras_model=False,
    num_hidden_layers=12
)

output1 = Lambda(lambda x: x[:, 0])(bert.model.output)
output = Dropout(rate=0.1)(output1)

output = Dense(
    units=2, activation='softmax'
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),  # 用足够小的学习率
    # optimizer=PiecewiseLinearLearningRate(Adam(5e-5), {10000: 1, 30000: 0.1}),
    metrics=['accuracy'],
)


def evaluate(data):
    total, right = 0., 0.
    for y_true, x_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        print(
            u'val_acc: %.5f, best_val_acc: %.5f\n' %
            (val_acc, self.best_val_acc)
        )


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=20,
        callbacks=[evaluator]
    )
    # model.load_weights('best_model.weights')
