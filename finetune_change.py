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

maxlen = 256
batch_size = 4
config_path = '../english_uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../english_uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../english_uncased_L-12_H-768_A-12/vocab.txt'


# train_path = 'demo_data20/train'
# validation_path = 'demo_data20/validation'
train_path = '../pan21/train'
validation_path = '../pan21/validation'


def read_label(label_file):
    labels = {}
    for label in glob.glob(os.path.join(label_file, 'truth-problem-*.json')):
        with open(label, 'r', encoding='utf-8') as lf:
            curr_label = json.load(lf)
            labels[os.path.basename(label)[14:-5]] = curr_label
    return labels


def val_load(filename):
    train_labels = read_label(filename)
    data = []
    for document_path in glob.glob(filename + '/*.txt'):
        # 读取每一个文本并赋予对应id
        with open(document_path, encoding="utf8") as file:
            document = file.read()
        share_id = os.path.basename(document_path)[8:-4]
        para_list = document.split('\n')
        change_labels = train_labels[share_id]['changes']
        para_pre = None
        for id, para in enumerate(para_list):
            if id == 0:
                para_pre = para
                continue
            label = change_labels[id-1]
            para_curr = para
            data.append((para_pre, para_curr, int(label)))
            para_pre = para
    return data


def separate_para_label(paragraphs_label):
    separate_label = []
    for i in range(len(paragraphs_label)):
        if i == 0:
            continue
        for a in range(i):
            if paragraphs_label[a] != paragraphs_label[i]:
                separate_label.append(1)
            else:
                separate_label.append(0)
    return separate_label


def train_load(filename):
    train_labels = read_label(filename)
    data_plus = []
    for document_path in glob.glob(filename + '/*.txt'):
        # 读取每一个文本并赋予对应id
        with open(document_path, encoding="utf8") as file:
            document = file.read()
        share_id = os.path.basename(document_path)[8:-4]
        para_list = document.split('\n')
        author_labels = train_labels[share_id]['paragraph-authors']
        separate_labels = separate_para_label(author_labels)
        for id, para in enumerate(para_list):
            for i in range(id):
                data_plus.append((para_list[i], para_list[id], int(separate_labels.pop(0))))
    return data_plus


train_data = train_load(train_path)
valid_data = val_load(validation_path)


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text1, text2, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text1, text2, maxlen=maxlen)
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
test_generator = data_generator(valid_data, batch_size)

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_pool=True,
    return_keras_model=False,
    # num_hidden_layers=12
)


# output1 = Lambda(lambda x: x[:, 0])(bert.model.output)
# output = Dropout(rate=0.1)(output1)
output = Dropout(rate=0.1)(bert.model.output)
output = Dense(
    units=2, activation='softmax'
)(output)

model = keras.models.Model(bert.model.input, output)
# model.summary()

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(2e-5),  # 用足够小的学习率
    metrics=['accuracy'],
)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
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
            model.save_weights('finetune_all_256/best_model.weights')
            model.save('finetune_all_256/best_model.h5')
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=3,
        callbacks=[evaluator]
    )

    model.load_weights('finetune_all_256/best_model.weights')
    with open('finetune_all_256/result.txt', 'a', encoding='utf-8') as result_file:
        result_file.write(u'final test acc: %05f\n' % (evaluate(test_generator)))
    print(u'final test acc: %05f\n' % (evaluate(test_generator)))
