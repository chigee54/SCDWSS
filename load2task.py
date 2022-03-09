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
from itertools import chain
from keras.models import load_model

set_gelu('tanh')  # 切换gelu版本

maxlen = 256
BATCH_SIZE = 2
config_path = '../english_uncased_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../english_uncased_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '../english_uncased_L-12_H-768_A-12/vocab.txt'
# dict_path = '/home/zhangzhijie21/venu/vocab.txt'
# config_path = '/home/zhangzhijie21/venu/bert_config.json'
# checkpoint_path = '/home/zhangzhijie21/venu/bert_model.ckpt'
# train_path = 'demo_data20/train'
# validation_path = 'demo_data20/validation'
validation_path = '../pan21/validation'
# validation_path = '/media/training-datasets/pan21-style-change-detection/pan21-style-change-detection-dev'


def read_label(label_file):
    labels = {}
    for label in glob.glob(os.path.join(label_file, 'truth-problem-*.json')):
        with open(label, 'r', encoding='utf-8') as lf:
            curr_label = json.load(lf)
            labels[os.path.basename(label)[14:-5]] = curr_label
    return labels


# 将任务3标签拆分为1,0
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


para_len_plus = []
def data_load(filename):
    train_labels = read_label(filename)
    data = []
    data_plus = []
    for document_path in glob.glob(filename + '/*.txt'):
        # 读取每一个文本并赋予对应id
        with open(document_path, encoding="utf8") as file:
            document = file.read()
        share_id = os.path.basename(document_path)[8:-4]
        para_list = document.split('\n')
        change_labels = train_labels[share_id]['changes']
        author_labels = train_labels[share_id]['paragraph-authors']
        separate_labels = separate_para_label(author_labels)
        if len(para_list)-1 != len(change_labels):
            print(share_id)
            para_list.pop(-1)
        para_len_plus.append((share_id, len(change_labels), len(separate_labels)))
        para_pre = None
        for id, para in enumerate(para_list):
            if id == 0:
                para_pre = para
                continue
            label = change_labels[id-1]
            para_curr = para
            data.append((para_pre, para_curr, int(label)))
            para_pre = para
            for i in range(id):
                data_plus.append((para_list[i], para_list[id], int(separate_labels.pop(0))))
    return data, data_plus


# train_data = data_load(train_path)
valid_data, valid_data_plus = data_load(validation_path)


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
# def data_generator(data, batchsize):
#     batch_token_ids, batch_segment_ids, batch_labels = [], [], []
#     for data_sp in data:
#         token_ids, segment_ids = tokenizer.encode(data_sp[0], data_sp[1], maxlen=maxlen)
#         batch_token_ids.append(token_ids)
#         batch_segment_ids.append(segment_ids)
#         batch_labels.append([data_sp[2]])
#         if len(batch_token_ids) == batchsize:
#             batch_token_ids = sequence_padding(batch_token_ids)
#             batch_segment_ids = sequence_padding(batch_segment_ids)
#             batch_labels = sequence_padding(batch_labels)
#             yield [batch_token_ids, batch_segment_ids], batch_labels
#             batch_token_ids, batch_segment_ids, batch_labels = [], [], []
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
# train_generator = data_generator(train_data, BATCH_SIZE)
valid_generator = data_generator(valid_data, BATCH_SIZE)
# test_generator = data_generator(valid_data, BATCH_SIZE)
valid_plus_generator = data_generator(valid_data_plus, BATCH_SIZE)

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

# model.compile(
#     loss='sparse_categorical_crossentropy',
#     optimizer=Adam(2e-5),  # 用足够小的学习率
#     metrics=['sparse_categorical_accuracy'],
# )


# 把真实标签读取出来并赋予对应id，存为字典
def read_ground_truth_files(truth_folder):
    truth = {}
    for truth_file in glob.glob(os.path.join(truth_folder, 'truth-problem-*.json')):
        with open(truth_file, 'r') as fh:
            curr_truth = json.load(fh)
            truth[os.path.basename(truth_file)[14:-5]] = curr_truth
    return truth


# 把分类器预测出来的标签读取出来并赋予对应id，存为字典
def read_solution_files(solutions_folder):
    solutions = {}
    for solution_file in glob.glob(os.path.join(solutions_folder, 'solution-problem-*.json')):
        with open(solution_file, 'r') as fh:
            curr_solution = json.load(fh)
            solutions[os.path.basename(solution_file)[17:-5]] = curr_solution
    return solutions


def extract_task_results(truth, solutions, task):
    all_solutions = []
    all_truth = []
    for problem_id, truth_instance in truth.items():
        all_truth.append(truth_instance[task])
        try:
            all_solutions.append(solutions[problem_id][task])
        except KeyError as _:
            print("No solution file found for problem %s, exiting." % problem_id)
            exit(0)
    return all_truth, all_solutions


def compute_score_single_predictions(truth, solutions, task):
    truth, solution = extract_task_results(truth, solutions, task)
    return f1_score(truth, solution, average='micro')


def compute_score_multiple_predictions(truth, solutions, task, labels):
    task2_truth, task2_solution = extract_task_results(truth, solutions, task)
    # task 2 - lists have to be flattened first
    return f1_score(list(chain.from_iterable(task2_truth)),
                    list(chain.from_iterable(task2_solution)), average='macro', labels=labels)


# 将预测的标签合成为任务3的标准标签
def compound_label(separate_label_list, length):
    paragraphs_label = [1]
    each_para_label = 0
    dict_label = {}
    pre_index_left = 0
    pre_index_right = 0
    pre_index = 0
    for index in range(length):
        if index == 0:
            pre_index_left = index
            pre_index_right = 0
            pre_index = 0
            continue
        dict_label[index] = separate_label_list[(pre_index_left + pre_index): (pre_index_right + index)]
        pre_index_left = pre_index_left + index
        pre_index_right = pre_index_right + index
    # same_id = []
    for k, v in dict_label.items():
        # verify_list = []
        # for id in same_id:
        #     verify_list.append(v[id])
        # num_1, num_0 = verify_list.count(1), verify_list.count(0)
        # if num_1 > num_0:
        #     for id in same_id:
        #         v[id] = 1
        # if num_1 < num_0:
        #     for id in same_id:
        #         v[id] = 0
        # if num_1 == num_0 != 0:
        #     for id in same_id:
        #         v[id] = v[same_id[0]]
        for n in range(k):
            if v[n] == 1:
                got = 0
                for x in range(len(v[:n])):
                    if v[x] == 0:
                        got = 1
                if got == 1:
                    continue
                each_para_label = max(paragraphs_label) + 1
            else:
                # if n not in same_id:
                #     same_id.append(n)
                # if k not in same_id:
                #     same_id.append(k)
                each_para_label = paragraphs_label[n]
                break
        paragraphs_label.append(each_para_label)
    return paragraphs_label


def write_output(filename, k, v):
    line = '\nmeasure{{\n  task_id: "{}"\n  score: "{}"\n}}\n'.format(k, str(v))
    print(line)
    open(filename, "a").write(line)


def predict(data):
    y_pred_list = []
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_pred_list.append(list(y_pred))
    return list(chain.from_iterable(y_pred_list))


def save_result(changes_list, author_sep_list, para_len_plus, outputpath):
    for share_id, l, p in para_len_plus:
        changes_label = []
        author_label = []
        multi_author = 0
        for i in range(l):
            changes_label.append(changes_list.pop(0))
        for i in range(p):
            author_label.append(author_sep_list.pop(0))
        if 1 in changes_label:
            multi_author = 1
        paragraph_authors = compound_label(author_label, l+1)
        solution = {
            'multi-author': multi_author,
            'changes': changes_label,
            'paragraph-authors': paragraph_authors
        }
        file_name = outputpath + '/solution-problem-' + share_id + '.json'
        with open(file_name, 'w') as file_handle:
            json.dump(solution, file_handle, default=myconverter)


def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()


if __name__ == '__main__':
    # model = load_model('finetune_all_256/best_model.h5')
    # model.load_weights('finetune_all_256/best_model.weights')
    model.load_weights('/home/zhangzhijie21/venu/best_model.weights')
    output_dir = '/home/zhangzhijie21/result'

    predict_result = predict(valid_generator)
    predict_plus_result = predict(valid_plus_generator)

    save_result(predict_result, predict_plus_result, para_len_plus, output_dir)

    solutions = read_solution_files(output_dir)
    truth = read_ground_truth_files(validation_path)

    task1_results = compute_score_single_predictions(truth, solutions, 'multi-author')
    task2_results = compute_score_multiple_predictions(truth, solutions, 'changes', labels=[0, 1])
    task3_results = compute_score_multiple_predictions(truth, solutions, 'paragraph-authors', labels=[1, 2, 3, 4, 5])

    for k, v in {
        "task1_score": task1_results,
        "task2_score": task2_results,
        "task3_score": task3_results
    }.items():
        write_output(os.path.join(output_dir, 'evaluation.txt'), k, v)
