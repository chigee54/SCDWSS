'''
此工程是为‘2021 style change detection’任务做准备
目的是复现‘2020 style change detection’参赛队伍排名第一的论文方法
利用bert做特征提取器，随机森林做分类器，对task1，task2，task3进行测试
此工程是用pytorch加载了bert模型权重，特征提取生成句子embedding
用的是sklearn的随机森林分类器，只能用CPU跑

注意事项
1. 此代码方法参考(https://github.com/aarish407/Style-Change-Detection-Using-BERT)
2. 此代码评测参考(https://github.com/pan-webis-de/pan-code/tree/master/clef21/style-change-detection)
3. 此代码专门用来跑task2，所以注释掉了task1，task3
'''

import torch
import json
import time
import joblib
import pickle
import os
import glob
import numpy as np
from numpy import *
from itertools import chain
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from SplitIntoSentences import split_into_sentences
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel, BertForMaskedLM

# 加载bert模型
tokenizer = BertTokenizer.from_pretrained('./bert-base-cased')
model = BertModel.from_pretrained('./bert-base-cased')
if torch.cuda.is_available():
    model = model.cuda()
model.eval()


# 利用bert做特征提取
def generate_sentence_embedding(sentence):
    marked_sentence = "[CLS] " + sentence + " [SEP]"
    tokenized_sentence = tokenizer.tokenize(marked_sentence)
    if len(tokenized_sentence) > 512:  # truncate the sentence if it is longer than 512
        tokenized_sentence = tokenized_sentence[:512]

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_sentence)
    segment_ids = [1] * len(tokenized_sentence)

    token_tensor = torch.tensor([indexed_tokens])
    segment_tensor = torch.tensor([segment_ids])

    if torch.cuda.is_available():
        token_tensor = token_tensor.cuda()
        segment_tensor = segment_tensor.cuda()

    with torch.no_grad():
        encoded_layers, _ = model(token_tensor, segment_tensor)

    token_embeddings = torch.stack(encoded_layers, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = torch.sum(token_embeddings[-4:, :, :], dim=0)
    sentence_embedding_sum = torch.sum(token_embeddings, dim=0)

    del marked_sentence
    del tokenized_sentence
    del indexed_tokens, segment_ids
    del token_tensor
    del segment_tensor
    del encoded_layers
    del token_embeddings

    return sentence_embedding_sum


# Fit the classifier and perform prediction, save result to the folder
def generate_embeddings_train_predict(corpora, corpora_predict, outputpath):
    # create a classifier to task1(document level)
    # clf_one = RandomForestClassifier(n_estimators=1800, criterion='gini', min_samples_leaf=1, min_samples_split=2)

    # create a classifier to task2(two paragraph level)
    # clf_two = RandomForestClassifier(n_estimators=250, criterion='gini', min_samples_leaf=1, min_samples_split=2)

    # create a classifier to task3(average paragraph level)
    clf_three = RandomForestClassifier(n_estimators=250, criterion='gini', min_samples_leaf=1, min_samples_split=2)

    document_embeddings_list = []
    document_label_list = []
    separate_label_list = []
    two_paragraphs_label_list = []
    two_paragraphs_embeddings_list = []
    paragraphs_embeddings_list = []
    document_id = 0
    for document_path in corpora:
        # 读取每一个文本并赋予对应id
        with open(document_path, encoding="utf8") as file:
            document = file.read()
        share_id = os.path.basename(document_path)[8:-4]
        print(share_id)
        if not document or not share_id:
            continue

        document_embeddings = torch.zeros(768)
        if torch.cuda.is_available():
            document_embeddings = document_embeddings.cuda()

        sentence_count = 0
        paragraphs_embeddings = []
        two_paragraphs_embeddings = []
        pure_para_embedding_list = []
        pure_para_length_list = []
        paragraphs = document.split('\n')

        previous_para_embeddings = None
        previous_para_length = None

        for paragraph_index, paragraph in enumerate(paragraphs):
            sentences = split_into_sentences(paragraph)

            current_para_embeddings = torch.zeros(768)
            embed_first = torch.ones(768)
            if torch.cuda.is_available():
                current_para_embeddings = current_para_embeddings.cuda()
                embed_first = embed_first.cuda()

            current_para_length = len(sentences)

            for sentence in sentences:
                sentence_count += 1
                sentence_embedding = generate_sentence_embedding(sentence)
                current_para_embeddings.add_(sentence_embedding)
                document_embeddings.add_(sentence_embedding)
                del sentence_embedding, sentence

            if previous_para_embeddings is not None:
                two_para_embeddings, _ = compute_avg_two_para(previous_para_length, current_para_length,
                                                           previous_para_embeddings, current_para_embeddings)
                two_paragraphs_embeddings.append(two_para_embeddings)

            previous_para_embeddings = current_para_embeddings
            previous_para_length = current_para_length
            pure_para_length_list.append(current_para_length)
            pure_para_embedding_list.append(current_para_embeddings)

            # 按顺序叠加embedding
            if paragraph_index == 0:
                paragraphs_embeddings.append(embed_first)
            for i in range(paragraph_index):
                embed, length = compute_avg_two_para(pure_para_length_list[i],
                                                     pure_para_length_list[paragraph_index],
                                                     pure_para_embedding_list[i],
                                                     pure_para_embedding_list[paragraph_index])
                paragraphs_embeddings.append(embed)

            del sentences
            del paragraph

        del previous_para_embeddings, previous_para_length
        del current_para_embeddings, current_para_length

        para_embed_length = len(two_paragraphs_embeddings)

        # two_paragraphs_embeddings = torch.stack(two_paragraphs_embeddings, dim=0)
        paragraphs_embeddings = torch.stack(paragraphs_embeddings, dim=0)
        # document_embeddings = document_embeddings / sentence_count

        if torch.cuda.is_available():
            # document_embeddings = document_embeddings.cpu()
            # two_paragraphs_embeddings = two_paragraphs_embeddings.cpu()
            paragraphs_embeddings = paragraphs_embeddings.cpu()

        bool_id = '{}{}\t'.format(share_id, str(np.isnan(paragraphs_embeddings.numpy()).any()))
        open(os.path.join(output_dir, "evaluation.txt"), "a").write(bool_id)

        #  添加训练集标签
        truth_label = read_ground_truth_files(input_train)
        # document_label = truth_label[share_id]['multi-author']
        two_paragraphs_label = truth_label[share_id]['changes']
        paragraphs_label = truth_label[share_id]['paragraph-authors']

        # 当遇到少标签时，自动补1
        if para_embed_length > len(two_paragraphs_label):
            two_paragraphs_label.append(1)
            paragraphs_label.append(1)

        separate_label = separate_para_label(paragraphs_label)
        # document_label_list.append(document_label)
        # two_array_para_label = array(two_paragraphs_label)
        array_separate_label = array(separate_label)
        if document_id == 0:
            document_id += 1
            # two_paragraphs_label_list = two_array_para_label
            separate_label_list = array_separate_label
            # two_paragraphs_embeddings_list = two_paragraphs_embeddings
            paragraphs_embeddings_list = paragraphs_embeddings
        else:
            # two_paragraphs_label_list = np.append(two_paragraphs_label_list, two_array_para_label)
            separate_label_list = np.append(separate_label_list, array_separate_label)
            # two_paragraphs_embeddings_list = torch.cat([two_paragraphs_embeddings_list, two_paragraphs_embeddings], dim=0)
            paragraphs_embeddings_list = torch.cat([paragraphs_embeddings_list, paragraphs_embeddings], dim=0)

        # document_embeddings = document_embeddings.numpy()
        # document_embeddings_list.append(document_embeddings)

    # 检验是否有NaN值，有的话就补0
    # print(np.isnan(two_paragraphs_embeddings_list.numpy()).any())
    print(np.isnan(paragraphs_embeddings_list.numpy()).any())
    # two_paragraphs_embeddings_list.numpy()[np.isnan(two_paragraphs_embeddings_list.numpy())] = 0
    paragraphs_embeddings_list.numpy()[np.isnan(paragraphs_embeddings_list.numpy())] = 0

    # clf_one = clf_one.fit(document_embeddings_list, document_label_list)
    # clf_two = clf_two.fit(two_paragraphs_embeddings_list, two_paragraphs_label_list)
    clf_three = clf_three.fit(paragraphs_embeddings_list, separate_label_list)

    # 用fit好的分类器进行预测结果，保存
    def generate_embeddings_predict(corpora, outputpath):
        for document_path in corpora:
            # 读取每一个文本并赋予id
            with open(document_path, encoding="utf-8") as file:
                document = file.read()
            share_id = os.path.basename(document_path)[8:-4]
            if not document or not share_id:
                continue
            document_embeddings = torch.zeros(768)
            if torch.cuda.is_available():
                document_embeddings = document_embeddings.cuda()

            sentence_count = 0
            two_paragraphs_embeddings = []
            paragraphs_embeddings = []
            paragraphs = document.split('\n')

            previous_para_embeddings = None
            previous_para_length = None

            for paragraph_index, paragraph in enumerate(paragraphs):
                sentences = split_into_sentences(paragraph)
                current_para_embeddings = torch.zeros(768)
                if torch.cuda.is_available():
                    current_para_embeddings = current_para_embeddings.cuda()

                current_para_length = len(sentences)
                for sentence in sentences:
                    sentence_count += 1
                    sentence_embedding = generate_sentence_embedding(sentence)
                    current_para_embeddings.add_(sentence_embedding)
                    document_embeddings.add_(sentence_embedding)
                    del sentence_embedding, sentence

                if previous_para_embeddings is not None:
                    two_para_embeddings, _ = compute_avg_two_para(previous_para_length, current_para_length,
                                                                  previous_para_embeddings, current_para_embeddings)
                    two_paragraphs_embeddings.append(two_para_embeddings)

                previous_para_embeddings = current_para_embeddings
                previous_para_length = current_para_length
                pure_para_length_list.append(current_para_length)
                pure_para_embedding_list.append(current_para_embeddings)

                # 按顺序叠加embedding
                if paragraph_index == 0:
                    paragraphs_embeddings.append(embed_first)
                for i in range(paragraph_index):
                    embed, length = compute_avg_two_para(pure_para_length_list[i],
                                                         pure_para_length_list[paragraph_index],
                                                         pure_para_embedding_list[i],
                                                         pure_para_embedding_list[paragraph_index])
                    paragraphs_embeddings.append(embed)

                del sentences
                del paragraph
            del previous_para_embeddings, previous_para_length
            del current_para_embeddings, current_para_length

            # two_paragraphs_embeddings = torch.stack(two_paragraphs_embeddings, dim=0)
            paragraphs_embeddings = torch.stack(paragraphs_embeddings, dim=0)
            # document_embeddings = document_embeddings / sentence_count
            # document_embeddings = document_embeddings.unsqueeze(0)

            if torch.cuda.is_available():
                # document_embeddings = document_embeddings.cpu()
                # two_paragraphs_embeddings = two_paragraphs_embeddings.cpu()
                paragraphs_embeddings = paragraphs_embeddings.cpu()

            # PREDICTIONS
            # try:
            #     document_label = clf_one.predict(document_embeddings)
            # except:
            #     document_label = [0]

            # try:
            #     two_paragraphs_labels = clf_two.predict(two_paragraphs_embeddings)
            # except:
            #     two_paragraphs_labels = np.zeros(len(paragraphs) - 1)
            # two_paragraphs_labels = two_paragraphs_labels.astype(np.int32)

            try:
                separate_label_list = clf_three.predict(paragraphs_embeddings)
            except:
                separate_label_list = np.ones(len(paragraphs))
            separate_label_list = separate_label_list.astype(np.int32)
            separate_label_list = separate_label_list.tolist()
            paragraphs_labels = compound_label(separate_label_list, len(paragraphs))

            solution = {
                # 'multi-author': document_label[0],
                # 'changes': two_paragraphs_labels.tolist(),
                'paragraph-authors': paragraphs_labels
            }

            file_name = outputpath + '/solution-problem-' + share_id + '.json'
            with open(file_name, 'w') as file_handle:
                json.dump(solution, file_handle, default=myconverter)

            # del document_embeddings, document_label
            del solution
            del document, share_id
            del paragraphs
    generate_embeddings_predict(corpora_predict, outputpath)


# compute the average of two paragraphs embeddings
def compute_avg_two_para(para_length_a, para_length_b, para_embedding_a, para_embedding_b):
    cp_para_lengths = para_length_a + para_length_b
    cp_para_embeddings = (para_embedding_a + para_embedding_b) / cp_para_lengths
    return cp_para_embeddings, cp_para_lengths


# 将任务3标签拆分为1,0
def separate_para_label(paragraphs_label):
    separate_label = []
    for i in range(len(paragraphs_label)):
        if i == 0:
            separate_label.append(paragraphs_label[0])
            continue
        for a in range(i):
            if paragraphs_label[a] != paragraphs_label[i]:
                separate_label.append(1)
            else:
                separate_label.append(0)
    return separate_label


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
            pre_index_right = 1
            pre_index = 1
            continue
        else:
            dict_label[index] = separate_label_list[(pre_index_left + pre_index):(pre_index_right + index)]
            pre_index_left = pre_index_left + index
            pre_index_right = pre_index_right + index
    for k, v in dict_label.items():
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
                each_para_label = paragraphs_label[n]
        paragraphs_label.append(each_para_label)
    return paragraphs_label


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
    return f1_score(list(chain.from_iterable(task2_truth)), list(chain.from_iterable(task2_solution)), average='macro', labels=labels)


def write_output(filename, k, v):
    line = '\nmeasure{{\n  task_id: "{}"\n  score: "{}"\n}}\n'.format(k, str(v))
    print(line)
    open(filename, "a").write(line)


def myconverter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()


if __name__ == '__main__':
    input_train = 'demo_data20/train'
    input_val = 'demo_data20/validation1'
    output_dir = 'task3_result'
    dataset_train = glob.glob(input_train + '/*.txt')
    dataset_val = glob.glob(input_val + '/*.txt')
    generate_embeddings_train_predict(dataset_train, dataset_val, output_dir)
    solutions = read_solution_files(output_dir)
    truth = read_ground_truth_files(input_val)
    # task1_results = compute_score_single_predictions(truth, solutions, 'multi-author')
    # task2_results = compute_score_multiple_predictions(truth, solutions, 'changes', labels=[0, 1])
    task3_results = compute_score_multiple_predictions(truth, solutions, 'paragraph-authors', labels=[1, 2, 3, 4, 5])

    # for k, v in {
    #     "task1_score": task1_results,
    #     "task2_score": task2_results,
    #     "task3_score": task3_results}.items():
    #     write_output(os.path.join(output_dir, 'evaluation.txt'), k, v)

    write_output(os.path.join(output_dir, "evaluation.txt"), 'task3_score', task3_results)