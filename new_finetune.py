def data_load(filename):
    l = []
    a = []
    with open(filename, 'r', encoding='utf-8') as f:
        for i in f.readlines():
            label, sentence = i.strip().split('\t')
            l.append((sentence, int(label)))
            a.append(int(label))
    return len(set(a)), l
