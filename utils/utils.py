#!/usr/bin/env python
# coding: utf-8


import numpy as np
import random
import matplotlib.pyplot as plt


from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn import metrics
import tensorflow as tf

G_TRAIN_RATIO = 0.8
G_TEST_RATIO = 0.2
NOISE_DX = 0.1

colors = ['#FF1493', '#FF00FF', '#7B68EE', '#0000FF', '#800080', '#4B0082', '#FFB6C1', '#808000', '#DC143C', '#4169E1',
          '#00BFFF', '#5F9EA0', '#00FFFF', '#00CED1', '#2F4F4F', '#00FF7F', '#2E8B57', '#FFFF00', '#FFD700', '#FFA500',
          '#FF4500', '#000000']
gmarker = ['+','o','^','s','*','p']

# np.random.shuffle(colors)

def syc_shuffle(arr_list):
    state = np.random.get_state()
    for e in arr_list:
        np.random.shuffle(e)
        np.random.set_state(state)


def split_dataset_ex_static(view_data, label,train_ratio = -1,test_ratio = -1):
    if train_ratio >= 0:
        TRAIN_RATIO = train_ratio
    else:
        TRAIN_RATIO = G_TRAIN_RATIO

    if test_ratio >= 0:
        TEST_RATIO = test_ratio
    else:
        TEST_RATIO = G_TEST_RATIO
    length = len(view_data[0])
    # 分成10分
    train_idx = []
    test_idx = []
    block_len = int(length / 10)
    block_idx = range(0, length, block_len)

    for bi in block_idx:
        if bi + block_len < length:
            candi_idx = range(bi, bi + block_len)
        else:
            candi_idx = range(bi, length)
        train_idx.extend(candi_idx[0:int(block_len * TRAIN_RATIO)])
        test_idx.extend(candi_idx[int(block_len * (TRAIN_RATIO+TEST_RATIO)):])

    # print('test_idx_ex2')
    # print(test_idx)

    train_view_data = [[] for e in range(len(view_data))]
    train_label = label[train_idx]

    test_view_data = [[] for e in range(len(view_data))]
    test_label = label[test_idx]

    for e in range(len(view_data)):
        train_view_data[e] = view_data[e][train_idx]
        test_view_data[e] = view_data[e][test_idx]

    return train_view_data, train_label, test_view_data, test_label

def split_dataset_ex_with_label_balance_static(view_data,label,train_ratio = -1,test_ratio = -1):
    if train_ratio >= 0 :
        TRAIN_RATIO = train_ratio
    else:
        TRAIN_RATIO = G_TRAIN_RATIO

    if test_ratio >= 0 :
        TEST_RATIO = test_ratio
    else:
        TEST_RATIO = G_TEST_RATIO
    label_dict = dict()
    #max_cls = np.max(label)
    for i,l in enumerate(label):
        if label_dict.__contains__(l):
            label_dict[l].append(i)
        else:
            label_dict[l] = [i]
    #每个类别按比例划分
    cls_keys = list(label_dict.keys())
    nview = len(view_data)
    train_view_data = [[] for v in range(nview)]
    test_view_data =  [[] for v in range(nview)]
    train_label = []
    test_label = []

    label_count_dict = dict()
    for ck in cls_keys:
        num_of_cls = len(label_dict[ck])
        label_count_dict[ck] = 'count:'+str(num_of_cls)
        idx = [i for i in range(num_of_cls)]#random.sample(range(num_of_cls),k=num_of_cls)
        train_idx = np.asarray(label_dict[ck])[idx[0:int(num_of_cls*TRAIN_RATIO)]]
        test_idx = np.asarray(label_dict[ck])[idx[int(num_of_cls*TRAIN_RATIO):int(num_of_cls*(TRAIN_RATIO+TEST_RATIO))]]

        for v in range(nview):
            train_view_data[v].extend(view_data[v][train_idx])
            test_view_data[v].extend(view_data[v][test_idx])

        train_label.extend(label[train_idx])
        test_label.extend(label[test_idx])


    print(label_count_dict)

    #shuffle
    train_shuffle_list = []
    test_shuffle_list = []
    for v in range(nview):
        train_shuffle_list.append(train_view_data[v])
        test_shuffle_list.append(test_view_data[v])

    train_shuffle_list.append(train_label)
    test_shuffle_list.append(test_label)

    syc_shuffle(train_shuffle_list)
    syc_shuffle(test_shuffle_list)

    for v in range(nview):
        train_view_data[v] = np.asarray(train_view_data[v])
        test_view_data[v] = np.asarray(test_view_data[v])


    return train_view_data,np.asarray(train_label),test_view_data,np.asarray(test_label)

def split_dataset_n_shot_static(view_data,label,n_shot,):

    label_dict = dict()
    #max_cls = np.max(label)
    for i,l in enumerate(label):
        if label_dict.__contains__(l):
            label_dict[l].append(i)
        else:
            label_dict[l] = [i]
    #每个类别按比例划分
    cls_keys = list(label_dict.keys())
    nview = len(view_data)
    train_view_data = [[] for v in range(nview)]
    test_view_data =  [[] for v in range(nview)]
    train_label = []
    test_label = []

    label_count_dict = dict()
    for ck in cls_keys:
        num_of_cls = len(label_dict[ck])
        label_count_dict[ck] = 'count:'+str(num_of_cls)
        idx = [i for i in range(num_of_cls)]#random.sample(range(num_of_cls),k=num_of_cls)
        train_idx = np.asarray(label_dict[ck])[idx[0:n_shot]]
        test_idx = np.asarray(label_dict[ck])[idx[0:n_shot]]

        for v in range(nview):
            train_view_data[v].extend(view_data[v][train_idx])
            test_view_data[v].extend(view_data[v][test_idx])

        train_label.extend(label[train_idx])
        test_label.extend(label[test_idx])


    print(label_count_dict)



    for v in range(nview):
        train_view_data[v] = np.asarray(train_view_data[v])
        test_view_data[v] = np.asarray(test_view_data[v])


    return train_view_data,np.asarray(train_label),test_view_data,np.asarray(test_label)

def split_dataset_ex_with_label_balance(view_data,label,train_ratio = -1,test_ratio = -1):
    if train_ratio >= 0:
        TRAIN_RATIO = train_ratio
    else:
        TRAIN_RATIO = G_TRAIN_RATIO

    if test_ratio >= 0:
        TEST_RATIO = test_ratio
    else:
        TEST_RATIO = G_TEST_RATIO
    label_dict = dict()
    #max_cls = np.max(label)
    for i,l in enumerate(label):
        if label_dict.__contains__(l):
            label_dict[l].append(i)
        else:
            label_dict[l] = [i]
    #每个类别按比例划分
    cls_keys = list(label_dict.keys())
    nview = len(view_data)
    train_view_data = [[] for v in range(nview)]
    test_view_data =  [[] for v in range(nview)]
    train_label = []
    test_label = []

    label_count_dict = dict()
    for ck in cls_keys:
        num_of_cls = len(label_dict[ck])
        label_count_dict[ck] = 'count:'+str(num_of_cls)
        idx = random.sample(range(num_of_cls),k=num_of_cls)
        train_idx = np.asarray(label_dict[ck])[idx[0:int(num_of_cls*TRAIN_RATIO)]]
        test_idx = np.asarray(label_dict[ck])[idx[int(num_of_cls*TRAIN_RATIO):int(num_of_cls*(TRAIN_RATIO+TEST_RATIO))]]

        for v in range(nview):
            train_view_data[v].extend(view_data[v][train_idx])
            test_view_data[v].extend(view_data[v][test_idx])

        train_label.extend(label[train_idx])
        test_label.extend(label[test_idx])


    print(label_count_dict)

    #shuffle
    train_shuffle_list = []
    test_shuffle_list = []
    for v in range(nview):
        train_shuffle_list.append(train_view_data[v])
        test_shuffle_list.append(test_view_data[v])

    train_shuffle_list.append(train_label)
    test_shuffle_list.append(test_label)

    syc_shuffle(train_shuffle_list)
    syc_shuffle(test_shuffle_list)

    for v in range(nview):
        train_view_data[v] = np.asarray(train_view_data[v])
        test_view_data[v] = np.asarray(test_view_data[v])


    return train_view_data,np.asarray(train_label),test_view_data,np.asarray(test_label)





def split_dataset_ex(view_data, label,train_ratio = -1,test_ratio = -1):
    #return split_dataset_ex_static(view_data,label)
    if train_ratio > 0 :
        TRAIN_RATIO = train_ratio
    if test_ratio > 0 :
        TEST_RATIO = test_ratio
    return split_dataset_ex_with_label_balance(view_data,label,train_ratio,test_ratio)
    length = len(view_data[0])
    random_idx = random.sample(range(length), k=length)

    train_idx = random_idx[:int(length * TRAIN_RATIO)]
    test_idx = random_idx[int(length * TRAIN_RATIO):int(length * (TRAIN_RATIO+TEST_RATIO))]

    train_view_data = [[] for e in range(len(view_data))]
    train_label = label[train_idx]

    test_view_data = [[] for e in range(len(view_data))]
    test_label = label[test_idx]

    for e in range(len(view_data)):
        train_view_data[e] = view_data[e][train_idx]
        test_view_data[e] = view_data[e][test_idx]

    return train_view_data, train_label, test_view_data, test_label


def draw_with_tsne(rep, label, metric='euclidean',title="",last_extend=0):
    tsne = manifold.TSNE(n_components=2)#, init='pca', random_state=501)
    new_tsne_rep = tsne.fit_transform(rep)
    plt.rcParams['figure.dpi'] = 220
    plt.rcParams['figure.figsize'] = [6.0, 6.0]
    #fig = plt.figure()
    fig, ax = plt.subplots()

    # ax = Axes3D(fig)
    if last_extend >0:
        for rep_i, rep in enumerate(new_tsne_rep[:-last_extend]):
            x, y = rep[0], rep[1]
            ax.scatter(x, y, s=0.5, c=colors[label[rep_i]])
        for rep_i, rep in enumerate(new_tsne_rep[-last_extend:]):
            x, y = rep[0], rep[1]
            ax.scatter(x, y, s=20, c='#050505')
    else:
        for rep_i, rep in enumerate(new_tsne_rep[:]):
            x, y = rep[0], rep[1]
            ax.scatter(x, y, s=0.5, c=colors[label[rep_i]])
    plt.show()



def draw_dataset(dataset_name,view_reps,view_names,labels,last_extend=0):
    nview = len(view_reps)
    plt.rcParams['figure.dpi'] = 220
    plt.rcParams['figure.figsize'] = [6.0,6.0]
    fig = plt.figure()

    nrow = int(nview / 2+0.5)
    ncol = 2
    if nview == 1:
        ncol = 1
    plt.suptitle(dataset_name)
    for v in range(nview):
        vrep = view_reps[v]
        tsne = manifold.TSNE(n_components=2,random_state=10)
        tsne_rep = tsne.fit_transform(vrep)
        if last_extend > 0:
            x = tsne_rep[:-last_extend,0]
            y = tsne_rep[:-last_extend,1]
        else:
            x = tsne_rep[:, 0]
            y = tsne_rep[:, 1]
        ax = fig.add_subplot(nrow*100+(ncol)*10+v+1)
        for i,sx in enumerate(x):
            ax.scatter(sx,y[i],s=0.5,c=colors[labels[i]])
        #ax.scatter(x,y,s=POINT_SIZE,c=clist)
        if last_extend > 0:
            for tr in tsne_rep[-last_extend:]:
                ax.scatter(tr[0],tr[1],s=30.0,c='#050505',marker='*')
        ax.set_title(view_names[v])
    #
    plt.show()

def draw_dataset_ex(dataset_name,view_reps,view_names,labels,last_extend=0):
    nview = len(view_reps)
    plt.rcParams['figure.dpi'] = 220
    plt.rcParams['figure.figsize'] = [6.0,6.0]
    fig = plt.figure()

    nrow = int(nview / 2+0.5)
    ncol = 2
    if nview == 1:
        ncol = 1
    plt.suptitle(dataset_name)

    # 将每个视图的表示放在一起表示
    stacked_rep = []
    for v in range(nview):
        stacked_rep.extend(view_reps[v]) #
    tsne = manifold.TSNE(n_components=2)
    stacked_tsne_rep = tsne.fit_transform(stacked_rep)

    sample_num = len(view_reps[0])
    for v in range(nview):
        tsne_rep = stacked_tsne_rep[(v*sample_num):((v+1)*sample_num)]
        if last_extend > 0:
            x = tsne_rep[:-last_extend,0]
            y = tsne_rep[:-last_extend,1]
        else:
            x = tsne_rep[:, 0]
            y = tsne_rep[:, 1]
        ax = fig.add_subplot(nrow*100+(ncol)*10+v+1)
        for i,sx in enumerate(x):
            ax.scatter(sx,y[i],s=0.5,c=colors[labels[i]],marker=gmarker[v])
        #ax.scatter(x,y,s=POINT_SIZE,c=clist)
        if last_extend > 0:
            for tr in tsne_rep[-last_extend:]:
                ax.scatter(tr[0],tr[1],s=30.0,c='#050505',marker='*')
        ax.set_title(view_names[v])
    #
    plt.show()
    # 画融合的
    fig, ax = plt.subplots()
    for v in range(nview):
        tsne_rep = stacked_tsne_rep[(v * sample_num):((v + 1) * sample_num)]
        if last_extend > 0:
            x = tsne_rep[:-last_extend, 0]
            y = tsne_rep[:-last_extend, 1]
        else:
            x = tsne_rep[:, 0]
            y = tsne_rep[:, 1]

        for i, sx in enumerate(x):
            ax.scatter(sx, y[i], s=0.5, c=colors[labels[i]], marker=gmarker[v])
        # ax.scatter(x,y,s=POINT_SIZE,c=clist)
        if last_extend > 0:
            for tr in tsne_rep[-last_extend:]:
                ax.scatter(tr[0], tr[1], s=40.0, c='#050505',marker='*')
        #ax.set_title(view_names[v])
    plt.show()


def get_tf_session():
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # 设置随机种子
    #tf.random.set_random_seed(1234)
    return sess


def normalize_view_data(view_data, select_view):
    return [(view_data[s] - np.min(view_data[s])) / (np.max(view_data[s]) - np.min(view_data[s])) for s in select_view]
