#!/usr/bin/env python
# coding: utf-8



from PMDSL_Net import PMDSL
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, classification_report
from utils import *
from pylab import *

BATCH_SIZE = 64
TRAIN_EPOCH = 200
TEST_EPOCH = 2
TOP_K = 5
HDIM_2 = 64
CDIM_1 = 64
CDIM_2 = 64
SELECT_VIEW = [0, 1, 2, 3, 4, 5]
nview = len(SELECT_VIEW)
DELTA = 0.5
LAMDA = 0.001
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.0001

DATA_DIR = '../data/'

dataset = 'caltech101-20_P.npy'
data_dict = dict(np.load(DATA_DIR + dataset)[()])
train_dict = data_dict['train']
train_view_data_ = [train_dict['view_data'][s] for s in SELECT_VIEW]
train_label_ = train_dict['label']
test_dict = data_dict['test']
test_view_data = [test_dict['view_data'][s] for s in SELECT_VIEW]
test_label = test_dict['label']
view_name = [data_dict['view_name'][s] for s in SELECT_VIEW]
NUM_OF_CLASS = np.max(train_dict['label'] + 1)
view_dim = [data_dict['view_dim'][s] for s in SELECT_VIEW]
print(dataset)
print(view_name)
print(view_dim)

view_data = []
for v in range(nview):
    vd = []
    vd.extend(train_view_data_[v])
    vd.extend(test_view_data[v])
    view_data.append(np.asarray(vd))

label = []
label.extend(train_label_)
label.extend(test_label)

# sample rate from training set
train_view_data, train_label, _, _ = split_dataset_ex_with_label_balance_static(train_view_data_, train_label_,
                                                                                train_ratio=1.0, test_ratio=0.0)

# drawing
acc_list = []  # knn
pacc_list = []  # predict
epo_list = []
los_list = []


def main():
    sess = get_tf_session()
    net = PMDSL(nview, view_dim, hdim_2=HDIM_2, cdim_1=CDIM_1, cdim_2=CDIM_2,
                num_of_class=NUM_OF_CLASS, lamda=LAMDA, delta=DELTA, weight_decay=WEIGHT_DECAY,
                learning_rate=LEARNING_RATE)
    sess.run(tf.initialize_all_variables())

    max_acc = 0.00
    max_pacc = 0.0
    print('Start training')
    for e in range(TRAIN_EPOCH):

        A_ = train_view_data
        AL_ = np.asarray(train_label)

        train_size = len(A_[0])  # d

        m_feed_dict = dict()
        for v in range(nview):
            m_feed_dict[net.input_A[v]] = A_[v]

        m_feed_dict[net.input_L] = train_label
        loss = sess.run(net.loss, feed_dict=m_feed_dict)
        print("train epoch:%d loss:%.4f " % (e, loss))
        cur_index = 0
        while cur_index < train_size:
            input_A = [[] for v in range(nview)]
            if cur_index + BATCH_SIZE < train_size:
                for v in range(nview):
                    input_A[v] = A_[v][cur_index:cur_index + BATCH_SIZE]
                input_L = AL_[cur_index:cur_index + BATCH_SIZE]
            else:
                for v in range(nview):
                    input_A[v] = A_[v][cur_index:]
                input_L = AL_[cur_index:]
            cur_index += BATCH_SIZE

            m_feed_dict = {}

            for v in range(nview):
                m_feed_dict[net.input_A[v]] = input_A[v]

            m_feed_dict[net.input_L] = input_L

            sess.run(net.update, feed_dict=m_feed_dict)

        if e % TEST_EPOCH == 0:
            print('test')

            m_feed_dict = {}
            for v in range(nview):
                m_feed_dict[net.input_A[v]] = train_view_data[v]

            train_rep, train_predict_label = sess.run([net.AREP, net.predict_label], feed_dict=m_feed_dict)

            m_feed_dict = {}
            for v in range(nview):
                m_feed_dict[net.input_A[v]] = test_view_data[v]
            test_rep, predict_label, vpredict_label = sess.run([net.AREP, net.predict_label, net.vpredict_label],
                                                               feed_dict=m_feed_dict)

            # 计算两两之间的距离
            test_rep_dist = cdist(test_rep, train_rep)

            test_predict_labels = []
            for idx, dst in enumerate(test_rep_dist):
                sorted_idx = np.argsort(dst)[:TOP_K]
                # 统计TOP5最多的类别
                knn_labels = train_label[sorted_idx]
                label_counts = [0 for en in range(NUM_OF_CLASS)]
                for kl in knn_labels:
                    label_counts[kl] += 1
                mx_label = np.argmax(label_counts)
                test_predict_labels.append(mx_label)

            acc = accuracy_score(test_label, test_predict_labels)
            print('ACC:%.4f' % acc)
            predict_acc = accuracy_score(test_label, predict_label)
            # cls_rpt = classification_report(test_label, predict_label, output_dict=True)
            # print(cls_rpt['weighted avg'])
            print('PACC:%.4f' % (predict_acc))

            acc_list.append(acc)
            pacc_list.append(predict_acc)
            epo_list.append(e)
            los_list.append(loss)

            if acc > max_acc or predict_acc > max_pacc:
                if acc > max_acc:
                    max_acc = acc
                if predict_acc > max_pacc:
                    max_pacc = predict_acc

                f = open('record.txt', 'a+')
                f.write("EPO:%d ACC:%.4f PACC:%.4f MAX_ACC:%.4f MAX_PACC:%.4f\n" % (e, acc, predict_acc,max_acc,max_pacc))
                f.close()

            if e > TRAIN_EPOCH - (2 * TEST_EPOCH):
                m_feed_dict = {}
                for v in range(nview):
                    m_feed_dict[net.input_A[v]] = view_data[v]
                    view_protos = sess.run(net.vcenters)
                fusion_protos = sess.run(net.centers)
                test_view_rep, test_fusion_rep, test_fusion_r_rep = sess.run(
                    [net.Aq_sig, net.As_sig, net.Ar_rep], feed_dict=m_feed_dict)
                view_rep = []
                for v in range(nview):
                    vr = list(test_view_rep[v])
                    vr.extend(view_protos[v])
                    view_rep.append(vr)
                fusion_rep = list(test_fusion_rep)
                fusion_rep.extend(fusion_protos)
                fusion_rep = np.asarray(fusion_rep)
                draw_dataset(dataset + '_view_specifc', view_rep, view_names=view_name, labels=label,
                             last_extend=NUM_OF_CLASS)
                draw_dataset_ex(dataset + '_fusion', [fusion_rep], view_names=['fusion'], labels=label,
                                last_extend=NUM_OF_CLASS)


    sess.close()

    # draw

    mpl.rcParams['font.sans-serif'] = ['SimHei']
    x_axis_data = epo_list
    y1_data = acc_list
    y2_data = pacc_list
    mv = np.max(los_list)
    y3_data = [l / mv for l in los_list]
    plt.plot(x_axis_data, y1_data, marker='o', ms=0.1, label='acc of KNN')
    plt.plot(x_axis_data, y2_data, marker='*', ms=0.1, label='acc of predict')
    plt.plot(x_axis_data, y3_data, marker='>', ms=0.1, label='train loss')
    plt.title(dataset)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
