import tensorflow as tf
import numpy as np

SQ_RATE = 10


def distance(features, centers):
    f_2 = tf.reduce_sum(tf.pow(features, 2), axis=1, keep_dims=True)
    c_2 = tf.reduce_sum(tf.pow(centers, 2), axis=1, keep_dims=True)
    dist = f_2 - 2 * tf.matmul(features, centers, transpose_b=True) + tf.transpose(c_2, perm=[1, 0])
    return dist


def softmax_loss(logits, labels):
    labels = tf.to_int32(labels)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                   logits=logits, name='xentropy')
    return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def predict(features, centers):
    dist = distance(features, centers)
    prediction = tf.argmin(dist, axis=1, name='prediction')
    return tf.cast(prediction, tf.int32)


class PMDSL:
    def __init__(self, nview, vdims, hdim_2, cdim_1, cdim_2, num_of_class, lamda=0.001, delta=0.5, weight_decay=0.0001,
                 learning_rate=0.01):

        self.nview = nview
        self.vdims = vdims

        self.input_A = []

        self.input_L = tf.placeholder(tf.int32, [None])

        # inputs
        for nv in range(nview):
            ph_A = tf.placeholder(tf.float32, [None, vdims[nv]], 'Anc_input_v' + str(nv + 1))
            self.input_A.append(ph_A)

        # weights and biases
        # p q r s

        self.Wp = []
        self.bp = []
        self.Wq = []
        self.bq = []
        for nv in range(nview):
            Wp_ = tf.Variable(tf.random_uniform([vdims[nv], int(np.log(vdims[nv]) * SQ_RATE)], -1.0, 1.0),
                              name='Wp_v' + str(nv + 1))
            bp_ = tf.Variable(tf.random_uniform([int(np.log(vdims[nv]) * SQ_RATE)], -1.0, 1.0),
                              name='bp_v' + str(nv + 1))
            Wq_ = tf.Variable(tf.random_uniform([int(np.log(vdims[nv]) * SQ_RATE), hdim_2], -1.0, 1.0),
                              name='Wq_v' + str(nv + 1))
            bq_ = tf.Variable(tf.random_uniform([hdim_2], -1.0, 1.0), name='bq_v' + str(nv + 1))
            self.Wp.append(Wp_)
            self.bp.append(bp_)
            self.Wq.append(Wq_)
            self.bq.append(bq_)
        self.Wa = tf.Variable(tf.random_uniform([hdim_2 * nview, hdim_2 * nview], -1.0, 1.0), name='Wa')
        # then concat and project ( metric based fusion)
        # self.Wr = [M1;M2;M3...;Mv]
        self.Wr = tf.Variable(tf.random_uniform([hdim_2 * nview, cdim_1], -1.0, 1.0), name='Wr')
        self.br = tf.Variable(tf.random_uniform([cdim_1], -1.0, 1.0), name='br')
        self.Ws = tf.Variable(tf.random_uniform([cdim_1, cdim_2], -1.0, 1.0), name='Ws')
        self.bs = tf.Variable(tf.random_uniform([cdim_2], -1.0, 1.0), name='bs')

        self.params = [self.Ws, self.Wr]
        for e in self.Wp:
            self.params.append(e)

        for e in self.Wq:
            self.params.append(e)

        # p q r s
        self.Ap_rep = []
        self.Ap_sig = []
        self.Aq_rep = []
        self.Aq_sig = []

        # p layer
        for nv in range(nview):
            Ap_rep_ = tf.nn.xw_plus_b(self.input_A[nv], self.Wp[nv], self.bp[nv])
            Ap_sig_ = tf.sigmoid(Ap_rep_)
            self.Ap_rep.append(Ap_rep_)
            self.Ap_sig.append(Ap_sig_)

        # q layer
        for nv in range(nview):
            Aq_rep_ = tf.nn.xw_plus_b(self.Ap_sig[nv], self.Wq[nv], self.bq[nv])
            Aq_sig_ = tf.sigmoid(Aq_rep_)
            self.Aq_rep.append(Aq_rep_)
            self.Aq_sig.append(Aq_sig_)

        self.vcenters = []  # the shared prototypes
        vc = tf.Variable(tf.zeros([num_of_class, hdim_2]), name='cetners_of_view' + str(0 + 1))
        for nv in range(nview):
            self.vcenters.append(vc)

        self.Avd2c = []  # view distance to centers (prototypes)

        for nv in range(nview):
            dist_f2c = distance(self.Aq_sig[nv], self.vcenters[nv])
            self.Avd2c.append(dist_f2c)

        # concat layer ( fast impl for fusion )
        self.A_concat = tf.concat(self.Aq_sig, axis=1)  #

        # r layer
        self.Ar_rep = tf.nn.xw_plus_b(self.A_concat, self.Wr, self.br)
        self.Ar_sig = tf.sigmoid(self.Ar_rep)

        # s layer
        self.As_rep = tf.nn.xw_plus_b(self.Ar_sig, self.Ws, self.bs)
        self.As_sig = tf.sigmoid(self.As_rep)

        self.AM = self.As_sig  # route

        # view loss ( L_{latent} )
        self.vpl_loss = []
        self.vdce_loss = []

        for nv in range(nview):
            dist_f2c = self.Avd2c[nv]
            logits = -dist_f2c / 1.0
            vdce_loss = softmax_loss(logits, self.input_L)
            self.vdce_loss.append(vdce_loss)
            vbatch_centers = tf.gather(self.vcenters[nv], self.input_L)
            vpl_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.Aq_sig[nv] - vbatch_centers), axis=1))
            self.vpl_loss.append(vpl_loss)

        # fusion prototypes
        self.centers = tf.Variable(tf.zeros([num_of_class, cdim_2]), name='cetners')

        self.dist_f2c = distance(self.AM, self.centers)  # batch_szie*NUM_OF_CLASS
        logits = -self.dist_f2c / 1.0
        self.dce_loss = softmax_loss(logits, self.input_L)
        batch_centers = tf.gather(self.centers, self.input_L)
        self.pl_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.AM - batch_centers), axis=1))

        # predict
        self.predict_label = predict(self.AM, self.centers)
        self.vpredict_label = [predict(self.Aq_sig[v], self.vcenters[v]) for v in range(nview)]

        # weight decay
        w = 0.0
        for e in self.params:
            w = w + tf.nn.l2_loss(e)
        w = w * weight_decay
        for v in range(nview):
            row_ids = []
            for i in range(64):
                row_ids.append(i + v * 64)
            rows = tf.gather(self.Wr, row_ids)
            for i in range(self.Wr.shape[1]):
                v = tf.gather(rows, i, axis=1)
                # w = w + 0.01*tf.nn.l2_loss(v)

        self.loss = w + self.dce_loss * (1 - delta) + lamda * self.pl_loss * (1 - delta)
        for v in range(nview):
            self.loss += self.vdce_loss[v] * delta
            self.loss += lamda * self.vpl_loss[v] * delta

        # optimizer
        global_step = tf.Variable(0, trainable=False)
        lr_step = tf.train.exponential_decay(learning_rate, global_step, 100, 0.9, staircase=True)  # 学习率递减
        self.opt = tf.train.AdamOptimizer(lr_step)
        self.update = self.opt.minimize(self.loss)

        #
        # extension route

        AM = self.AM
        self.AREP = AM
