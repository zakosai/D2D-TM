import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, flatten, batch_norm, maxout
from tensorflow import sigmoid
import tensorflow.keras.backend as K
from tensorflow.contrib.framework import argsort
import numpy as np
import os
import argparse


class Translation:
    def __init__(self, batch_size, dim_A, dim_B, encode_dim_A, decode_dim_A, encode_dim_B, decode_dim_B, adv_dim_A,
                 adv_dim_B, z_dim, share_dim, z_A=None, z_B=None, eps=1e-10, lambda_0=0.1, lambda_1=0.1, lambda_2=100,
                 lambda_3=0.01,
                 lambda_4=100, learning_rate=1e-4):
        self.batch_size = batch_size
        self.dim_A = dim_A
        self.dim_B = dim_B
        self.encode_dim_A = encode_dim_A
        self.encode_dim_B = encode_dim_B
        self.decode_dim_A = decode_dim_A
        self.decode_dim_B = decode_dim_B
        self.adv_dim_A = adv_dim_A
        self.adv_dim_B = adv_dim_B
        self.z_dim = z_dim
        self.eps = eps
        self.share_dim = share_dim
        self.lambda_0 = lambda_0
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_4 = lambda_4
        self.learning_rate = learning_rate
        self.active_function = tf.nn.tanh
        # self.z_A = z_A
        # self.z_B = z_B
        self.train = True
        self.freeze = True
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    def enc(self, x, scope, encode_dim, reuse=False):
        x_ = x
        if self.train:
            x_ = tf.nn.dropout(x_, 0.7)
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(encode_dim)):
                x_ = fully_connected(x_, encode_dim[i], scope="enc_%d"%i,
                                     weights_regularizer=self.regularizer, trainable=self.freeze)
                # y = maxout(x_, encode_dim[i])
                # x_ = tf.reshape(y, x_.shape)
                x_ = tf.nn.leaky_relu(x_, alpha=0.5)
                # x_ = tf.nn.tanh(x_)

                print(x_.shape)
        return x_

    def dec(self, x, scope, decode_dim, reuse=False):
        x_ = x
        if self.train:
            x_ = tf.nn.dropout(x_, 0.7)
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(decode_dim)-1):
                x_ = fully_connected(x_, decode_dim[i], scope="dec_%d" % i,
                                     weights_regularizer=self.regularizer, trainable=self.freeze)
                x_ = tf.nn.leaky_relu(x_, alpha=0.5)
                # x_ = tf.nn.tanh(x_)
            x_ = fully_connected(x_, decode_dim[-1], scope="last_dec",
                             weights_regularizer=self.regularizer, trainable=self.freeze)
        return x_

    def adversal(self, x, scope, adv_dim, reuse=False):
        x_ = x

        with tf.variable_scope(scope, reuse=reuse):
            # if self.train:
            x_ = tf.nn.dropout(x_, 0.7)
            for i in range(len(adv_dim)-1):
                x_ = fully_connected(x_, adv_dim[i], self.active_function, scope="adv_%d" % i)
            x_ = fully_connected(x_, adv_dim[-1], scope="adv_last")
        return x_

    def share_layer(self, x, scope, dim, reuse=False):
        x_ = x
        if self.train:
            x_ = tf.nn.dropout(x_, 0.7)
        with tf.variable_scope(scope, reuse=reuse):
            for i in range(len(dim)):
                x_ = fully_connected(x_, dim[i],  scope="share_%d"%i,
                                     weights_regularizer=self.regularizer)
                # y = maxout(x_, dim[i])
                # x_ = tf.reshape(y, x_.shape)
                x_ = tf.nn.leaky_relu(x_, alpha=0.5)
                # x_ = tf.nn.tanh(x_)

        return x_

    def gen_z(self, h, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            z_mu = fully_connected(h, self.z_dim, self.active_function, scope="z_mu", weights_regularizer=self.regularizer)
            z_sigma = fully_connected(h, self.z_dim,  self.active_function, scope="z_sigma",
                                      weights_regularizer=self.regularizer)
            e = tf.random_normal(tf.shape(z_mu))
            z = z_mu + tf.sqrt(tf.maximum(tf.exp(z_sigma), self.eps)) * e
        return z, z_mu, z_sigma

    def encode(self, x, scope, dim, reuse_enc, reuse_share, reuse_z=False):
        h = self.enc(x, "encode_%s"%scope, dim, reuse_enc)
        h = self.share_layer(h, "encode", self.share_dim, reuse_share)
        z, z_mu, z_sigma = self.gen_z(h, "encode", reuse=reuse_z)
        return z, z_mu, z_sigma

    def decode(self, x, scope, dim, reuse_dec, reuse_share):
        x = self.share_layer(x, "decode", self.share_dim[::-1], reuse_share)
        x = self.dec(x, "decode_%s"%scope, dim, reuse_dec)
        return x

    def loss_kl(self, mu, sigma):
        return 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(mu) + tf.exp(sigma) - sigma - 1, 1))

    def loss_reconstruct(self, x, x_recon):

        log_softmax_var = tf.nn.log_softmax(x_recon)

        neg_ll = -tf.reduce_mean(tf.reduce_sum(
            log_softmax_var * x,
            axis=-1))
        return neg_ll


    def loss_recsys(self, pred, label):
        return tf.reduce_mean(tf.reduce_sum(K.binary_crossentropy(label, pred), axis=1))

    def loss_discriminator(self, x, x_fake):
        loss_real = tf.reduce_mean(tf.squared_difference(x, 1))
        loss_fake = tf.reduce_mean(tf.squared_difference(x_fake, 0))
        return loss_real + loss_fake
    def loss_generator(self, x):
        return tf.reduce_mean(tf.squared_difference(x, 1))


    def build_model(self):
        self.x_A = tf.placeholder(tf.float32, [None, self.dim_A], name='input_A')
        self.x_B = tf.placeholder(tf.float32, [None, self.dim_B], name='input_B')

        x_A = self.x_A
        x_B = self.x_B

        # VAE for domain A
        z_A, z_mu_A, z_sigma_A = self.encode(x_A, "A", self.encode_dim_A, False, False)
        y_AA = self.decode(z_A, "A", self.decode_dim_A, False, False)

        # VAE for domain B
        z_B, z_mu_B, z_sigma_B = self.encode(x_B, "B", self.encode_dim_B, False, True, True)
        y_BB = self.decode(z_B, "B", self.decode_dim_B, False, True)

        # Adversal
        y_BA = self.decode(z_B, "A", self.decode_dim_A, True, True)
        adv_AA = self.adversal(y_AA, "adv_A", self.adv_dim_A)
        adv_BA = self.adversal(y_BA, "adv_A", self.adv_dim_A, reuse=True)


        y_AB = self.decode(z_A, "B", self.decode_dim_B, True, True)
        adv_BB = self.adversal(y_BB, "adv_B", self.adv_dim_B)
        adv_AB = self.adversal(y_AB, "adv_B", self.adv_dim_B, reuse=True)

        # Cycle - Consistency
        z_ABA, z_mu_ABA, z_sigma_ABA = self.encode(y_AB, "B", self.encode_dim_B, True, True, True)
        y_ABA = self.decode(z_ABA, "A", self.decode_dim_A, True, True)
        z_BAB, z_mu_BAB, z_sigma_BAB = self.encode(y_BA, "A", self.encode_dim_A, True, True, True)
        y_BAB = self.decode(z_BAB, "B", self.decode_dim_B, True, True)



        # Loss VAE
        loss_VAE_A = self.lambda_1 * self.loss_kl(z_mu_A, z_sigma_A) + self.lambda_2 * self.loss_reconstruct(x_A, y_AA)
        loss_VAE_B = self.lambda_1 * self.loss_kl(z_mu_B, z_sigma_B) + self.lambda_2 * self.loss_reconstruct(x_B, y_BB)
        self.loss_VAE = loss_VAE_A + loss_VAE_B

        # Loss GAN
        loss_d_A = self.lambda_0 * self.loss_discriminator(adv_AA, adv_BA)
        loss_d_B = self.lambda_0 * self.loss_discriminator(adv_BB, adv_AB)
        self.loss_d= loss_d_A + loss_d_B
        self.adv_AA = adv_AA
        self.adv_AB = adv_BA
        self.y_AA = y_AA
        self.y_BB = y_BB

        # Loss cycle - consistency (CC)
        loss_CC_A = self.lambda_3 * self.loss_kl(z_mu_ABA, z_sigma_ABA) + \
                    self.lambda_4 * self.loss_reconstruct(x_A,y_BA) + self.lambda_4 * self.loss_reconstruct(x_A, y_ABA)
        loss_CC_B = self.lambda_3 * self.loss_kl(z_mu_BAB, z_sigma_BAB) + self.lambda_4 * \
                    self.loss_reconstruct(x_B,y_AB) + self.lambda_4 * self.loss_reconstruct(x_B, y_BAB)



        self.loss_CC = loss_CC_A + loss_CC_B

        self.loss_val_a = self.lambda_4 * self.loss_reconstruct(x_A, y_BA)
        self.loss_val_b = self.lambda_4 * self.loss_reconstruct(x_B, y_AB)
        self.y_BA = y_BA
        self.y_AB = y_AB

        self.loss_gen =  self.loss_CC + 0.1 * tf.losses.get_regularization_loss() +\
                        self.loss_generator(y_AB) + self.loss_generator(y_ABA) + self.loss_generator(y_BAB) +\
                        self.loss_generator(y_BA)
        # self.loss_gen = drself.loss_CC + 0.1 * tf.losses.get_regularization_loss() - loss_d_A - loss_d_B



        self.loss_dis = loss_d_A + loss_d_B


        self.train_op_VAE_A = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_VAE_A)
        self.train_op_VAE_B = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_VAE_B)
        self.train_op_gen = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_gen)

        adv_var_A = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="adv_A")
        adv_var_B = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="adv_B")
        print(adv_var_A, adv_var_B)
        self.train_op_dis_A = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_d_A,
                                                                                         var_list=adv_var_A)
        self.train_op_dis_B = tf.train.AdamOptimizer(self.learning_rate).minimize(loss_d_B,
                                                                                  var_list=adv_var_B)


def create_dataset(A="Health", B="Clothing"):
    dense_A = read_data("data/%s_%s/%s_user_product.txt"%(A,B,A))
    num_A = 0
    for i in dense_A:
        if num_A < max(i):
            num_A = max(i)
    num_A += 1
    user_A = one_hot_vector(dense_A, num_A)

    dense_B = read_data("data/%s_%s/%s_user_product.txt"%(A, B, B))
    num_B = 0
    for i in dense_B:
        if num_B < max(i):
            num_B = max(i)
    num_B += 1
    user_B = one_hot_vector(dense_B, num_B)

    return user_A, user_B, dense_A, dense_B, num_A, num_B

def read_data(filename):
    f = list(open(filename).readlines())
    f = [i.split(" ") for i in f]
    f = [[int(j) for j in i] for i in f]
    f = [i[1:] for i in f]
    return f

def read_data2(filename):
    data = list(open(filename).readlines())
    data = data[1:]
    n_data = len(data)
    print(len(data))
    data = [d.strip() for d in data]
    data = [d.split(", ") for d in data]
    data = [d[:3] for d in data]
    data = np.array(data).reshape(n_data, 3).astype(np.int32)
    return data

def one_hot_vector(A, num_product):
    one_hot_A = np.zeros((len(A), num_product))

    for i, row in enumerate(A):
        for j in row:
            if j< num_product:
                one_hot_A[i,j] = 1
    return one_hot_A

def one_hot_vector2(A, num_product):
    one_hot = np.zeros((6557, num_product))
    for i in A:
        one_hot[i[0], i[1]] = i[2]
    return one_hot

def test_same_domain(dense, num_product):
    input_user = np.zeros((len(dense), num_product))
    dense_test = [0]*len(dense)
    for i,d in enumerate(dense):
        num_input = int(len(d)*0.8)
        for j in range(0, num_input):
            input_user[i, d[j]] = 1
        dense_test[i] = d[num_input:]
    return input_user, dense_test

def calc_recall_same_domain(pred, test, m=[100], type=None):

    for k in m:
        pred_ab = np.argsort(-pred)
        recall = []
        ndcg = []
        for i in range(len(pred_ab)):
            num_train = int(len(test[i])*0.8)
            u_train = test[i][:num_train]
            u_test = test[i][num_train:]
            p = list(pred_ab[i, :(k+num_train)])
            for t in u_train:
                if t in p:
                    p.remove(t)
            p = p[:k]

            hits = set(u_test) & set(p)

            #recall
            recall_val = float(len(hits)) / len(u_test)
            recall.append(recall_val)

            #ncdg
            score = []
            for j in range(k):
                if p[j] in hits:
                    score.append(1)
                else:
                    score.append(0)
            actual = dcg_score(score, pred[i, p], k)
            best = dcg_score(score, score, k)
            if best == 0:
                ndcg.append(0)
            else:
                ndcg.append(float(actual) / best)

        print("k= %d, recall %s: %f, ndcg: %f"%(k, type, np.mean(recall), np.mean(ndcg)))

def calc_recall(pred, test, m=[100], type=None):

    for k in m:
        pred_ab = np.argsort(-pred)[:, :k]
        recall = []
        ndcg = []
        for i in range(len(pred_ab)):
            p = pred_ab[i]
            hits = set(test[i]) & set(p)

            #recall
            recall_val = float(len(hits)) / len(test[i])
            # if recall_val > 0.5:
            #     print(i, p, hits, type)
            recall.append(recall_val)

            #ncdg
            score = []
            for j in range(k):
                if p[j] in hits:
                    score.append(1)
                else:
                    score.append(0)
            actual = dcg_score(score, pred[i, p], k)
            best = dcg_score(score, score, k)
            if best == 0:
                ndcg.append(0)
            else:
                ndcg.append(float(actual) / best)

        print("k= %d, recall %s: %f, ndcg: %f"%(k, type, np.mean(recall), np.mean(ndcg)))


    return np.mean(np.array(recall))

def dcg_score(y_true, y_score, k=50):
    """Discounted cumulative gain (DCG) at rank K.

    Parameters
    ----------
    y_true : array, shape = [n_samples]
        Ground truth (true relevance labels).
    y_score : array, shape = [n_samples, n_classes]
        Predicted scores.
    k : int
        Rank.

    Returns
    -------
    score : float
    """
    order = np.argsort(y_score)[::-1]
    y_true = np.take(y_true, order[:k])

    gain = 2 ** y_true - 1

    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gain / discounts)

def calc_rmse(pred, test):
    idx = np.where(test != 0)
    pred = pred[idx]
    test = test[idx]
    return np.sqrt(np.mean((test-pred)**2))

def load_rating(path, thred, test_size):
    dense_A = []
    dense_B = []
    i = 0
    for line in open(path):
        a = line.strip().split()
        if i >= test_size:
            l = [int(x) for x in a]
            if l[0] < thred:
                l = [i for i in l if i < thred]
                dense_A.append(l)
            else:
                l = [i for i in l if i >= thred]
                l =[i - thred for i in l ]
                print(l)
                dense_B.append(l)
        i += 1
    return dense_A, dense_B

def main():
    iter = 300
    batch_size= 500
    args = parser.parse_args()
    A = args.A
    B = args.B
    checkpoint_dir = "translation/%s_%s/"%(A,B)
    user_A, user_B, dense_A, dense_B, num_A, num_B = create_dataset(A, B)
    z_dim = 50
    adv_dim_A = adv_dim_B = [100, 1]

    if A == "Drama" or A == "Romance":
        k = [10, 20, 30, 40, 50]
        dim = 200
        share = 100
    else:
        k = [50, 100, 150, 200, 250, 300]
        dim = 600
        share = 200

    print(k)

    encoding_dim_A = [600]
    encoding_dim_B = [600]
    share_dim = [200]
    decoding_dim_A = [600, num_A]
    decoding_dim_B = [600, num_B]


    assert len(user_A) == len(user_B)
    perm = np.random.permutation(len(user_A))
    total_data = len(user_A)
    train_size = int(total_data * 0.7)
    val_size = int(total_data * 0.05)

    # user_A = user_A[perm]
    # user_B = user_B[perm]

    user_A_train = user_A[:train_size]
    user_B_train = user_B[:train_size]

    user_A_val = user_A[train_size:train_size+val_size]
    user_B_val = user_B[train_size:train_size+val_size]
    user_A_test = user_A[train_size+val_size:]
    user_B_test = user_B[train_size+val_size:]

    dense_A_test = dense_A[(train_size + val_size):]
    dense_B_test = dense_B[(train_size + val_size):]


    model = Translation(batch_size, num_A, num_B, encoding_dim_A, decoding_dim_A, encoding_dim_B,
                        decoding_dim_B, adv_dim_A, adv_dim_B, z_dim, share_dim, learning_rate=1e-3, lambda_2=1,
                        lambda_4=0.1)
    model.build_model()

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=20)
    max_recall = 0
    dense_A_val = dense_A[train_size:train_size+val_size]
    dense_B_val = dense_B[train_size:train_size+val_size]

    for i in range(1, iter):
        shuffle_idx = np.random.permutation(train_size)
        train_cost = 0
        for j in range(int(train_size/batch_size)):
            list_idx = shuffle_idx[j*batch_size:(j+1)*batch_size]
            x_A = user_A_train[list_idx]
            x_B = user_B_train[list_idx]

            feed = {model.x_A: x_A,
                    model.x_B: x_B}

            if i <20:
                _, loss_vae = sess.run([model.train_op_VAE_A, model.loss_VAE], feed_dict=feed)
                _, loss_vae = sess.run([model.train_op_VAE_B, model.loss_VAE], feed_dict=feed)
                loss_gen = loss_dis = loss_cc = 0
            # elif i>=50 and i < 100:
            #     _, loss_vae = sess.run([model.train_op_VAE_B, model.loss_VAE], feed_dict=feed)
            #     loss_gen = loss_dis = loss_cc = 0
            else:
                model.freeze = False
                _, loss_gen, loss_vae, loss_cc = sess.run([model.train_op_gen, model.loss_gen, model.loss_VAE,
                                                        model.loss_CC], feed_dict=feed)

                sess.run([model.train_op_dis_A],feed_dict=feed)
                # _, loss_gen, loss_vae, loss_cc = sess.run([model.train_op_gen_B, model.loss_gen, model.loss_VAE,
                #                                            model.loss_CC], feed_dict=feed)
                sess.run([model.train_op_dis_B], feed_dict=feed)
                loss_dis = 0
            # print(adv_AA, adv_AB)
            # _, loss_dis = sess.run([model.train_op_dis, model.loss_dis], feed_dict=feed)
            # _, loss_rec = sess.run([model.train_op_rec, model.loss_rec], feed_dict=feed)

        # print("Loss last batch: loss gen %f, loss dis %f, loss vae %f, loss rec %f, loss cc %f"%(loss_gen, loss_dis,
        #                                                                         loss_vae, loss_rec, loss_cc))

        # Validation Process
        if i%10 == 0:
            model.train = False
            print("Loss last batch: loss gen %f, loss dis %f, loss vae %f,loss cc %f" % (
            loss_gen, loss_dis, loss_vae, loss_cc))
            #                                                                         loss_vae, loss_gan, loss_cc))
            loss_gen, loss_val_a, loss_val_b, y_ba, y_ab = sess.run([model.loss_gen, model.loss_val_a,
                                                                     model.loss_val_b, model.y_BA, model.y_AB],
                                              feed_dict={model.x_A:user_A_val, model.x_B:user_B_val})


            recall = calc_recall(y_ba, dense_A_val, [50]) + calc_recall(y_ab, dense_B_val, [50])
            print("Loss gen: %f, Loss val a: %f, Loss val b: %f, recall %f" % (loss_gen, loss_val_a, loss_val_b,
                                                                               recall))
            if recall > max_recall:
                max_recall = recall
                saver.save(sess, os.path.join(checkpoint_dir, 'translation-model'), i)
                loss_test_a, loss_test_b, y_ab, y_ba = sess.run(
                    [model.loss_val_a, model.loss_val_b, model.y_AB, model.y_BA],
                 feed_dict={model.x_A: user_A_test, model.x_B: user_B_test})
                print("Loss test a: %f, Loss test b: %f" % (loss_test_a, loss_test_b))

                # y_ab = y_ab[test_B]
                # y_ba = y_ba[test_A]

                calc_recall(y_ba, dense_A_test, k, type="A")
                calc_recall(y_ab, dense_B_test, k, type="B")

                #test same domain
                # input_A_test, domain_A_test = test_same_domain(dense_A_test, num_A)
                # y_aa = sess.run(model.y_AA, feed_dict={model.x_A:input_A_test})
                # calc_recall(y_aa, domain_A_test, [50], type="A")
                #
                # input_B_test, domain_B_test = test_same_domain(dense_B_test, num_B)
                # y_bb = sess.run(model.y_BB, feed_dict={model.x_B:input_B_test})
                # calc_recall(y_bb, domain_B_test, [50], type="B")

            model.train = True
        if i%100 == 0:
            model.learning_rate /= 10
            print("decrease lr to %f"%model.learning_rate)

            # pred = np.array(y_ab).flatten()
            # test = np.array(user_B_val).flatten()
            # rmse = calc_rmse(pred, test)
            # print("Loss val a: %f, Loss val b: %f, rmse %f" % (loss_val_a, loss_val_b, rmse))
            # if rmse < max_recall:
            #     max_recall = rmse
            #     saver.save(sess, os.path.join(checkpoint_dir, 'translation-model'), i)

    print(max_recall)
    # model.train = False
    # loss_test_a, loss_test_b, y_ab, y_ba = sess.run([model.loss_val_a, model.loss_val_b, model.y_AB, model.y_BA],
    #                         feed_dict={model.x_A: user_A_test[200:],model.x_B: user_B_test[200:]})
    # print("Loss test a: %f, Loss test b: %f" % (loss_test_a, loss_test_b))
    # model.train = True
    #
    # dense_A_test = dense_A[(train_size+200):]
    # dense_B_test = dense_B[(train_size+200):]
    #
    #
    # print("recall B: %f"%(calc_recall(y_ab, dense_B_test)))
    # print("recall A: %f" % (calc_recall(y_ba, dense_A_test)))

    # pred_a = np.array(y_ba).flatten()
    # test_a = np.array(user_A_test).flatten()
    # print("rmse A %f"%calc_rmse(pred_a, test_a))
    #
    # pred_a = np.array(y_ab).flatten()
    # test_a = np.array(user_B_test).flatten()
    # print("rmse B %f" % calc_rmse(pred_a, test_a))

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--A',  type=str, default="Health",
                   help='domain A')
parser.add_argument('--B',  type=str, default='Grocery',
                   help='domain B')
parser.add_argument('--k',  type=int, default=100,
                   help='top-K')
if __name__ == '__main__':
    main()



