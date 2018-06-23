import tensorflow as tf
import numpy as np
import entropy

class Model:  # (Basically uses 'get' functions with lazy loading. Structure inspired by https://danijar.com/structuring-your-tensorflow-models/)

    def __init__(self, input_ph, target_ph, learning_rate_ph, beta, d, squared_IB_functional):
        self.input_ph = input_ph
        self.target_ph = target_ph
        self.learning_rate_ph = learning_rate_ph
        self.beta = beta
        self.d = d
        self.squared_IB_functional = squared_IB_functional

        self.log_eta2 = tf.get_variable('log_eta2', dtype=tf.float32, initializer=0.001) # maximum likelihood estimate for log variance of mixture model
        self.log_sigma2 = tf.constant(np.log(1), dtype=tf.float32)                       # encoder noise variance

        # for fitting the GMM
        self.distance_matrix_ph = tf.placeholder(dtype=tf.float32, shape=[None, None])  # placeholder to speed up scipy optimizer
        self.neg_llh_eta = entropy.GMM_negative_LLH(self.distance_matrix_ph, self.log_eta2, self.d)   # negative log-likelihood for the 'width' of the GMM
        self.eta_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.neg_llh_eta, var_list=[self.log_eta2])

        # learning curves
        self.learning_curve = []
        self.Ixt_curve = []
        self.Iyt_curve = []

        # build the graph
        self.encoder()
        self.decoder()
        self.distance_matrix()
        self.loss()
        self.training_step()
        self.evaluate()

    def encoder(self):
        if not hasattr(self, '_T'):
            with tf.variable_scope('encoder'):
                T1 = tf.layers.dense(self.input_ph, 800, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
                T2 = tf.layers.dense(T1, 800, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
                self._T_no_noise = tf.layers.dense(T2, self.d, activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())

                sigma = tf.exp(0.5 * self.log_sigma2)
                self._T = self._T_no_noise + tf.random_normal(shape=tf.shape(self._T_no_noise), mean=0.0, stddev=sigma, dtype=tf.float32)

        return self._T, self._T_no_noise

    def decoder(self):
        if not hasattr(self, '_Y'):
            with tf.variable_scope('decoder'):
                T, _ = self.encoder()
                Y1 = tf.layers.dense(T, 800, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
                self._Y = tf.layers.dense(Y1, self.target_ph.shape[1], activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer())

        return self._Y

    def distance_matrix(self):
        if not hasattr(self, '_distance_matrix'):
            _, T_no_noise = self.encoder()
            self._distance_matrix = entropy.pairwise_distance(T_no_noise)

        return self._distance_matrix

    def loss(self):
        if not hasattr(self, '_loss'):
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target_ph, logits=self.decoder()))
            H_T = entropy.GMM_entropy(self.distance_matrix(), tf.log(tf.exp(self.log_sigma2) + tf.exp(self.log_eta2)), self.d, 'upper')
            H_T_given_X = entropy.Gaussian_entropy(self.d, self.log_sigma2)
            self._Ixt = H_T - H_T_given_X
            self._Iyt = tf.log(10.0) - cross_entropy
            if self.squared_IB_functional:
                self._loss = self.beta * tf.square(self._Ixt) - self._Iyt
            else:
                self._loss = self.beta * self._Ixt - self._Iyt

        return self._loss, self._Ixt, self._Iyt

    def training_step(self):
        if not hasattr(self, '_training_step'):
            adam_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=0.0001)
            loss, _, _ = self.loss()
            self._training_step = adam_optimizer.minimize(loss, var_list=tf.trainable_variables(scope='encoder') + tf.trainable_variables('decoder'))

        return self._training_step

    def evaluate(self):  # evaluate the models accuracy
        if not hasattr(self, '_evaluate'):
            correct = tf.equal(tf.argmax(self.target_ph, axis=1), tf.argmax(self.decoder(), axis=1))
            self._evaluate = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

        return self._evaluate

    def update_learning_curves(self, loss, Ixt, Iyt):
        self.learning_curve.append(loss)
        self.Ixt_curve.append(Ixt)
        self.Iyt_curve.append(Iyt)
