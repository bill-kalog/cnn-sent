import tensorflow as tf
import numpy as np
from math import ceil
import sys


class CNN(object):
    """ CNN model """

    def __init__(self, config, sess, word_vectors=[]):
        self.n_epochs = config['n_epochs']
        self.kernel_sizes = config['kernel_sizes']
        self.n_filters = config['n_filters']
        self.edim = config['edim']
        self.n_words = config['n_words']
        self.std_dev = config['std_dev']
        self.l2_reg = config['l2_regularization']
        self.sentence_len = config['sentence_len']
        self.x = tf.placeholder(tf.int32, [None, self.sentence_len], name="x")
        self.num_classes = config['classes_num']
        self.y = tf.placeholder(tf.float32, [None, self.num_classes], name="y")
        self.dropout_prob = tf.placeholder(tf.float32, name="dropout_prob")
        self.word_vectors = word_vectors
        # Keeping track of l2 regularization loss
        self.l2_loss = tf.constant(0.0)

        self.make_graph(config)

    def make_graph(self, config):

        pooled_outputs = []
        self.conv_loss = 0

        if not config['seperate_filters']:

            with tf.name_scope("word_embeddings"):
                print(" ============  Word vector shape {}  ======".format(
                    self.word_vectors.shape))

                self.word_embeddings = []
                all_embeddings = []
                # counter = 0
                for index_ in range(self.word_vectors.shape[0]):
                    if config['train_embeddings'][index_] is None:
                        continue
                    extracted_emb = tf.get_variable(
                        "W0_" + str(index_), shape=[self.n_words, self.edim],
                        trainable=config['train_embeddings'][index_],
                        initializer=tf.constant_initializer(
                            np.array(self.word_vectors[index_]))
                    )
                    self.word_embeddings.append(extracted_emb)
                    temp = tf.nn.embedding_lookup(
                        extracted_emb, self.x)
                    all_embeddings.append(temp)
            self.embedded_chars = tf.stack(all_embeddings, axis=3)

            print ("emb_char shape: {}".format(self.embedded_chars.shape))

            for i, kernel_size in enumerate(self.kernel_sizes):
                with tf.name_scope("conv-maxpool-%s" % kernel_size):
                    # declare conv layer
                    filter_shape = [kernel_size, self.edim,
                                    len(all_embeddings), self.n_filters]
                    W = tf.Variable(tf.truncated_normal(
                        filter_shape, stddev=0.01), trainable=True,
                        name="W")
                    b = tf.Variable(tf.constant(
                        0.1, shape=[self.n_filters]), trainable=True,
                        name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars,
                        W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                    # declare non linearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    self.conv_loss += tf.nn.l2_loss(W)
                    # Max pool relu output
                    pool = tf.nn.max_pool(
                        h, ksize=[1, self.sentence_len - kernel_size + 1, 1, 1],
                        strides=[1, 1, 1, 1], padding="VALID", name="pool")
                    pooled_outputs.append(pool)

            # Combine all the pooled features
            total_filters_num = self.n_filters * len(self.kernel_sizes)
            # self.h_pool = tf.concat(3, pooled_outputs) #old version
            # self.h_pool = tf.concat(pooled_outputs, 3)
            # self.h_pool_flat = tf.reshape(self.h_pool, [-1, total_filters_num])

        else:  # use seperate filters
            with tf.name_scope("word_embeddings"):
                print(" ==================  Word vector shape {}  ======".format(self.word_vectors.shape))
                self.word_embeddings = []
                all_embeddings = []
                for index_ in range(self.word_vectors.shape[0]):
                    if config['train_embeddings'][index_] is None:
                        continue
                    extracted_emb = tf.get_variable(
                        "W0_" + str(index_), shape=[self.n_words, self.edim],
                        trainable=config['train_embeddings'][index_],
                        initializer=tf.constant_initializer(
                            np.array(self.word_vectors[index_]))
                    )
                    self.word_embeddings.append(extracted_emb)
                    temp = tf.nn.embedding_lookup(
                        extracted_emb, self.x)
                    embedded_chars_expanded = tf.expand_dims(
                        temp, -1)
                    all_embeddings.append(embedded_chars_expanded)

            for index_, embedded_char_matrix in enumerate(all_embeddings):
                for i, kernel_size in enumerate(self.kernel_sizes):
                    with tf.name_scope("conv-maxpool-{}-{}".format(
                            kernel_size, index_)):
                        # declare conv layer
                        filter_shape = [kernel_size, self.edim,
                                        1, self.n_filters]
                        W = tf.Variable(tf.truncated_normal(
                            filter_shape, stddev=0.01), trainable=True,
                            name="W")
                        b = tf.Variable(tf.constant(
                            0.01, shape=[self.n_filters]), trainable=True,
                            name="b")
                        conv = tf.nn.conv2d(
                            embedded_char_matrix,
                            # test_stop_grad,
                            W, strides=[1, 1, 1, 1], padding="VALID",
                            name="conv")
                        # declare non linearity
                        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                        # self.conv_loss += tf.nn.l2_loss(W)
                        # Max pool relu output
                        pool = tf.nn.max_pool(
                            h, ksize=[1, self.sentence_len - kernel_size + 1, 1, 1],
                            strides=[1, 1, 1, 1], padding="VALID", name="pool")
                        pooled_outputs.append(pool)

            # Combine all the pooled features
            total_filters_num = self.n_filters * len(self.kernel_sizes) * \
                len(all_embeddings)

        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, total_filters_num])

        # use dropout before fc layer
        with tf.name_scope("drop-out"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_prob)

        # build fc layer
        with tf.name_scope("output"):
            self.fully_con_W = tf.Variable(tf.truncated_normal(
                [total_filters_num, self.num_classes], stddev=0.1),
                name="W_fc")
            b = tf.Variable(tf.constant(
                0.1, shape=[self.num_classes]), name="b")
            self.l2_loss = tf.nn.l2_loss(self.fully_con_W)
            # self.l2_loss += self.conv_loss
            # self.l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(
                self.h_drop, self.fully_con_W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.scores, labels=self.y)
            self.loss = tf.reduce_mean(losses) + self.l2_loss * self.l2_reg

        # Calculate Accuracy
        with tf.name_scope("accuracy"):
            correct_pred = tf.equal(self.predictions, tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_pred, "float"), name="accuracy")
