import tensorflow as tf
# import data_utils
from conf import config
# from model import CNN
from datasets import Dataset
from word_vectors import WordVectors
import train
import sys

# data, labels, w2idx = data_utils.get_data(config['paths'])


dataset = Dataset("SST")

# sys.exit(0)

# dataset = Dataset("IMDB", preprocess=True)
# dataset = Dataset("MR", preprocess=True)


# path_1 = config["pretrained_vectors"]
pretrained_vectors = []
for index, type_ in enumerate(config['word_vector_type']):
    pretrained_vectors.append(WordVectors(
        type_, config["pretrained_vectors"][index]))
    print ("loaded vectors {}".format(config['word_vector_type'][index]))


# test_list = dataset.cv_split()
# train_set, labels_tr, dev_set, labels_dev = dataset.cv_split(index=5)
# data = dataset.cv_split(index=0)

# data = dataset.cv_split(index=5)

data = dataset.cv_split(index=2)
# # data = dataset.cv_split(index=5)


# dataset_1 = Dataset("MR", preprocess=True)
# sp_1 = dataset_1.cv_split(index=5)

# # data = [dataset.tokenized[:25000], dataset.labels_verbose[:25000],
# #         dataset.tokenized[25000:], dataset.labels_verbose[25000:]]

# data = [data[0] + sp_1[0], data[1] + sp_1[1], sp_1[2], sp_1[3]]


# data[2] = sp_1[2]
# data[3] = sp_1[3]

# print (len(data[0]), len(data[2]), len(data[1]), len(data[3]))

# data = dataset.cv_split(index=1)
# print (len(data[0]), len(data[2]), len(data[1]), len(data[3]))

# data = dataset.cv_split(index=3)
# print (len(data[0]), len(data[2]), len(data[1]), len(data[3]))

# sys.exit(0)


# config['n_words'] = len(w2idx) + 1

# with tf.Session() as sess:
#     net = models.CNN(config, sess)
#     net.train(data, labels)
# print (test_list[0][0][0], test_list[0][1][0])
# print (len(train_set), len(dev_set), len(labels_tr), len(labels_dev))
# sys.exit(0)

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        train.set_train(
            sess, config, data,
            pretrained_vectors)
