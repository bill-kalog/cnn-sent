import time
import datetime
import os
import tensorflow as tf
# from conf import config
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib import learn
import process_utils
import sys
import json
import operator
import pandas as pd
import re


def eval_model(sess, g, checkpoint_paths, data, config):

    def test_step(x_batch, y_batch):
        """
        Evaluate model on a test set
        """
        feed_dict = {
            x_values: x_batch,
            y_values: y_batch,
            dropout_prob: 1.0
        }
        accuracy = sess.run(
            [graph_accuracy],
            feed_dict)[0]
        print("---- accuracy {} ----".format(accuracy))

    def save_test_summary(x_batch, y_batch, x_strings_batch, name_):
        '''
        save info for a batch in order to plot in
        bokeh later
        '''
        path_ = os.path.join(out_dir, name_)
        y_net = []
        prob_net = []
        layer = []
        true_labels = []
        feed_dict = {
            x_values: x_batch,
            y_values: y_batch,
            dropout_prob: 1.0
        }
        output_ = [graph_predictions, graph_true_predictions,
                   graph_probs, graph_state_]
        predictions, true_pred, probs, fc_layer = sess.run(
            output_, feed_dict)
        prob_net = probs.tolist()
        layer = fc_layer.tolist()
        y_net = predictions.tolist()
        true_labels = true_pred.tolist()

        process_utils.save_info(
            x_strings_batch, true_labels, y_net, prob_net, layer, path_)

    # output directory for data
    timestamp = str(int(time.time()))
    out_dir = os.path.join(checkpoint_paths, "..", "evaluations", timestamp)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    # load data
    dx_train, y_train, dx_test, y_test = data
    vc_path = os.path.join(checkpoint_paths, "..", "vocabulary")
    vc_processor = learn.preprocessing.VocabularyProcessor.restore(vc_path)
    x_test = np.array(list(vc_processor.transform(dx_test)))

    # load old configuration file
    saved_conf_path = os.path.join(checkpoint_paths, "..", "config.json")
    with open(saved_conf_path) as json_data:
        saved_conf = json.load(json_data)

    # retrieve configuration info
    config['n_words'] = saved_conf['n_words']
    config['sentence_len'] = saved_conf['sentence_len']

    # load model
    last_model = tf.train.latest_checkpoint(checkpoint_paths)

    num_ = None
    if num_ is not None:
        temp = last_model[:last_model.find("model-") + len("model-")]
        last_model = "{}{}".format(temp, num_)

    print ("About to load: {}".format(last_model))
    saver = tf.train.import_meta_graph("{}.meta".format(last_model))
    saver.restore(sess, last_model)
    print ("Model loaded")

    # Get placeholders
    x_values = g.get_operation_by_name('x').outputs[0]
    y_values = g.get_operation_by_name('y').outputs[0]
    dropout_prob = g.get_operation_by_name("dropout_prob").outputs[0]

    graph_predictions = g.get_operation_by_name(
        "output/predictions").outputs[0]
    graph_true_predictions = g.get_operation_by_name(
        "accuracy/ArgMax").outputs[0]
    graph_probs = g.get_operation_by_name(
        "output/Softmax").outputs[0]
    graph_state_ = g.get_operation_by_name(
        "Reshape").outputs[0]
    graph_accuracy = g.get_operation_by_name(
        "accuracy/accuracy").outputs[0]

    test_step(x_test, y_test)
    save_test_summary(
        x_test, y_test, dx_test, 'metrics_test_{}.pkl'.format(
            last_model[last_model.find("model-") + len("model-"):]))




