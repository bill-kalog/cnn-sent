import time
import datetime
import os
import tensorflow as tf
# from conf import config
import numpy as np
from model import RNN
from model import RNN_Attention
from dmn import DMN
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib import learn
import process_utils
import sys
import json
import operator
import pandas as pd
import re



def eval_model(sess, g, checkpoint_paths, data, config):

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







