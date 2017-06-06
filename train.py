import time
import datetime
import os
import tensorflow as tf
# from conf import config
import numpy as np
from model import CNN
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.contrib import learn
import process_utils
import sys
import json


def set_train(sess, config, data, pretrained_embeddings=[]):
    best_achieved_accuracy = 0  # best achieved vlaidation accuracy
    epochs_best_acc_not_changed = 0  # number of epochs that the best accuracy hasn't improved
    candidate_accuracy = 0  # accuracy at current step

    # Build vocabulary
    # x_train, y_train, x_dev, y_dev = data
    dx_train, y_train, dx_dev, y_dev = data

    max_document_length = max([len(x.split(" ")) for x in dx_train])
    # trim sentences if too big
    if max_document_length > 500:
        max_document_length = 100
    vocab_processor = learn.preprocessing.VocabularyProcessor(
        max_document_length)

    vocab_processor.fit(dx_train + dx_dev)  # build vocabulary based on both train and dev set
    # vocab_processor.fit(dx_train)
    x_train = np.array(list(vocab_processor.transform(dx_train)))
    x_dev = np.array(list(vocab_processor.transform(dx_dev)))

    # print (x_train[1])
    # ############ vocabulary_ info from
    # http://stackoverflow.com/questions/40661684/tensorflow-vocabularyprocessor#40741660
    # Extract word:id mapping from the object.
    vocab_dict = vocab_processor.vocabulary_._mapping
    # print (vocab_dict)
    # sys.exit(0)

    # # Sort the vocabulary dictionary on the basis of values(id).
    # # Both statements perform same task.
    # sorted_vocab = sorted(vocab_dict.items(), key=operator.itemgetter(1))
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])

    # # Treat the id's as index into list and create a list of words in the ascending order of id's
    # # word with id i goes at index i of the list.
    vocabulary = list(list(zip(*sorted_vocab))[0])

    # store vocabulary

    # initialize vector values
    # (z, x, y)
    init_embd = config['std_dev'] * np.random.randn(
        len(config['word_vector_type']) + 1, len(vocab_dict), config['edim'])
    if pretrained_embeddings:
        for index_3d, stored_embedding in enumerate(pretrained_embeddings):
            # fix mappings based on pretrainied vectors
            counts = 0
            mappings = {}
            for index, entry in enumerate(vocabulary):
                if entry in stored_embedding.word_to_index:
                    vec_index = stored_embedding.word_to_index[entry]
                    mappings[vec_index] = index
                    counts += 1
                    init_embd[index_3d, index] = \
                        stored_embedding.vectors[vec_index]
            print (" Found {} words in pretrained vectors out of {}".format(
                counts, len(vocabulary)))
            stored_embedding.set_mappings(mappings)

    print("Vocabulary Size: {}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {}/{},{}".format(
        len(y_train), len(y_dev), len(y_train) + len(y_dev)))

    # build convNet graph
    config['n_words'] = len(vocab_processor.vocabulary_)
    config['sentence_len'] = x_train.shape[1]

    network = CNN(config, sess, init_embd)

    print ("number of words:{} sentence length:{}".format(
        config['n_words'], config['sentence_len']))

    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(config['learning_rate'])
    # optimizer = tf.train.AdadeltaOptimizer(1e-3)
    grads_and_vars = optimizer.compute_gradients(network.loss)
    train_op = optimizer.apply_gradients(
        grads_and_vars, global_step=global_step)
    if config['clipping_weights']:
        weight_clipping = tf.assign(network.fully_con_W, tf.clip_by_norm(
            network.fully_con_W, 3, name="CLIP"))

    fc_layer_norm = tf.norm(network.fully_con_W)

    # freeze graph ?????????????
    # tf.get_default_graph().finalize()
    # tf.getDefaultGraph().finalize()

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram(
                "{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar(
                "{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", network.loss)
    acc_summary = tf.summary.scalar("accuracy", network.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(
        train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(
        dev_summary_dir, sess.graph)

    # grad summaries
    grad_summaries_dir = os.path.join(out_dir, "summaries", "grad")
    grad_summaries_writer = tf.summary.FileWriter(
        grad_summaries_dir, sess.graph)

    # Checkpointing
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    best_models_dir = os.path.abspath(os.path.join(out_dir, "best_snaps"))
    best_models_prefix = os.path.join(best_models_dir, "model")
    # Tensorflow assumes this directory already exists so we need to create it
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # Checkpointing
    sent_dir = os.path.abspath(os.path.join(out_dir, "sent_representations"))
    # Tensorflow assumes this directory already exists so we need to create it
    if not os.path.exists(sent_dir):
        os.makedirs(sent_dir)
        os.makedirs(best_models_dir)
    saver = tf.train.Saver(tf.global_variables())

    # Write vocabulary
    vocab_processor.save(os.path.join(out_dir, "vocabulary"))

    sess.run(tf.global_variables_initializer())
    # tf.get_default_graph().finalize()

    def train_step(x_batch, y_batch):
        """
        A single training step
        """
        feed_dict = {
            network.x: x_batch,
            network.y: y_batch,
            network.dropout_prob: config["dropout_rate"]
        }

        _, step, summaries, loss, accuracy, word_embd, grad_summary = sess.run(
            [train_op, global_step, train_summary_op,
             network.loss, network.accuracy, network.word_embeddings,
             grad_summaries_merged],
            feed_dict)
        if config['clipping_weights']:
            sess.run([weight_clipping])
        cur_norm = sess.run([fc_layer_norm])

        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}, norm: {}".format(
            time_str, step, loss, accuracy, cur_norm))

        train_summary_writer.add_summary(summaries, step)
        grad_summaries_writer.add_summary(grad_summary, step)

        # steps = [1, 100, 200, 300]
        # if step in steps:
        #     for i_, sub_embd_tensor in enumerate(word_embd):
        #         # write part of the w_emb tensor for checking
        #         tensor_path_file = os.path.join(
        #             out_dir, "summaries", "tensor_step_" + str(step) + "_t_" +
        #             str(i_) + ".txt"
        #         )
        #         with open(tensor_path_file, 'w') as testing_file:
        #             a_counter = 0
        #             for row in sub_embd_tensor:
        #                 a_counter += 1
        #                 testing_file.write("{}\n".format(row))
        #                 if a_counter == 10:
        #                     break

        if step == config['save_step']:
            # extracting embeddings info
            # https://github.com/normanheckscher/mnist-tensorboard-embeddings/blob/master/mnist_t-sne.py
            # http://stackoverflow.com/questions/40849116/how-to-use-tensorboard-embedding-projector/41370610#41370610
            # Generate metadata
            metadata_path = os.path.join(
                out_dir, "summaries", 'metadata.tsv')
            # metadata = os.path.join(LOG_DIR, 'metadata.tsv')
            with open(metadata_path, 'w') as metadata_file:
                for row in vocabulary:
                    metadata_file.write('{}\n'.format(row))

            embd_tensors = []
            summary_path = os.path.join(out_dir, "summaries")
            writer = tf.summary.FileWriter(summary_path, sess.graph)
            configuration = projector.ProjectorConfig()
            for i_, sub_embd_tensor in enumerate(word_embd):
                w_var = tf.Variable(sub_embd_tensor, name='w_vars_' + str(i_))

                embd_tensors.append(w_var)

                sess.run(w_var.initializer)

                # configuration = projector.ProjectorConfig()
                # One can add multiple embeddings.
                embedding = configuration.embeddings.add()
                embedding.tensor_name = w_var.name
                # Link this tensor to its metadata file (e.g. labels).
                embedding.metadata_path = metadata_path
                # Saves a config file that TensorBoard will read during startup.
                # writer = tf.summary.FileWriter(summary_path, sess.graph)
                projector.visualize_embeddings(
                    writer, configuration)

            out = sess.run(embd_tensors)
            saver = tf.train.Saver(embd_tensors)
            saver.save(sess, os.path.join(
                out_dir, "summaries", 'embeddings_.ckpt'))

            print (len(vocabulary), len(word_embd))

    def dev_step(x_batch, y_batch, writer=None):
        """
        Evaluates model on a dev set
        """
        feed_dict = {
            network.x: x_batch,
            network.y: y_batch,
            network.dropout_prob: 1.0
        }
        step, summaries, loss, accuracy = sess.run(
            [global_step, dev_summary_op, network.loss, network.accuracy],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(
            time_str, step, loss, accuracy))
        global candidate_accuracy
        candidate_accuracy = accuracy
        if writer:
            writer.add_summary(summaries, step)

    def save_dev_summary(x_batch, y_batch, x_strings_batch, name_):
        '''
        save info for a batch in order to plot in
        bokeh later
        '''
        path_ = os.path.join(sent_dir, name_)
        y_net = []
        prob_net = []
        layer = []
        true_labels = []
        feed_dict = {
            network.x: x_batch,
            network.y: y_batch,
            network.dropout_prob: 1.0

        }
        output_ = [network.predictions, network.true_predictions,
                   network.probs, network.h_pool_flat]
        predictions, true_pred, probs, fc_layer = sess.run(
            output_, feed_dict)
        prob_net = probs.tolist()
        layer = fc_layer.tolist()
        y_net = predictions.tolist()
        true_labels = true_pred.tolist()

        process_utils.save_info(
            x_strings_batch, true_labels, y_net, prob_net, layer, path_)

    # Generate batches
    print ("About to build batches for x:{} with number of words".format(
        len(x_train), config['n_words']))
    batches = process_utils.batch_iter(
        list(zip(x_train, y_train)), config['batch_size'], config['n_epochs'])
    batches_per_epoc = int((len(x_train) - 1) / config['batch_size']) + 1

    conf_path = os.path.abspath(os.path.join(out_dir, "config.json"))
    json.dump(config, open(conf_path, 'w'), indent="\t")
    print("Saved configuration file at: {}".format(conf_path))

    print ("train loop starting for every batch")
    global candidate_accuracy
    global best_achieved_accuracy
    global epochs_best_acc_not_changed
    for batch in batches:
        x_batch, y_batch = zip(*batch)
        train_step(x_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        burn_in_period = batches_per_epoc * 0.5
        print (
            "burn_in_period {}, mod_calc {}, cand_acc {}, b_acc {}, iter {},"
            "ep_not changed {}".format(
                burn_in_period, current_step % config['evaluate_every'],
                candidate_accuracy, best_achieved_accuracy, current_step,
                epochs_best_acc_not_changed))
        if current_step % batches_per_epoc == 0:
            epochs_best_acc_not_changed += 1
        if current_step > burn_in_period and current_step % config['evaluate_every'] == 0:
            if candidate_accuracy > best_achieved_accuracy:
                best_achieved_accuracy = candidate_accuracy
                print ("---- New best vlidation accuracy acheived !! -----")
                epochs_best_acc_not_changed = 0
                saver.save(sess, best_models_prefix, global_step=current_step)
            if epochs_best_acc_not_changed > 21:
                print (
                    "early stopping, model hasn't improved for 20 epochs..."
                    "best achieved validation accuracy {}".format(
                        best_achieved_accuracy))
                break

        if current_step % config['evaluate_every'] == 0:
            print("\nEvaluation:")
            dev_step(x_dev, y_dev, writer=dev_summary_writer)
            print("")
            # evalute only every 250 steps if possible
            if current_step % 250 == 0:
                save_dev_summary(
                    x_dev, y_dev, dx_dev,
                    "metrics_step_{}.pkl".format(current_step))
            # save_dev_summary(
            #     x_train, y_train, dx_train,
            #     "metrics_train_step_{}.pkl".format(current_step))
        if current_step % config['checkpoint_every'] == 0:
            path = saver.save(
                sess, checkpoint_prefix, global_step=current_step)
            print("Saved model checkpoint to {}\n".format(path))
