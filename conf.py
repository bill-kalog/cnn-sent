# configuration file
config = {

    #  # # dataset to load arguments
    'emd_dat_directory': '../datasets',  # parent directory where a dataset is stored
    'sst_finegrained': True,  # used only when/if loading SST choose [5, 2] classes
    'classes_num': 5,  # number of classes


    # # # network architecture specific arguments
    'n_epochs': 30,
    'seperate_filters': False,  # choose between seperate filters per word \
    # embedding matrix or not (True, False)
    'clipping_weights': False,
    # 'kernel_sizes': [7, 7, 7, 7],
    'kernel_sizes': [3, 4, 5],
    # 'kernel_sizes': [3, 4, 5, 6, 7, 7],
    # 'kernel_sizes': [7, 7, 7, 8, 8, 8],
    # decide which embeddings to finetune (must have a value for the random
    # vector (last one) as well. specifically for the random vector can be
    # True, False or None (if want to skip completely)
    # 'train_embeddings': [False, True, True],  # True],
    # 'train_embeddings': [False, True, True, True, True, None],
    # 'train_embeddings': [False, True, True, None],
    # 'train_embeddings': [False, True, None],  # , True]
    'train_embeddings': [False, True],
    # 'train_embeddings': [True],
    'dropout_rate': 0.7,
    # 'val_split': 0.4,
    'edim': 300,  # dimension of word embeddings
    'n_words': None,  # Leave as None, dictionary size
    # 'std_dev': 0.05,
    'std_dev': 0.01,  # variance
    'sentence_len': None,  # max sentence length
    'n_filters': 100,
    'batch_size': 128,
    'l2_regularization': 3,  # weight of l2 regularizer
    'evaluate_every': 5,  # evaluate on dev set
    'checkpoint_every': 200,  # strore a checkpoint
    'num_checkpoints': 5,
    # 'paths': ['data/rt-polarity.pos', 'data/rt-polarity.neg']
    # 'pretrained_vectors': '../datasets/glove_6B/glove.6B.100d.txt'


    # # # pretrained networks arguments
    # type of each pretrained word vector to be loaded (based on
    # values available/implemented inside word_vectors classs)
    # leave these two lists empty when not using pretrained embeddings
    'word_vector_type': ['glove', 'glove'],
    # 'word_vector_type': ['fastText', 'fastText', 'glove'],
    # 'word_vector_type': ['glove', 'glove'],
    # 'word_vector_type': ['levy', 'levy'],
    # 'word_vector_type': ['fastText', 'fastText'],
    # 'word_vector_type': ["W2V", "W2V"],
    # 'word_vector_type': ['glove', 'glove', 'fastText', 'W2V', 'levy'],
    # 'word_vector_type': ["W2V"],
    'word_vector_type': ['glove'],
    # 'word_vector_type': [],  # use only random vectors

    # 'pretrained_vectors': ['../datasets/glove_42B/glove.42B.300d.txt']
    'pretrained_vectors': ['../datasets/fastText/wiki.en.vec',
                           '../datasets/fastText/wiki.en.vec',
                           '../datasets/glove_6B/glove.6B.300d.txt'],

    'pretrained_vectors': ['../datasets/glove_6B/glove.6B.300d.txt',
                           '../datasets/glove_6B/glove.6B.300d.txt'],

    # 'pretrained_vectors': ['../datasets/fastText/wiki.en.vec',
    #                        '../datasets/fastText/wiki.en.vec'],

    'pretrained_vectors': ['../datasets/glove_6B/glove.6B.100d.txt',
                           '../datasets/glove_6B/glove.6B.100d.txt'],

    # 'pretrained_vectors': [
    #     '../datasets/w2vec/GoogleNews-vectors-negative300.bin',
    #     '../datasets/w2vec/GoogleNews-vectors-negative300.bin'],

    # 'pretrained_vectors': ['../datasets/levy/bow5.words',
    #                        '../datasets/levy/bow5.words'],

    # 'pretrained_vectors': ['../datasets/glove_6B/glove.6B.300d.txt',
    #                        '../datasets/glove_6B/glove.6B.300d.txt',
    #                        '../datasets/fastText/wiki.en.vec',
    #                        '../datasets/w2vec/GoogleNews-vectors-negative300' +
    #                        '.bin',
    #                        '../datasets/levy/bow5.words'],

    'pretrained_vectors': ['../datasets/glove_6B/glove.6B.300d.txt'],
    # 'pretrained_vectors': [
    #     '../datasets/w2vec/GoogleNews-vectors-negative300.bin'],

    # 'pretrained_vectors': []  # use only random vectors

}
