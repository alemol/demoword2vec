# -*- coding: utf-8 -*-
#
# Creates a Word2vec model for mexican spanish
#
# Created by Alex Molina
# July 2020
#
# UPDATE: Adapted on October 2022 
#    - to migrate from Gensim 3.8.0 to Gensim 4.0.0
#    - minimal version for demo
#

import gensim
from gensim.models import KeyedVectors
from utils import text_generator
import re


DATADIR = 'data/'
VECSIZE = 80 # Dimensionality of the feature vectors
WINDOW = 5  # The maximum distance between the current and predicted word within a sentence.
MINFREC = 3  # Ignores all words with total frequency lower than this.
NEPOCHS = 17 # Number of iterations (epochs) over the corpus
ALSOWORDS = 1  # If set to 1 trains word-vectors (in skip-gram fashion) simultaneous with DBOW doc-vector training; If 0, only trains doc-vectors (faster).
NTHREADS = 3 #  Use these many worker threads to train the model (=faster training with multicore machines).


def save4tf(embeddings_path, path_vectors, path_words):
    words = []
    vectors = []
    with open(embeddings_path) as f:
        l = f.readlines()
        l = l[1:]
        for line in l:
            #items = re.split(r'[^\s]+(.+)', line)
            m = re.search(r'(?P<words>[^ ]+) (?P<vec>.+)\n', line)
            #print(m.group('words'),)
            words.append(m.group('words')+'\n')
            vectors.append((m.group('vec')).replace(' ', '\t')+'\n')
    with open(path_vectors, 'w') as v:
        v.writelines(vectors)
    with open(path_words, 'w') as w:
        w.writelines(words)

if __name__ == '__main__':
    # efficent training input text reader
    print('Reading training data...')
    data_root_dir = DATADIR
    train_corpus = [s for s in text_generator(data_root_dir, tok_and_tag=True)]
    print('Reading training data...DONE')
    print(type(train_corpus), type(train_corpus[1]), train_corpus[1])
    # Described in "Distributed Representations of Sentences and Documents" 
    # Theory   https://arxiv.org/abs/1405.4053v2
    # Practice https://radimrehurek.com/gensim/models/doc2vec.html
    model = gensim.models.doc2vec.Doc2Vec(vector_size=VECSIZE,
        window=WINDOW,
        min_count=MINFREC,
        epochs=NEPOCHS,
        workers=NTHREADS,
        dbow_words=ALSOWORDS)
    print('Building vocabulary...')
    # Can be a dictionary
    model.build_vocab(train_corpus)
    print('Building vocabulary...DONE')
    vocab_words = list(model.wv.index_to_key)
    print('Model Vocab size', len(vocab_words))
    # Update the model's neural weights.
    print('Training the doc2vec model...')
    model.train(
        corpus_iterable=train_corpus,
        # .corpus_count ONLY if documents is the same corpus that was provided to build_vocab
        total_examples=model.corpus_count,
        epochs=model.epochs)
    print('Training the doc2vec model...DONE')
    # Using the model to generate document embeddings
    text = 'Hola Karina este es el texto de prueba'
    print('text:\n', text)
    unseen_tokens = gensim.utils.simple_preprocess(text)
    vector = model.infer_vector(unseen_tokens)
    print('Semantic encoded text:\n', vector)
    w = 'Karina'
    print('palabra:\n', w)
    try:
        vector = model[w]    
    except KeyError as e:
        print('The word', w, 'is OOV')
        vector = None
    print('Semantic encoded word:\n', vector)
    print(type(vector))
    # TODO Saving the model into a file
    modelfname="doc2vec_model_10k_mexnews_2017-2019"
    model.save(modelfname)
    # Saving embeddings
    embeddings_path="embeddings_10k_mexnews_2017-2019"
    print('Saving the model to', embeddings_path)
    model.wv.save_word2vec_format(embeddings_path)
    print('Model Saved', embeddings_path)
    # TF tsv format for projector
    save4tf(embeddings_path,
        'vectors10k_mexnews.tsv',
        'words10k_mexnews.tsv')
    # Loading model from a file
    #print('Loading the model from', modelpath)
    #wv_from_text = KeyedVectors.load_word2vec_format(modelpath, binary=False)
    #print('Model Loaded from', modelpath)
