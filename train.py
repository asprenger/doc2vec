# -*- coding: utf-8 -*-

"""
Train a Doc2vec model.
"""

import os
import codecs
import argparse
import logging
import shutil
import json
from random import shuffle, randint
from datetime import datetime 
from collections import namedtuple, OrderedDict
import multiprocessing
import gensim
import gensim.models.doc2vec
from gensim.models import Doc2Vec
import time

def read_lines(path):
    '''Return lines in file'''
    return [line.strip() for line in codecs.open(path, "r", "utf-8")]

def current_time_ms():
    return int(time.time()*1000.0)

def make_timestamped_dir(base_path):
    output_path = os.path.join(base_path, str(current_time_ms()) + '_' + str(randint(1000000, 9999999)))
    clean_make_dir(output_path)
    return output_path

def clean_make_dir(path):
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)

def load_stopwords(stopwords_path):
    logging.info("Loading stopwords: %s", stopwords_path)
    stopwords = read_lines(stopwords_path)
    return dict(map(lambda w: (w.lower(), ''), stopwords))

TaggedDocument = namedtuple('TaggedDocument', 'tags words')


def run(doc_path, output_base_dir, stopwords_path, vocab_min_count, num_epochs, algorithm, vector_size, alpha, min_alpha, train):
    
    # As soon as FAST_VERSION is not -1, there are compute-intensive codepaths that avoid holding 
    # the python global interpreter lock, and thus you should start to see multiple cores engaged.
    # For more details see: https://github.com/RaRe-Technologies/gensim/issues/532
    assert gensim.models.doc2vec.FAST_VERSION > -1, "This will be painfully slow otherwise"

    stopwords = load_stopwords(stopwords_path)

    cores = multiprocessing.cpu_count()

    all_docs = []

    logging.info('Loading documents: %s', doc_path)

    # TODO look at gensim.models.word2vec.LineSentence or gensim.models.word2vec.PathLineSentences

    doc_id = 0
    for i, line in enumerate(open(doc_path, encoding="utf-8")):
        line = line.strip()
        words = gensim.utils.to_unicode(line).split() # review utf-8 handling

        # remove stopwords and single letter terms
        words = [w for w in words if w not in stopwords and len(w) > 1]

        all_docs.append(TaggedDocument([doc_id], words))
        doc_id += 1
        if doc_id % 10000 == 0:
            logging.info('Loaded %s documents', doc_id)

    if algorithm == 'pv_dmc':
        # PV-DM with concatenation
        # window=5 (both sides) approximates paper's 10-word total window size
        # PV-DM w/ concatenation adds a special null token to the vocabulary: '\x00'
        model = Doc2Vec(dm=1, dm_concat=1, vector_size=vector_size, window=5, negative=5, hs=0, 
                        min_count=vocab_min_count, workers=cores)
    elif algorithm == 'pv_dma':
        # PV-DM with average
        # window=5 (both sides) approximates paper's 10-word total window size
        model = Doc2Vec(dm=1, dm_mean=1, vector_size=vector_size, window=5, negative=5, hs=0, 
                        min_count=vocab_min_count, workers=cores)
    elif algorithm == 'pv_dbow':
        # PV-DBOW 
        model = Doc2Vec(dm=0, vector_size=vector_size, negative=5, hs=0, min_count=vocab_min_count, 
                        workers=cores)
    else:
        raise ValueError('Unknown algorithm: %s' % algorithm)
        
    logging.info('Algorithm: %s' % str(model))

    logging.info('Build vocabulary')
    model.build_vocab(all_docs)
    vocab_size = len(model.wv.vocab)
    logging.info('Vocabulary size: %d', vocab_size)

    target_dir = make_timestamped_dir(output_base_dir)

    vocab_path = os.path.join(target_dir, 'vocabulary')
    logging.info('Save vocabulary to: %s', vocab_path)
    with open(vocab_path, 'w') as f:
        term_counts = [[term, value.count] for term, value in model.wv.vocab.items()]
        term_counts.sort(key=lambda x: -x[1])
        for x in term_counts:
            f.write('%s, %d\n' % (x[0], x[1]))

    if train:        

        alpha_delta = (alpha - min_alpha) / num_epochs

        logging.info('Shuffle documents')
        shuffle(all_docs)

        logging.info('Start training')
        model.train(all_docs, total_examples=len(all_docs), epochs=num_epochs, start_alpha=alpha, end_alpha=min_alpha)
        
        model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

        logging.info('Save model to: %s', target_dir)
        model.save(os.path.join(target_dir, 'doc2vec.model'))

        model_meta = {
            'doc_path': doc_path,
            'target_dir': target_dir,
            'stopwords_path': stopwords_path,
            'algorithm': algorithm,
            'vocab_min_count': vocab_min_count,
            'num_epochs': num_epochs,
            'algorithm': algorithm,
            'vector_size': vector_size,
            'alpha': alpha,
            'min_alpha': min_alpha,
            'vocab_size': vocab_size
        }

        model_meta_path = os.path.join(target_dir, 'model.meta')
        logging.info('Save model metadata to: %s', model_meta_path)
        with open(model_meta_path, 'w') as outfile:
            json.dump(model_meta, outfile)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', help='train algorithm, one of: [pv_dmc, pv_dma, pv_dbow]', default='pv_dmc')
    parser.add_argument('--vocab-min-count', help='ignores all words with total frequency lower than this.', type=int, default=10)
    parser.add_argument('--vector-size', help='word and document vector size', type=int, default=100)
    parser.add_argument('--epochs', help='number of training epochs', type=int, default=20)
    parser.add_argument('--alpha', help='initial learning rate', type=float, default=0.025)
    parser.add_argument('--min-alpha', help='learning rate will linearly drop to min_alpha as training progresses', type=float, default=0.001)
    parser.add_argument('--train', help='train model', dest='train', action='store_true')
    parser.add_argument('--no-train', help='no model training, only generate vocabulary', dest='train', action='store_false')
    parser.set_defaults(train=True)
    parser.add_argument('stopwords', help='stopwords file path')
    parser.add_argument('input_path', help='train documents input path')
    parser.add_argument('output_base_dir', help='output base directory')
    args = parser.parse_args()
    logging.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s', level=logging.INFO)
    run(doc_path=args.input_path, output_base_dir=args.output_base_dir, stopwords_path=args.stopwords, vocab_min_count=args.vocab_min_count, 
        num_epochs=args.epochs, algorithm=args.algorithm, vector_size=args.vector_size, alpha=args.alpha, min_alpha=args.min_alpha, 
        train=args.train)

if __name__ == '__main__':
    main()
