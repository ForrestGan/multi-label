#!/usr/bin/env python
# -*- coding:utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
from collections import Counter
import os
import sys
import itertools
import numpy as np
import argparse
import logging
import mxnet as mx

classNum=15


def load_data_and_label():
    file_path = "./data_seg"
    examples =[line.decode("utf-8") for line in list(open(file_path).readlines())]
    labels = [e.split("\t")[0].split("|") for e in examples]
    data = [e.split("\t")[1].split(" ") for e in examples]
    return data, labels


def pad_sentences(sentences, padding_word="</s>"):
    seq_len = max([len(x) for x in sentences])
    padd_seq = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = seq_len - len(sentence)
        new_sentence = sentence + [padding_word]*num_padding
        padd_seq.append(new_sentence)
    return padd_seq


def build_vocab(sentences):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary, label_vocab):
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y_labels = np.zeros((len(labels), classNum))
    y = np.array([[label_vocab[l] for l in label] for label in labels])
    for index, label in enumerate(labels):
        for l in label:
            word_index = label_vocab[l]
            y_labels[index][word_index] = 1
    return [x, y_labels]


def load_data():
    sentences, labels = load_data_and_label()
    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    label_vocab, label_vocab_inv = build_vocab(labels)
    x, y = build_input_data(sentences_padded, labels, vocabulary, label_vocab)
    return [x, y, vocabulary, vocabulary_inv, label_vocab, label_vocab_inv]


def data_iter(batch_size, num_embeded):
    x, y, vocab, vocab_in, label_vocab, label_vocab_inv = load_data()
    embed_size = num_embeded
    sentence_size = x.shape[1]
    vocab_size = len(vocab)

    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))

    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    x_train, x_dev = x_shuffled[:-4000], x_shuffled[-4000:]
    y_train, y_dev = y_shuffled[:-4000], y_shuffled[-4000:]
    print('Train/Valid split: %d/%d' % (len(y_train), len(y_dev)))
    print('train shape:', x_train.shape)
    print('valid shape:', x_dev.shape)
    print('sentence max words', sentence_size)
    print('embedding size', embed_size)
    print('vocab size', vocab_size)
    train = mx.io.NDArrayIter(x_train, y_train, batch_size, shuffle=True)
    valid = mx.io.NDArrayIter(x_dev, y_dev, batch_size)

    return (train, valid, sentence_size, embed_size, vocab_size)