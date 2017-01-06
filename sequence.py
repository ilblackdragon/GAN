from functools import partial
import logging

import numpy as np
import scipy.misc

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers

import model


def sequence_generator(x, length, hidden_size, scope='Generator', reuse=False):
  with tf.variable_scope(scope, reuse=reuse):
    decoder_inputs = [x for _ in range(length)] # tf.tile(tf.expand_dims(x, 1), [1, 10, 1])
    cell = tf.nn.rnn_cell.GRUCell(hidden_size)
    outputs, _ = tf.nn.seq2seq.rnn_decoder(decoder_inputs, x, cell,
        loop_function=lambda prev, i: prev)
    outputs = tf.stack(outputs, 1)
    return outputs


def sequence_discriminator(x, length, hidden_size, scope='Discriminator', reuse=False):
  with tf.variable_scope(scope, reuse=reuse):
    x = tf.unstack(x, length, 1)
    cell = tf.nn.rnn_cell.GRUCell(hidden_size)
    _, encoding = tf.nn.rnn(x, cell)
    return layers.linear(encoding, 1), None


def sequence_autoencoder_discriminator(x, length, hidden_size,
    scope='Discriminator', reuse=False, pretrained=None):
  with tf.variable_scope(scope, reuse=reuse):
    x = tf.unstack(x, length, 1)
    cell = tf.nn.rnn_cell.GRUCell(hidden_size)
    go = tf.zeros(tf.shape(x[0]))
    outputs, _ = tf.nn.seq2seq.tied_rnn_seq2seq(
        x, [go] + x, cell, loop_function=lambda prev, i: prev)
    outputs = tf.stack(outputs[:-1], 1)
    if pretrained is not None:
      tf.contrib.framework.init_from_checkpoint(
        pretrained, {scope + '/': scope + '/'})
    return outputs, None


def embed_features(
    feature, vocab_size, embed_dim, scope='Embed', reuse=False,
    pretrained=None, trainable=True):
  with tf.variable_scope(scope, reuse=reuse):
    embeded = layers.embed_sequence(
        feature, vocab_size=vocab_size, embed_dim=embed_dim,
        trainable=trainable)
    if pretrained is not None:
      tf.contrib.framework.init_from_checkpoint(
        pretrained, {scope + '/': scope + '/'})
    return embeded


def outbed_generated(generated_feature, scope='Embed'):
  with tf.variable_scope(scope, reuse=True):
    embed = tf.get_variable('EmbedSequence/embeddings')
    outbed = tf.expand_dims(tf.transpose(embed), [0])
    logits = tf.matmul(generated_feature, tf.tile(outbed,
        [tf.shape(generated_feature)[0], 1, 1]))
    generated_ids =  tf.argmax(logits, axis=2)
    tf.identity(generated_ids[0], name='generated_ids')
    return logits, generated_ids

