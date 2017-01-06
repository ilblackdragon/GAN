from functools import partial
import logging

import numpy as np
import scipy.misc
import pandas

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers

import model
import sequence

flags = tf.app.flags
flags.DEFINE_string('mode', 'ebgan', 'GAN mode.')
flags.DEFINE_bool('pretrain', False, 'Pre-train auto-encoder for EBGAN')
flags.DEFINE_integer('max_doc_length', 10, 
                     'Maximum document length to truncate or pad to.')
FLAGS = flags.FLAGS

def _array_to_string(arr, vocab):
   return ' '.join([vocab.reverse(x) for x in arr])


class PrintWordsHook(tf.train.SessionRunHook):

    def __init__(self, tensors, vocab, every_n_iter=1000):
        super(PrintWordsHook, self).__init__()
        self._tensors = tensors
        self._vocab = vocab
        self._every_n_iter = every_n_iter

    def begin(self):
        self._step = 0

    def before_run(self, run_context):
        if self._step % self._every_n_iter != 0:
            return None
        return tf.train.SessionRunArgs({t: t for t in self._tensors})

    def after_run(self, run_context, run_values):
        if self._step % self._every_n_iter == 0:
          for t in self._tensors:
            print('%s: %s' % (
              t, _array_to_string(run_values.results[t], self._vocab)))
        self._step += 1


def autoencoder_model(feature, target, mode, params):
  """Autoencodes sequence model."""
  vocab_size = params.get('vocab_size')
  embed_dim = params.get('embed_dim')

  tf.identity(feature[0], name='feature')
  embed_feature = sequence.embed_features(
    feature, vocab_size=vocab_size, embed_dim=embed_dim)
  output, _ = sequence.sequence_autoencoder_discriminator(
    embed_feature, length=FLAGS.max_doc_length, hidden_size=embed_dim)
  logits, predictions = sequence.outbed_generated(output)

  # Loss and training.
  loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, feature)
  loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
  train_op = layers.optimize_loss(
      loss, tf.train.get_global_step(),
      learning_rate=params['learning_rate'],
      optimizer=params.get('optimizer', 'Adam'))
  return predictions, loss, train_op


def train_autoencoder(texts, vocab, vocab_size, embed_dim):
  params = {
    'learning_rate': 0.01,
    'vocab_size': vocab_size,
    'embed_dim': embed_dim,
  }
  est = learn.SKCompat(learn.Estimator(
    model_fn=autoencoder_model, model_dir='models/ae_dbpedia', params=params))
  print_words_monitor = PrintWordsHook(
    ['Embed_1/generated_ids:0', 'feature:0'], vocab, every_n_iter=100)
  est.fit(x=texts, y=None, max_steps=3000, batch_size=32,
          monitors=[print_words_monitor])


def main(): 
  tf.logging._logger.setLevel(logging.INFO)
  
  # Load and preprocess data.
  dbpedia = learn.datasets.load_dataset('dbpedia')
  x_train = pandas.DataFrame(dbpedia.train.data)[1]
  vocab_processor = learn.preprocessing.VocabularyProcessor(FLAGS.max_doc_length)
  x_train = np.array(list(vocab_processor.fit_transform(x_train)))
  vocab_size = len(vocab_processor.vocabulary_)
  print('Total words: %d' % vocab_size)
  for idx in range(2):
    print(x_train[idx], _array_to_string(x_train[idx], vocab_processor.vocabulary_))

  # Configure.
  embed_dim = 50
  hidden_size = 50

  pretrained = None
  embeddings_trainable = True
  if FLAGS.pretrain:
    train_autoencoder(x_train, vocab_processor.vocabulary_, vocab_size,
                      hidden_size)
    pretrained = 'models/ae_dbpedia/'
    embeddings_trainable = False

  # Configure GAN.
  params = {
    'learning_rate': 0.0005,
    'z_dim': 50,
    'feature_processor': partial(sequence.embed_features, vocab_size=vocab_size,
        embed_dim=embed_dim, pretrained=pretrained,
        trainable=embeddings_trainable),
    'generated_postprocess': lambda g: sequence.outbed_generated(g)[1],
    'generator': partial(sequence.sequence_generator, length=FLAGS.max_doc_length,
        hidden_size=hidden_size),
  }
  if FLAGS.mode == 'gan':
    params.update({
      'discriminator': partial(
        sequence.sequence_discriminator, length=FLAGS.max_doc_length, hidden_size=hidden_size),
      'loss_builder': model.make_gan_loss
    })
  elif FLAGS.mode == 'ebgan':
    params.update({
      'discriminator': partial(sequence.sequence_autoencoder_discriminator, 
                               length=FLAGS.max_doc_length,
                               hidden_size=hidden_size,
                               pretrained=pretrained),
      'loss_builder': partial(model.make_ebgan_loss, epsilon=0.5)
    })
  est = learn.SKCompat(learn.Estimator(
      model_fn=model.gan_model, model_dir='models/gan_dbpedia/', params=params))

  # Setup monitors.
  print_monitor = tf.train.LoggingTensorHook(
    ['loss_discr', 'loss_generator'], every_n_iter=100)
  print_words_monitor = PrintWordsHook(
    ['Embed_1/generated_ids:0'], vocab_processor.vocabulary_, every_n_iter=100)

  # Train for a bit.
  est.fit(x=x_train, y=None, steps=100000, batch_size=32, 
          monitors=[print_monitor, print_words_monitor])

  ## TODO: Evaluate.
  output = est.predict(x=np.zeros([10, FLAGS.max_doc_length], dtype=np.int32))
  print(output)


if __name__ == "__main__":
  main()

