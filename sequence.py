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
    outputs, _ = tf.nn.seq2seq.rnn_decoder(decoder_inputs, x, cell)
#        loop_function=lambda prev, i: prev)
    outputs = tf.stack(outputs, 1)
    return outputs


def sequence_autoencoder_discriminator(x, length, hidden_size, scope='Discriminator', reuse=False):
  with tf.variable_scope(scope, reuse=reuse):
    x = tf.unstack(x, length, 1)
    cell = tf.nn.rnn_cell.GRUCell(hidden_size)
    go = tf.zeros(tf.shape(x[0]))
    outputs, _ = tf.nn.seq2seq.basic_rnn_seq2seq(x, [go] + x, cell)
    outputs = tf.stack(outputs[:-1], 1)
    return outputs, None


def embed_features(feature, vocab_size, embed_dim, scope='Embed', reuse=False):
  with tf.variable_scope(scope, reuse=reuse):
    return layers.embed_sequence(
        feature, vocab_size=vocab_size, embed_dim=embed_dim)


def outbed_generated(generated_feature, scope='Embed'):
  with tf.variable_scope(scope, reuse=True):
    embed = tf.get_variable('EmbedSequence/embeddings')
    outbed = tf.expand_dims(tf.transpose(embed), [0])
    logits = tf.matmul(generated_feature, tf.tile(outbed,
        [tf.shape(generated_feature)[0], 1, 1]))
    generated_ids =  tf.argmax(logits, axis=2)
    tf.identity(generated_ids[0], name='generated_ids')
    return generated_ids

        
def main():
  # Configure.
  mode = 'ebgan'
  vocab_size = 10
  embed_dim = 10
  length = 4
  hidden_size = 10
  params = {
    'learning_rate': 0.0005,
    'z_dim': 10,
    'feature_processor': partial(embed_features, vocab_size=vocab_size,
        embed_dim=embed_dim),
    'generated_postprocess': outbed_generated,
    'generator': partial(sequence_generator, length=length,
        hidden_size=hidden_size),
  }
  if mode == 'gan':
    params.update({
#      'discriminator': partial(sequence_discriminator),
      'loss_builder': model.make_gan_loss
    })
  elif mode == 'ebgan':
    params.update({
      'discriminator': partial(sequence_autoencoder_discriminator, 
                               length=length, hidden_size=hidden_size),
      'loss_builder': partial(model.make_ebgan_loss, epsilon=0.05)
    })
  tf.logging._logger.setLevel(logging.INFO)
  est = learn.SKCompat(learn.Estimator(
      model_fn=model.gan_model, model_dir='models/gan_sequence/', params=params))

  # Generate data.
  data = np.random.randint(0, vocab_size, (1000, length))
  data.sort()
  print([data[idx, :] for idx in range(5)])

  # Setup monitors.
  print_monitor = tf.train.LoggingTensorHook(['loss_discr', 'loss_generator',
    'Embed_1/generated_ids'], every_n_iter=100)

  # Train for a bit.
  est.fit(x=data, y=None, steps=10000, batch_size=32, 
          monitors=[print_monitor])

  ## Evaluate.
  output = est.predict(x=np.zeros([1000, length], dtype=np.int32))

  # Compute accuracy.
  actual = output.copy()
  actual.sort()
  print('\n'.join([str(output[idx, :]) for idx in range(10)]))
  print("Accuracy: %f" % (float(np.sum(np.all(output == actual, 1))) / len(output)))


if __name__ == "__main__":
  main()

