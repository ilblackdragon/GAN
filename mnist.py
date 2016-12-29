from functools import partial
import logging

import numpy as np
import scipy.misc

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers

import model
import visualization


def linear_generator(x, output_dim, scope='Generator'):
  with tf.variable_scope(scope):
    return layers.linear(x, output_dim * output_dim)


def conv_generator(x, output_dim, n_filters, scope='Generator'):
  with tf.variable_scope(scope):
    s4, s2 = int(output_dim / 4), int(output_dim / 2)
    z_ = layers.linear(x, s4 * s4 * n_filters * 2)
    h0 = tf.reshape(z_, [-1, s4, s4, n_filters * 2])
    h1 = layers.convolution2d_transpose(h0, n_filters, [5, 5], stride=2)
    h1 = tf.nn.elu(h1)
    h2 = layers.convolution2d_transpose(h1, 1, [5, 5], stride=2)
    return tf.reshape(tf.nn.tanh(h2), [-1, output_dim * output_dim])


def linear_discriminator(x, hidden_size, scope='Discriminator', reuse=False):
   with tf.variable_scope(scope, reuse=reuse):
     h0 = tf.tanh(layers.linear(x, hidden_size * 2))
     h1 = tf.tanh(layers.linear(h0, hidden_size * 2))
     h2 = tf.tanh(layers.linear(h1, hidden_size * 2))
     return tf.sigmoid(layers.linear(h2, 1))


def linear_autoencoder_discriminator(x, output_dim, hidden_sizes, encoding_dim, scope='Discriminator', reuse=False):
  with tf.variable_scope(scope, reuse=reuse):
    # Encoder.
    for hsz in hidden_sizes:
        x = tf.nn.elu(layers.linear(x, hsz))
    encoding = x = layers.linear(x, encoding_dim)
    # Decoder.
    for hsz in reversed(hidden_sizes):
        x = tf.nn.elu(layers.linear(x, hsz))
    decoding = layers.linear(x, output_dim * output_dim)
    return decoding, None

        
def main():
  # Configure.
  mode = 'ebgan'
  params = {
    'learning_rate': 0.0005,
    'z_dim': 100,
    'generator': partial(conv_generator, output_dim=28, n_filters=64),
  }
  if mode == 'gan':
    params.update({
      'discriminator': partial(conv_discriminator, hidden_size=10),
      'loss_builder': model.make_gan_loss
    })
  elif mode == 'ebgan':
    params.update({
      'discriminator': partial(linear_autoencoder_discriminator, output_dim=28,
          hidden_sizes=[50], encoding_dim=5),
      'loss_builder': partial(model.make_ebgan_loss, epsilon=0.05)
    })
  tf.logging._logger.setLevel(logging.INFO)
  mnist_data = learn.datasets.load_dataset('mnist')
  est = learn.SKCompat(learn.Estimator(
      model_fn=model.gan_model, model_dir='models/gan_mnist/', params=params))

  # Select subset of images.
  mnist_class = None
  imgs = mnist_data.train.images
  if mnist_class is not None:
      imgs = np.array([x for idx, x in enumerate(mnist_data.train.images) if
        mnist_data.train.labels[idx] == mnist_class])

  # Setup monitors.
  print_monitor = tf.train.LoggingTensorHook(['loss_discr', 'loss_generator'],
    every_n_iter=100)
  save_visual = visualization.SaveVisualizationHook('models/gan_mnist/sample.jpg')

  # Train for a bit.
  est.fit(x=imgs, y=None, steps=50000, batch_size=32, 
          monitors=[print_monitor, save_visual])


if __name__ == "__main__":
  main()

