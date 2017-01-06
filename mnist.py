from functools import partial
import logging

import numpy as np
import scipy.misc

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers

import images
import model
import visualization

flags = tf.app.flags
flags.DEFINE_string('mode', 'ebgan', 'GAN mode.')
flags.DEFINE_bool('pretrain', False, 'Pre-train auto-encoder for EBGAN')
FLAGS = flags.FLAGS


def main():
  tf.logging._logger.setLevel(logging.INFO)
  
  # Load MNIST data.
  mnist_data = learn.datasets.load_dataset('mnist')

  # Select subset of images.
  mnist_class = None
  imgs = mnist_data.train.images
  if mnist_class is not None:
      imgs = np.array([x for idx, x in enumerate(mnist_data.train.images) if
        mnist_data.train.labels[idx] == mnist_class])

  if FLAGS.pretrain:
    images.train_autoencoder(imgs)

   # Configure.
  params = {
    'learning_rate': 0.0005,
    'z_dim': 100,
    'generator': partial(images.conv_generator, output_dim=28, n_filters=64),
  }
  if FLAGS.mode == 'gan':
    params.update({
      'discriminator': partial(images.conv_discriminator, hidden_size=10),
      'loss_builder': model.make_gan_loss
    })
  elif FLAGS.mode == 'ebgan':
    pretrained = None
    if FLAGS.pretrain:
        pretrained = 'models/ae_mnist'
    params.update({
      'discriminator': partial(images.linear_autoencoder_discriminator, output_dim=28,
          hidden_sizes=[50], encoding_dim=5, pretrained=pretrained),
      'loss_builder': partial(model.make_ebgan_loss, epsilon=0.05)
    })
  est = learn.SKCompat(learn.Estimator(
      model_fn=model.gan_model, model_dir='models/gan_mnist/', params=params))

  # Setup monitors.
  print_monitor = tf.train.LoggingTensorHook(['loss_discr', 'loss_generator'],
    every_n_iter=100)
  save_visual = visualization.SaveVisualizationHook('models/gan_mnist/sample.jpg')

  # Train for a bit.
  est.fit(x=imgs, y=None, steps=50000, batch_size=32, 
          monitors=[print_monitor, save_visual])


if __name__ == "__main__":
  main()

