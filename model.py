from functools import partial
import logging

import numpy as np

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers


def make_gan_loss(feature, feature_generated, discr_true, discr_generated):
  loss_discr = tf.reduce_mean(-tf.log(discr_true) - tf.log(1 - discr_generated))
  loss_generator = tf.reduce_mean(-tf.log(discr_generated))
  return loss_discr, loss_generator


def make_ebgan_loss(feature, feature_generated, discr_true, discr_generated,
                    epsilon):
  loss_discr = (tf.reduce_mean(tf.square(feature - discr_true)) +
    tf.reduce_mean(tf.nn.relu(epsilon -
    tf.reduce_mean(tf.square(feature_generated - discr_generated), 1))))
  loss_generator = tf.reduce_mean(tf.square(feature_generated -
    discr_generated))
  return loss_discr, loss_generator


def norm(x):
  return x * tf.rsqrt(tf.reduce_mean(tf.square(x), keep_dims=True))

 
def gan_model(feature, unused_target, mode, params):
  """Customizable GAN model."""
  # Retrieve params
  generator = params.get('generator')
  discriminator = params.get('discriminator')
  loss_builder = params.get('loss_builder')
  feature_processor = params.get('feature_processor', lambda f: f)
  generated_postprocess = params.get('generated_postprocess', lambda f: f)
  z_dim = params.get('z_dim')
  initial_learning_rate = params.get('learning_rate', 0.05)
  decay_steps = params.get('decay_steps', 150)
  decay_rate = params.get('decay_rate', 0.95)

  # Process features.
  feature = feature_processor(feature)

  # Create uniform [-1, 1] noise Z of shape [batch_size, z_dim].
  z = tf.random_uniform(tf.pack([tf.shape(feature)[0], z_dim]), -1, 1, dtype=feature.dtype)
  z.set_shape([feature.get_shape()[0], z_dim])

  # Generate fake example.
  feature_generated = generator(z)
  feature_generated = tf.identity(feature_generated, name='generated')

  # Discriminate true and generated example.
  discr_true, _ = discriminator(feature)
  discr_generated, e_gen = discriminator(feature_generated, reuse=True)

  if e_gen is not None:
    e_gen = norm(e_gen)
    cov = tf.matmul(tf.transpose(e_gen), e_gen) * (1.0 - tf.eye(tf.shape(e_gen)[1]))
    regularizer = tf.reduce_mean(tf.abs(cov))
  else:
    regularizer = 0.0

  # Build GAN losses.
  loss_discr, loss_generator = loss_builder(
      feature, feature_generated, discr_true, discr_generated)
  loss_generator += 0.01 * regularizer
  loss_discr = tf.identity(loss_discr, name="loss_discr")
  loss_generator = tf.identity(loss_generator, name="loss_generator")
  tf.summary.scalar("loss/discr", loss_discr)
  tf.summary.scalar("loss/generator", loss_generator)
  tf.summary.scalar("loss/total", loss_discr + loss_generator)

  # Optimize Generator and Discriminator separately.
  variables = tf.trainable_variables()
  generator_params = [v for v in variables if v.name.startswith('Generator/')]
  discriminator_params = [v for v in variables if not v.name.startswith('Generator/')]
  gc = tf.contrib.framework.get_global_step()
#  learning_rate = tf.train.exponential_decay(
#    initial_learning_rate, gc, decay_steps, decay_rate, staircase=True)
  learning_rate = initial_learning_rate
  with tf.variable_scope('Discriminator'):
    discriminator_train_op = layers.optimize_loss(
      loss_discr, gc, variables=discriminator_params,
      learning_rate=learning_rate, optimizer='Adam', summaries=[])
  with tf.variable_scope('Generator'):
    generator_train_op = layers.optimize_loss(
      loss_generator, gc, variables=generator_params,
      learning_rate=learning_rate, optimizer='Adam', summaries=[])

  predictions = generated_postprocess(feature_generated)
  return (predictions, loss_discr + loss_generator,
    tf.group(discriminator_train_op, generator_train_op))


def autoencoder_model(feature, target, mode, params):
  """Autoencodes features with given function."""
  autoencoder_fn = params.get('autoencoder_fn')
  feature_processor = params.get('feature_processor', lambda f: f)
  generated_postprocess = params.get('generated_postprocess', lambda f: f)

  # Process features.
  feature = feature_processor(feature)

  # Auto-encode.
  generated, _ = autoencoder_fn(feature)

  # Loss and training.
  loss = tf.contrib.losses.mean_squared_error(feature, generated)
  train_op = layers.optimize_loss(
      loss, tf.train.get_global_step(),
      learning_rate=params['learning_rate'],
      optimizer=params.get('optimizer', 'Adam'))
 
  # Post process generated.
  prediction = generated_postprocess(generated)
  prediction = tf.identity(prediction, name='generated')
  return prediction, loss, train_op


def make_autoencoder(autoencoder_fn, model_dir, params=None):
  if params is None:
    params = {'learning_rate': 0.0001}
  params.update({
    'autoencoder_fn': autoencoder_fn,
  })
  return learn.Estimator(
    model_fn=autoencoder_model, model_dir=model_dir, params=params)

