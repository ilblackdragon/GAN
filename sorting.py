from functools import partial
import logging

import numpy as np
import scipy.misc

import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers

import model
import sequence

flags = tf.app.flags
flags.DEFINE_string('mode', 'ebgan', 'GAN mode.')
FLAGS = flags.FLAGS

        
def main():
  # Configure.
  vocab_size = 10
  embed_dim = 10
  length = 4
  hidden_size = 10
  params = {
    'learning_rate': 0.0005,
    'z_dim': 10,
    'feature_processor': partial(sequence.embed_features, vocab_size=vocab_size,
        embed_dim=embed_dim),
    'generated_postprocess': sequence.outbed_generated,
    'generator': partial(sequence.sequence_generator, length=length,
        hidden_size=hidden_size),
  }
  if FLAGS.mode == 'gan':
    params.update({
      'discriminator': partial(
        sequence.sequence_discriminator, length=length, hidden_size=hidden_size),
      'loss_builder': model.make_gan_loss
    })
  elif FLAGS.mode == 'ebgan':
    params.update({
      'discriminator': partial(sequence.sequence_autoencoder_discriminator, 
                               length=length, hidden_size=hidden_size),
      'loss_builder': partial(model.make_ebgan_loss, epsilon=0.05)
    })
  tf.logging._logger.setLevel(logging.INFO)
  est = learn.SKCompat(learn.Estimator(
      model_fn=model.gan_model, model_dir='models/gan_sorting/', params=params))

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

