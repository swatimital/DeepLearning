from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
from lr_utils import load_dataset
import numpy as np

FLAGS = None

def next_batch(num, data, labels):
    num_samples = data.shape[0]
    idx = np.arange(0, num_samples)
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i,] for i in idx]
    labels_shuffle = [labels[i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def prediction_predicate(y_actual, y_expected, samples):
    tf.Print(y_actual, [y_actual], message="y actual")
    cond = tf.less(y_actual, 0.5)
    cond_true = tf.fill([samples,1], 0.0)
    cond_false = tf.fill([samples,1], 1.0)
    y_truncated = tf.where(cond, cond_true, cond_false)
    y_truncated = tf.Print(y_truncated,[y_truncated], message="y truncated")
    return tf.equal(y_truncated, y_expected)


def main(_):
  train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

  train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1)
  test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1)
  train_set_x = train_set_x_flatten / 255.
  test_set_x = test_set_x_flatten / 255.

  train_samples = train_set_x.shape[0]
  train_pixels = train_set_x.shape[1]
  test_samples = test_set_x.shape[0]
  label_size = 1

  # Organize the data and feed it to associated dictionaries.
  data = {}

  data['train/image'] = train_set_x
  data['train/label'] = train_set_y[0].reshape(train_samples, label_size)
  data['test/image'] = test_set_x
  data['test/label'] = test_set_y[0].reshape(test_samples, label_size)

  print("train_set_x shape: " + str(data['train/image'].shape))
  print("train_set_y shape: " + str(data['train/label'].shape))
  print("test_set_x shape: " + str(data['test/image'].shape))
  print("test_set_y shape: " + str(data['test/label'].shape))

  # Create the model
  x = tf.placeholder(tf.float32, [None, train_pixels])
  W = tf.Variable(tf.zeros([train_pixels, label_size]))
  b = tf.Variable(tf.zeros([label_size]))
  y = tf.matmul(x, W) + b
  y_ = tf.placeholder(tf.float32, [None, label_size])
  z = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=y)
  #z = y_ * -tf.log(tf.sigmoid(y)) + (1 - y_) * -tf.log(1 - tf.sigmoid(y))

  cross_entropy = tf.reduce_mean(z)
  train_step = tf.train.GradientDescentOptimizer(0.0015).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  # Train
  for _ in range(3000):
    batch_xs, batch_ys = next_batch(200, data['train/image'], data['train/label'])
    #print("{0}\n{1}\n{2}\n{3}".format(batch_xs.shape, x.get_shape(), batch_ys.shape, y_.get_shape()))
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    #print(z.eval({x: data['train/image'], y_: data['train/label']}))

  # Test trained model
  correct_prediction = prediction_predicate(y, y_, train_samples)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  train_accuracy_val = sess.run(accuracy, feed_dict={x: data['train/image'], y_: data['train/label']})
  print(y.eval({x:data['train/image'], y_:data['train/label']}))

  correct_prediction = prediction_predicate(y, y_, test_samples)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  test_accuracy_val = sess.run(accuracy, feed_dict={x: data['test/image'], y_: data['test/label']})
  print(y.eval({x: data['test/image'], y_: data['test/label']}))

  print("Train Accuracy Val: ")
  print(train_accuracy_val)

  print("Test Accuracy Val: ")
  print(test_accuracy_val)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)