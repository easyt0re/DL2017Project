# this is from train image classifier

import tensorflow as tf

from tensorflow.python.ops import control_flow_ops
from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
images = mnist.train.images # Shape: 55000 x 784
labels = mnist.train.labels # Shape: 55000 x 10
images, labels = tf.train.batch([image, label], batch_size=32)
