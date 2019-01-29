import argparse
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('../dataset', one_hot=True)

parser = argparse.ArgumentParser()
parser.add_argument('-i')
args = parser.parse_args()
model = args.i

saver = tf.train.Saver()

sess = tf.Session()

saver.restore(sess, model)

x = tf.placeholder('float', shape=[None, 784])
y_ = tf.placeholder('float', shape=[None, 10])


