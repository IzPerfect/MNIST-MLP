import tensorflow as tf
import numpy as np

def network(X, layer1, layer2, layer3, keep_prob, name = 'network'):
	with tf.variable_scope(name):
		W1 = tf.get_variable('weight1', shape = [784, layer1], initializer = tf.contrib.layers.xavier_initializer())
		b1 = tf.Variable(tf.random_normal([layer1]), name='bias1')
		L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
		L1 = tf.nn.dropout(L1, keep_prob)
		
		W2 = tf.get_variable('weight2', shape = [layer1, layer2], initializer = tf.contrib.layers.xavier_initializer())
		b2 = tf.Variable(tf.random_normal([layer2]), name='bias2')
		L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
		L2 = tf.nn.dropout(L2, keep_prob)
		
		W3 = tf.get_variable('weight3', shape = [layer2, layer3], initializer = tf.contrib.layers.xavier_initializer())
		b3 = tf.Variable(tf.random_normal([layer3]), name='bias3')
		L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)
		L3 = tf.nn.dropout(L3, keep_prob)
		
		W4 = tf.get_variable('weight4', shape = [layer3, 10], initializer = tf.contrib.layers.xavier_initializer())
		b4 = tf.Variable(tf.random_normal([10]), name='bias4')
		
		hypothesis = tf.matmul(L3, W4) + b4
		
		return hypothesis
		

		
		
		
		
		