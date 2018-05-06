# coding: utf-8

import argparse
import os
import tensorflow as tf
tf.set_random_seed(180506)
from model import *

parser = argparse.ArgumentParser()

parser.add_argument('--layer1', dest = 'layer1', default = 64, help ='nodes of layer1')
parser.add_argument('--layer2', dest = 'layer2', default = 128, help ='nodes of layer2')
parser.add_argument('--layer3', dest = 'layer3', default = 256, help ='nodes of layer3')
parser.add_argument('--epoch', dest = 'epoch', default =10, help ='decide epoch', type = int)
parser.add_argument('--batch_size', dest = 'batch_size', default = 50, help = 'decide batch_size', type = int)
parser.add_argument('--learning_rate', dest = 'learning_rate', default = 0.001, help = 'decide batch_size', type = float)
parser.add_argument('--drop_rate', dest = 'drop_rate', default = 0.7, help = 'decide to drop rate', type = float)
parser.add_argument('--disp_num', dest = 'disp_num', default = 5, help = 'How many do you want to see', type = int)

args = parser.parse_args()

# define main
def main(_):
	tfconfig = tf.ConfigProto(allow_soft_placement=True)

	with tf.Session(config=tfconfig) as sess:
		networks = Net(sess, args)
		networks.train()
		networks.test()
		networks.disp()
	


if __name__ == '__main__':
    tf.app.run()
