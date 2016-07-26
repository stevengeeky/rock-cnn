# Matmul.py; aims to satisfy f(g(a), g(b)) by assuming
# S ~ f( (W1 * A) * (W2 * B) + b )

import tensorflow as tf
import numpy as np
import loader
import io

with tf.device("/gpu:0"):
	batch_size, features = 30, 16
	
	A = tf.placeholder(tf.float32, [None, loader.limit_size ** 2, 4])
	B = tf.placeholder(tf.float32, [None, loader.limit_size ** 2, 4])
	
	y_ = tf.placeholder(tf.float32, [None, 1])
	
	A_i = tf.reshape(A, [-1, loader.limit_size ** 2 * 4])
	B_i = tf.reshape(B, [loader.limit_size ** 2 * 4, -1])

	W1 = tf.Variable(tf.zeros([loader.limit_size ** 2 * 4, features], tf.float32), name="w1")
	W2 = tf.Variable(tf.zeros([features, loader.limit_size ** 2 * 4], tf.float32), name="w2")

	b1 = tf.Variable(tf.ones([features]), name='b1')
	b2 = tf.Variable(tf.ones([features, batch_size]), name='b2')
	b3 = tf.Variable(tf.ones([1]), name='b3')

	approx = tf.add( tf.matmul(tf.add(tf.matmul(A_i, W1), b1),
					   tf.add(tf.matmul(W2, B_i), b2)), b3 )
	
	cost = tf.reduce_mean(tf.pow(y_ - approx, 2))
	step = tf.train.AdamOptimizer(.5).minimize(cost)
	
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()
	
saver = tf.train.Saver()
#saver.restore(sess, "matmul/data.ckpt")

sxs, sys = loader.load_samples("data/samples/items")
sims = loader.get_nosofsky_similarities()
x1s, x2s, ys = [], [], []

for i in range(15):
	ax = loader.apply_transform(sxs[i])
	for j in range(i, 15):
		x1s.append( ax )
		x2s.append( loader.apply_transform(sxs[j]) )
		ys.append(sims[i, j])

ys = np.reshape(np.array(ys), [-1, 1])

r, i, index, lxs = "", -1, 0, len(x1s)

while True:
	i += 1
	if index + batch_size >= lxs:
		_x1s = x1s[index:lxs]
		_x2s = x2s[index:lxs]
		_ys = ys[index:lxs]
		index = 0
	else:
		nindex = index + batch_size
		_x1s = x1s[index:nindex]
		_x2s = x2s[index:nindex]
		_ys = ys[index:nindex]
		index = nindex
	if i % 100 == 0:
		stuff = "%d, %g" % (i, cost.eval( feed_dict={ A:_x1s, B:_x2s, y_:_ys } ))
		r += ("" if r == "" else "\n") + stuff
		print(stuff)
	if i % 10000 == 0 and i > 0:
		saver.save(sess, "matmul/data.ckpt")
		
		f = io.open("graph_matmul.csv", "w+")
		f.write(unicode(r))
		f.close()
	
	with tf.device("/gpu:0"):
		step.run( feed_dict={ A:_x1s, B:_x2s, y_:_ys } )
	