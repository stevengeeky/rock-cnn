# Matmul.py; aims to satisfy f(g(a), g(b)) by assuming

import tensorflow as tf
import numpy as np
import loader
import io, random, time

load_from_file = True
just_predict = False

batch_size = 10
regularization = 7e-6

weight_index, bias_index = 0, 0

def rem(am):
	print '\r{}'.format(" " * am),

def perc(rat):
	rem(4)
	acc = 1
	print ('\r%g%%' % (float(int(rat * pow(10, 2 + acc))) / pow(10, 2))),

def conv2d(x, W):
	return tf.nn.conv2d(x, W, [1, 1, 1, 1], 'SAME')

def pool(x, sz=(1, 1)):
	return tf.nn.max_pool(x, [1, sz[0], sz[1], 1], [1, sz[0], sz[1], 1], 'SAME')

def scale(a, b):
	a, b = np.array(a).flatten(), np.array(b).flatten()
	
	amin, bmin = np.amin(a), np.amin(b)
	amax, bmax = np.amax(a), np.amax(b)
	
	arange, brange = amax - amin, bmax - bmin
	
	return (a - amin) / arange * brange + bmin

def wvar(shape):
	global weight_index
	n = "w%d" % weight_index
	weight_index += 1
	
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial, name=n)

def _scramble(*items):
	l, il = len(items[0]), len(items)
	
	for i in range(l):
		ind = int(random.random() * l)
		
		for j in range(il):
			temp = items[j][ind]
			items[j][i] = items[j][ind]
			items[j][ind] = temp
	
	return items

def bvar(shape):
	global bias_index
	n = "w%d" % bias_index
	bias_index += 1
	
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial, name=n)

def pearson(a, b):
    a, b = a.flatten(), b.flatten()
    mean_a, mean_b = np.sum(a) / len(a), np.sum(b) / len(b)
    x, y = a - mean_a, b - mean_b
    return np.sum( x * y ) / np.sqrt( np.sum(x ** 2) * np.sum(y ** 2) )

def sq_err(a, b):
	ov = np.power( a.flatten() - b.flatten(), 2 )
	return np.sum(ov) / len(ov)

def add(*vals):
	res = tf.reduce_sum(tf.square(vals[0]))
	for i in range(len(vals) - 1):
		res = tf.add(res, tf.reduce_sum(tf.square(vals[i + 1])))
	return res

with tf.device("/gpu:0"):
	A = tf.placeholder(tf.float32, [None, loader.limit_size ** 2, 4])
	B = tf.placeholder(tf.float32, [None, loader.limit_size ** 2, 4])
	
	inp_a = tf.reshape(A, [-1, loader.limit_size, loader.limit_size, 4])
	inp_b = tf.reshape(B, [-1, loader.limit_size, loader.limit_size, 4])
	
	y_ = tf.placeholder(tf.float32, [None, 1])
	
	features = [32, 64, 128, 1024, 16]
	
	# Process 1
	Wa0 = wvar([5, 5, 4, features[0]])
	Wb0 = wvar([5, 5, 4, features[0]])
	
	ba0 = bvar([features[0]])
	bb0 = bvar([features[0]])
	
	ha0 = tf.nn.sigmoid(conv2d(inp_a, Wa0) + ba0 + conv2d(inp_b, Wa0) + ba0)
	hb0 = tf.nn.sigmoid(conv2d(inp_b, Wb0) + bb0 + conv2d(inp_a, Wb0) + bb0)
	
	pa0 = pool(ha0, (2, 2))
	pb0 = pool(hb0, (2, 2))
	
	# Process 2
	Wa1 = wvar([5, 5, features[0], features[1]])
	Wb1 = wvar([5, 5, features[0], features[1]])
	
	ba1 = bvar([features[1]])
	bb1 = bvar([features[1]])
	
	ha1 = tf.nn.sigmoid(conv2d(pa0, Wa1) + ba1 + conv2d(pb0, Wa1) + ba1)
	hb1 = tf.nn.sigmoid(conv2d(pb0, Wb1) + bb1 + conv2d(pa0, Wb1) + bb1)
	
	pa1 = pool(ha1, (3, 3))
	pb1 = pool(hb1, (3, 3))
	
	# Process 3
	Wa2 = wvar([5, 5, features[1], features[2]])
	Wb2 = wvar([5, 5, features[1], features[2]])
	
	ba2 = bvar([features[2]])
	bb2 = bvar([features[2]])
	
	ha2 = tf.nn.sigmoid(conv2d(pa1, Wa2) + ba2 + conv2d(pb1, Wa2) + ba2)
	hb2 = tf.nn.sigmoid(conv2d(pb1, Wb2) + bb2 + conv2d(pa1, Wb2) + bb2)
	
	pa2 = pool(ha2, (5, 5))
	pb2 = pool(hb2, (5, 5))
	
	# Process 4 (dense)
	pa2f = tf.reshape(pa2, [-1, 7 * 7 * features[2]])
	pb2f = tf.reshape(pb2, [-1, 7 * 7 * features[2]])
	
	Wa3 = wvar([7 * 7 * features[2], features[3]])
	Wb3 = wvar([7 * 7 * features[2], features[3]])
	
	ba3 = bvar([features[3]])
	bb3 = bvar([features[3]])
	
	ha3 = tf.nn.sigmoid(tf.matmul(pa2f, Wa3) + ba3 + tf.matmul(pb2f, Wa3) + ba3)
	hb3 = tf.nn.sigmoid(tf.matmul(pb2f, Wb3) + bb3 + tf.matmul(pa2f, Wb3) + bb3)
	
	keep_prob = tf.placeholder(tf.float32)
	dropa = tf.nn.dropout(ha3, keep_prob)
	dropb = tf.nn.dropout(hb3, keep_prob)
	
	# Process 5 (dense conversion)
	Wa4 = wvar([features[3], features[4]])
	Wb4 = wvar([features[3], features[4]])
	
	ba4 = bvar([features[4]])
	bb4 = bvar([features[4]])
	
	fa = tf.nn.sigmoid(tf.matmul(dropa, Wa4) + ba4 + tf.matmul(dropb, Wa4) + ba4)
	fb = tf.nn.sigmoid(tf.matmul(dropb, Wb4) + bb4 + tf.matmul(dropa, Wb4) + bb4)
	
	y = tf.reduce_mean( tf.pow( fa - fb, 2 ), 1 )
	
	# (inp_a * W_a + b_a) * (inp_b * W_b + b_b) + b
	cost = tf.reduce_mean( tf.pow( y_ - tf.clip_by_value(y, 1e-10, 1.0), 2 ) )
	regularizer = tf.mul( add( tf.div(add( Wa0, Wb0 ), 50 * 4 * features[0]), tf.div(add(Wa1, Wb1), 50 * features[0] * features[1]), tf.div(add(Wa2, Wb2), 50 * features[1] * features[2]), tf.div(add(Wa3, Wb3), 2 * 49 * features[2] * features[3]), tf.div(add(Wa4, Wb4), 2 * features[3] * features[4]) ), regularization )
	cost = tf.add(cost, regularizer)
	
	step = tf.train.GradientDescentOptimizer(.5).minimize(cost)
	 
	sess = tf.InteractiveSession()
	tf.initialize_all_variables().run()

saver = tf.train.Saver()

if load_from_file:
	saver.restore(sess, "matmul/data.ckpt")

sxs, sys = loader.load_samples("data/samples/items")
sims = loader.get_nosofsky_similarities()
x1s, x2s, ys = [], [], []

if just_predict:
	for i in range(30):
		ax = loader.apply_transform(sxs[i])
		for j in range(i + 1, 30):
			if i < 15 and j < 15:
				continue
			x1s.append( ax )
			x2s.append( loader.apply_transform(sxs[j]) )
			ys.append( sims[i, j] )
else:
	for i in range(15):
		ax = loader.apply_transform(sxs[i])
		for j in range(i, 15):
			x1s.append( ax )
			x2s.append( loader.apply_transform(sxs[j]) )
			ys.append( sims[i, j] )

ys = np.reshape(np.array(ys), [-1, 1])

r, i, index, lxs = "", -1, 0, len(x1s)

if just_predict:
	predicted_ys = []
	print("Predicting...")
	
	for i in range(lxs):
		app = y.eval(feed_dict={ A:[x1s[i]], B:[x2s[i]], keep_prob:1.0 })[0]
		predicted_ys.append( app )
	
	print("Rescaling results...")
	predicted_ys = scale(predicted_ys, ys)
	
	for i in range(len(ys)):
		print("%g, %g" % (ys[i], predicted_ys[i]))
	
	predicted_ys, ys = np.array(predicted_ys), np.array(ys)
	print("Pearson: %g\nSquared Error: %g" % (pearson(predicted_ys, ys), sq_err(predicted_ys, ys)) )

while True:
	if just_predict:
		break		# Don't train
	
	if i == -1:
		print("Starting training...")
	
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
		#print(y_resh.eval(feed_dict={ y_:_ys }))
		x1s, x2s, ys = _scramble(x1s, x2s, ys)
		
		stuff = "[%d] -> %g" % (i, cost.eval( feed_dict={ A:_x1s, B:_x2s, y_:_ys, keep_prob:1.0 } ))
		#r += ("" if r == "" else "\n") + stuff
		print(stuff)
	
	if i % 1000 == 0 and i > 0:
		saver.save(sess, "matmul/data.ckpt")
	
	#	f = io.open("graph_matmul.csv", "w+")
	#	f.write(unicode(r))
	#	f.close()
	
	with tf.device("/gpu:0"):
		step.run( feed_dict={ A:_x1s, B:_x2s, y_:_ys, keep_prob:0.5 } )
	