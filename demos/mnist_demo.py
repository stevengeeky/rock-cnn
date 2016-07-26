# Example demo of training a network on the mnist tensorflow demo.  It's super simple!
# Demo by Steven O'Riley; mnist data courtesy of TensorFlow and MNIST

# Include inclusion folder for network.py (avoid this appendage by putting network.py and loader.py in the same directory)
import sys
sys.path.append('include/')

# Where all of the network/cnn learning and evaluation magic happens
import network

# Importing the data for training use
from tensorflow.examples.tutorials.mnist import input_data

# For reference of first-time mnist users, here's the mainly contained variables and methods within mnist:
# 	mnist.train -> next_batch(batch_size) => x, y_
# 	mnist.test -> .images, .labels
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Create a new convolutional neural network
C = network.cnn()
C.device('gpu')		# Utilise the GPU (by default the network uses the CPU)

# Create an input layer vector.  The first tensor shape is that of our input, and the second is the shape we want
# to resize it to.  Note that shape matching is quite easy, just multiply the distinct values within each tensor
# together, and see if they match (28 * 28 * 1 == 784, so we can successfully reshape this tensor).
# The objective is to reshape the tensor into 4 dimensions, where the middle 2 contain the training sample data
# (it does not have to be the middle two, but is for the simplification of network.py).  Usually, the first dimension
# represents the amount of samples (None in instantiation and -1 in reshaping means an arbitrary size) and the last
# dimension represents the amount of channels or features that each individual piece of data (pixel, in this case)
# within the middle two dimensions contains.  In this case, we have 1 colour channel, ranging from 0 to 1 across each
# pixel.
# So to sum (this is the case for this example),
# Input Tensor Shape: 		[(batch size), (pixel amount)]
# Reshaped Tensor Shape: 	[(batch size), (image height), (image width), (channel amount)]
C.input_layer([None, 784], [-1, 28, 28, 1])

# Add a new hidden layer process for this network.  We will activate each weight and bias convolution with a
# rectified linear (computing max(features, 0) meaning our features will be bounded below), and compute 32 features
# for each 5x5 patch.  We will also pool each 2x2 patch across our activation, reducing each dataset dimension value
# by a factor of 2 (for a 784p = 28x28p image, 2x2 pooling will reduce it to a 14x14p = 196p image)
C.layer('relu', [5, 5], 32, (2, 2))

# Same as above, except now from each target tensor from which we have gathered 32 targets for each 5x5 patch, now
# compute from that pool 64 features across each new 5x5 patch.  Then, pool it again across each 2x2 block
C.layer('relu', [5, 5], 64, (2, 2))

# A dense layer is essentially a hidden layer, but prepares its activation to be converted into the classification
# labels within the output layer.  We must also pass in how large our new 'image' is after previous pooling, so in
# this case across each dimension of size 28, we have reduced it by 2 twice, meaning 24 / 2 / 2 = 7 across each
# dimension (granting a 7x7 activation pool) and for each one of those pixels we also have computed 64 features
# (1 channel => 32 features => 64 features).  This results in a 7 * 7 * 64 shaped 1D tensor.
# Finally, we pass in the amount of neurons to use for processing this final pool (for label conversion).
# in this case, we utilise 1024 neurons with relu activation
C.dense_layer(7 * 7 * 64, 'relu', 1024)

# And, lastly, we can specify the shape of our labels, and the activation to use for gathering them.
# We have an arbitrary batch size, so we'll also have an arbitrary label size (hence 'None'), and the amount
# of labels per each sample will be one of ten (0-9 granted by the mnist digit classifier)
# softmaxing results in a probability matrix of which all dimensions will sum to 1 (so if a one dimension has a value
# that is above .5, no other dimension will be as certain that their respective label is the correct one)
C.output_layer([None, 10], 'softmax')

# Now, from what we have specified, train the network with the given layers and activations.
# If you do not explicitly have samples and labels, like mnist (which uses mnist.train.next_batch), then
# 	that is okay, just use the property containing callable 'next_batch' as your samples and leave labels as 'None'
# I also specify that there is a .5 probability that a neuron will 'die'
# For our batch size, we will pass in 10 samples and labels each time
# I state that no limit shall be placed on the amount of training sessions the network will go through, so
# 	it will indefinitely train (and usually, if you want to leave your network running overnight, you would also
# 	save your values every iteration, specified on the latter parameter section of this method)
# Scramble samples every 100 (though for next_batch functions, this just simply logs information every 100th
# 	iteration)
# Do not transform the samples (it's actually not possible for samples that are not loaded with Pillow)
# Use cross entropy (sum(y_ * log(y)) cost function reduction) to depict whether or not the network's results are
# 	'good'	(could also use logit or squared error)
# Optimally minimize this value with Adam (could also use gradient descent)
# Set a learning rate of 10^-4 (a reasonably low value will just slow learning, but a value that is too high will
# 	prevent learning entirely (the cost function will diverge))
# Also log what time each sample set was scrambled/processed or saved, how many a second are processed (time/samples),
# 	what iteration the network training is on, what value the cost function is currently at, how matching from each
# 	new sample process is the highest argument, as well as the squared error between each actual value and
# 	predicted ones
C.train(mnist.train, None, .5, 10, None, 100, False, 'cross_entropy', 'adam', 1e-4, True, True)