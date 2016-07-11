import tensorflow as tf
import numpy as np
import random as rand
import loader, copy
import time, datetime

def _scramble(xs, ys, rand_transform=True):
    return loader.scramble_samples(xs, ys, False, rand_transform)

def _conv2d(x, W):
    return tf.nn.conv2d(x, W, [1, 1, 1, 1], 'SAME')

def _mp(x, stride=(1, 1)):
    sx, sy = stride[0], stride[1]
    return tf.nn.max_pool(x, [1, sx, sy, 1], [1, sx, sy, 1], 'SAME')

def _ap(x, stride=(1, 1)):
    sx, sy = stride[0], stride[1]
    return tf.nn.avg_pool(x, [1, sx, sy, 1], [1, sx, sy, 1], 'SAME')

def _format_time(howmany=0):
    return time.strftime("%H:%M:%S") + "/" + str(howmany)

class cnn(object):
    features = []
    targets = []

    weights = []
    biases = []
    activations = []
    pools = []

    outputs = []

    _device = "/cpu:0"

    _features = []
    _targets = []
    _sess = None
    _saver = None
    _vars = []

    weight_index = 0
    bias_index = 0
    _keep_prob = 1

    x = None
    y_ = None
    _graph = None
    keep_prob = 1

    name = None

    _l_p_s = 0
    _c_p_s = 0
    _curr_s = datetime.datetime.now().second
    _b_j_0 = 0
    _b_i_o = 0

    def __init__(self, name="network"):
        self.name = name

    def device(self, device_type="cpu", log=True):
        """Assigns a device for use with this network"""
        device_type = device_type.lower()
        
        if device_type in ("cpu|gpu").split("|"):
            self._device = "/" + device_type + ":0"
            if log:
                print("%s:Active" % device_type.upper())

    def input_layer(self, shape=[1, 1, 1, 1], unflatten=True):
        """Prepares an input layer within the network"""

        if len(shape) == 2:
            self._features.append(shape[1])
            self._targets.append(1)
        elif len(shape) == 3:
            self._features.append(shape[1])
            self._targets.append(shape[2])
        else:
            self._features.append(shape[-2])
            self._targets.append(shape[-1])
        
        with tf.device(self._device):
            self.x = tf.placeholder(tf.float32, shape=shape)
            x_i = self.x

            if type(unflatten) is list:
                shape = unflatten
                x_i = tf.reshape(self.x, shape)
            elif len(shape) == 2:
                common = int(np.sqrt(shape[1]))
                shape = [-1 if shape[0] == None else shape[0], common, common, 1]
                x_i = tf.reshape(self.x, shape)
            elif unflatten:
                common = int(np.sqrt(shape[1]))
                shape = [-1 if shape[0] == None else shape[0], common, common, shape[2]]
                x_i = tf.reshape(self.x, shape)

        self._features.append(shape[-2])
        self._targets.append(shape[-1])
        
        self.features.append(self.x)
        self.features.append(x_i)

    def layer(self, activation="relu", block_size=[5, 5], target_amount=32, pool_size=(1, 1), pooling="max"):
        """Prepares a hidden layer within the network"""
        activation = activation.lower()
        
        with tf.device(self._device):
            W = self._wvar([block_size[0], block_size[1], self._targets[-1], target_amount])
            b = self._bvar([target_amount])

            inner = _conv2d(self.features[-1], W) + b
            h = tf.nn.relu(inner) if activation == "relu" else tf.nn.elu(inner) if activation == "elu" else tf.nn.sigmoid(inner) if activation == "sigmoid" else tf.nn.elu(inner)
            p = _mp(h, pool_size) if pooling == "max" else _ap(h, pool_size)
        
        self.activations.append(activation)
        self._features.append(self._targets[-1])
        self._targets.append(target_amount)
        self.weights.append(W)
        self.biases.append(b)
        self.targets.append(h)
        self.pools.append(p)
        self.features.append(p)

    def dense_layer(self, size, activation="relu", target_amount=1024):
        """Prepares a convolutional preparation layer for linking to the final layer"""
        activation = activation.lower()
        
        with tf.device(self._device):
            flat = tf.reshape(self.features[-1], [-1, size])
            
            W = self._wvar([size, target_amount])
            b = self._bvar([target_amount])
            
            inner = tf.matmul(flat, W) + b
            h = tf.nn.relu(inner) if activation == "relu" else tf.nn.softsign(inner) if activation == "softsign" else tf.nn.softplus(inner) if activation == "softplus" else tf.nn.elu(inner) if activation == "elu" else tf.nn.sigmoid(inner) if activation == "sigmoid" else tf.nn.elu(inner)
            
            self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
            h = tf.nn.dropout(h, self.keep_prob)
        
        self.activations.append(activation)
        self._features.append(self._targets[-1])
        self._targets.append(target_amount)
        self.weights.append(W)
        self.biases.append(b)
        self.targets.append(h)
        self.features.append(h)

    def output_layer(self, shape=[None, 1], activation="softmax"):
        """Prepares a final layer for the network, linking the densely connected layer to output targets"""
        with tf.device(self._device):
            self.y_ = tf.placeholder(tf.float32, shape=shape, name="y_")

            W = self._wvar([self._targets[-1], shape[1]])
            b = self._bvar([shape[1]])

            inner = tf.matmul(self.features[-1], W) + b
            y = tf.nn.tanh(inner) if activation == "tanh" else tf.nn.softsign(inner) if activation == "softsign" else tf.nn.softplus(inner) if activation == "softplus" else tf.nn.sigmoid(inner) if activation == "sigmoid" else tf.nn.elu(inner) if activation == "elu" else tf.nn.softmax(inner)
        
        self.targets.append(self.y_)
        self.outputs.append(y)

        self._sess = tf.InteractiveSession()

    def push(self, xs, process=None):
        """Evaluates the predictive output of the model for input features 'xs'"""
        y = self.outputs[-1]
        
        with tf.device(self._device):
            fd = { self.x:xs, self.keep_prob:1 }
            if process != None:
                process = process.lower()
                y = tf.argmax(y, 1) if process == 'argmax' else y
            return y.eval(feed_dict=fd)

    def train(self, xs, ys, keep_probability=1, batch_size=10, iterations=1000, scramble_every=100, rand_transform=True, cost_func="cross_entropy", optimizer="adam", learning_rate=1e-4, log=True, flush=False, save_every=-1, save_where=""):
        """Trains a network with inputs 'xs' and predefined targets 'ys'"""
        # Available: cross_entropy, logit, squared_error
        # Available: adam, gradient_descent
        
        _xs, _ys = copy.deepcopy(xs), copy.deepcopy(ys)
        cost_func = cost_func.lower() if not callable(cost_func) else cost_func
        optimizer = optimizer.lower()
        
        y = self.outputs[-1]
        grab_obo = False
        if callable( getattr(xs, "next_batch", None) ):
            grab_obo = True
            _funct = xs
        elif rand_transform:
            xs, ys = _scramble(copy.deepcopy(_xs), copy.deepcopy(_ys), True)
        
        with tf.device(self._device):
            cy = tf.clip_by_value(y, 1e-10, 1)
            cost = cost_func(self.y_, cy) if callable(cost_func) else tf.reduce_mean(-tf.reduce_sum( self.y_ * tf.log(cy), reduction_indices=[1] )) if cost_func == "cross_entropy" else -tf.reduce_sum( self.y_ * tf.log(cy) + (1 - self.y_) * tf.log(1 - cy) ) if cost_func == "logit" else tf.reduce_mean(tf.pow(self.y_ - y, 2))
            opt = tf.train.AdamOptimizer(learning_rate) if optimizer == "adam" else tf.train.GradientDescentOptimizer(learning_rate)
            step = opt.minimize(cost)

            correct = tf.equal(tf.argmax(y, 1), tf.argmax(self.y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

            div_accuracy = tf.reduce_mean(tf.pow(self.y_ - y, 2))
            
            if not grab_obo:
                dropoff = int(len(xs) / batch_size + 1)

        if flush:
            self._saver = tf.train.Saver()
            with tf.name_scope(self.name):
                self._sess.run(tf.initialize_all_variables())

            if type(flush) is str:
                self.load(flush)

        form = "{:<6s} i {:<8d} J {:<11g} m {:<8g} s {:<8g}"
        for i in range(iterations):
            now = datetime.datetime.now()
            if (now.second != self._curr_s):
                self._l_p_s = self._c_p_s
                self._c_p_s = 0
                self._curr_s = now.second
            else:
                self._c_p_s += 1

            if grab_obo:
                xs, ys = _funct.next_batch(batch_size)
                fd = { self.x:xs, self.y_:ys, self.keep_prob:keep_probability }
            else:
                val = i % dropoff
                end = min(val + batch_size, len(xs))
                fd = { self.x:xs[val:end], self.y_:ys[val:end], self.keep_prob:keep_probability }
            
            if save_every > -1 and (i + 1) % save_every == 0:
                self.save(save_where)
                if log:
                    print(form.format("[" + _format_time(self._l_p_s) + "] sav", i + 1, cost.eval(fd), accuracy.eval(fd), div_accuracy.eval(fd)))
                
            if scramble_every != None and (i + 1) % scramble_every == 0:
                if not grab_obo:
                    xs, ys = _scramble(copy.deepcopy(_xs), copy.deepcopy(_ys), rand_transform)
                
                if log:
                    print(form.format("[" + _format_time(self._l_p_s) + "] scr", i + 1, cost.eval(fd), accuracy.eval(fd), div_accuracy.eval(fd)))
            
            with tf.device(self._device):
                step.run(feed_dict=fd)

    def flush_layers(self, load_where=None):
        """Prepares network for variable interaction (i.e. training, evaluation, etc.)"""
        self._saver = tf.train.Saver()

        with tf.name_scope(self.name):
            self._sess.run(tf.initialize_all_variables())
        if load_where != None:
             self.load(load_where)

    def save(self, outfile):
        """Saves the weights and biases of the network to 'outfile'"""
        if outfile[len(outfile) - 5:].lower() != ".ckpt":
            outfile += ".ckpt"
        
        self._saver.save(self._sess, outfile)

    def load(self, file):
        """Loads the weights and biases within 'file' into the network"""
        if file[len(file) - 5:].lower() != ".ckpt":
            file += ".ckpt"
        
        self._saver.restore(self._sess, file)

    def eval(self, t, xs=None, ys=None, keep_probability=None):
        """Evaluates a variable created in tensorflow"""
        with tf.device(self._device):
            if xs == None and ys == None and keep_probability == None:
                return t.eval()
            return t.eval(feed_dict={ self.x:xs, self.y_:ys, self.keep_prob:keep_probability })

    def _bvar(self, shape, name=-1):
        if name == -1:
            self.bias_index += 1
            name = "b" + str(self.bias_index)

        initial = tf.constant(.1, shape=shape)
        v = tf.Variable(initial, name=name)

        self.biases.append(v)
        self._vars.append(v)
        
        return v

    def _wvar(self, shape, name=-1):
        if name == -1:
            self.weight_index += 1
            name = "w" + str(self.weight_index)

        initial = tf.truncated_normal(shape, stddev=.1)
        v = tf.Variable(initial, name=name)

        self.weights.append(v)
        self._vars.append(v)
        
        return v

