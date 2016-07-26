# To compute g(x)
# Documented for purposes of open sources

# System util
from PIL import Image   # In case you don't know, PIL spans not from python image library, but from pillow.
                        # Make sure you have the latest version of Pillow installed
import random as rand   # random
import io               # io stuffs for file read/writes

# Where it gets interesting.  Loader is a utilisation for storing and loading lots of image/one-hot-set pairs.
# Though, you could just use it to load lots of scattered images within a directory quickly
import loader

# To do all of our cnn dirty work
import network as n

# And the main showpoints, just in case we want to do any custom modding with our cnn weights/biases/activations/variables
# Though, you don't technically need them for basic networks (because tensorflow and numpy work is done with network.py)
import numpy as np
import tensorflow as tf

C = n.cnn()         # Instantiate the CNN
C.device("gpu")     # Utilise the GPU (for me it's a GTX 1070)

# Pass an input layer
C.input_layer([None, loader.limit_size ** 2, 4], [-1, loader.limit_size, loader.limit_size, 4])

# Pass hidden layers
C.layer('sigmoid', [5, 5], 32, (2, 2))
C.layer('sigmoid', [5, 5], 64, (3, 3))
C.layer('sigmoid', [5, 5], 128, (5, 5))

# Pass a dense conversion layer
C.dense_layer(7 * 7 * 128, 'sigmoid', 1024)

# Pass an output layer
C.output_layer([None, 9], 'sigmoid')

def scale(a, _min, _max):
    a = np.array(a).flatten()
    mean_a = np.sum(a) / len(a)
    
    o_min = np.amin(a)
    
    o_range = np.amax(a) - o_min
    n_range = _max - _min
    
    return (a - o_min) * n_range / o_range + _min

def dot(a, b):
    a, b = a.flatten(), b.flatten()
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def pearson(a, b):
    a, b = a.flatten(), b.flatten()
    mean_a, mean_b = np.sum(a) / len(a), np.sum(b) / len(b)
    x, y = a - mean_a, b - mean_b
    return np.sum( x * y ) / np.sqrt( np.sum(x ** 2) * np.sum(y ** 2) )

# Comparing the correlation between the predicted ratings and the actual ones
def correlate():
    
    # Correlate outputs
    
    C.flush_layers("session0/training_data.ckpt")
    xs, ys = loader.load_samples("data/samples/items", log=True, table=loader.tvals)
    sims = loader.get_nosofsky_similarities()
    
    a, b, srs = [], [], []
    
    for i in range(len(xs)):
        ax = loader.apply_transform(xs[i])
        
        for j in range(i, len(xs)):
            if i < 15 and j < 15 or i == j:
                continue
            
            bx = loader.apply_transform(xs[j])
            sqm = np.square(ax - bx)
            
            ai = C.push([sqm])
            bi = sims[i, j]
            
            print("[%d, %d]" % (i + 1, j + 1))
            srs.append("%d, %d" % (i + 1, j + 1))
            
            a.append(ai); b.append(bi)
    
    a = np.reshape(a, [9, -1])
    print(np.shape(a))
    af = np.reshape(a, [-1, 9]); bf = np.array(b).flatten()
    mi, ma = np.amin(bf), np.amax(bf)
    
    r = ""
    
    for i in range(len(bf)):
        af[i] = scale(af[i], mi, ma)
        line = srs[i]
        
        for j in range(np.shape(af)[1]):
            line += ", %g" % af[i, j]
        
        line += ", %g" % bf[i]
        
        r += ("" if r == "" else "\n") + line
        print(line)
    
    f = io.open("correlations", "w+")
    f.write(unicode(r))
    f.close()
    
    for i in range(len(a)):
        cc = pearson( a[i], bf )
        print("[cc{%d}] %g" % (1 / 10 * (i + 1), cc))
    
    return
    
    # Old Comparison
    C.flush_layers("session0/training_data.ckpt")
    
    xs, ys = loader.load_samples("data/samples/items", log=True, table=loader.tvals)
    sims = loader.get_nosofsky_similarities()
    
    l = [[] for x in range(len(C.hidden_layers) + 1)]
    
    r = ""
    
    for i in range(len(xs)):
        ax = loader.apply_transform(xs[i])
        
        for j in range(i + 1, len(xs)):
            bx = loader.apply_transform(xs[j])
            
            print("[%d, %d]" % (i + 1, j + 1)),
            
            for k in range(len(C.hidden_layers)):
                layer = C.hidden_layers[k]
                l1 = layer.eval( C.apply(x=[ax], p=1.0) )
                l2 = layer.eval( C.apply(x=[bx], p=1.0) )
                
                l[k].append( dot( l1['h'], l2['h'] ) )
            
            l[len(l) - 1].append(sims[i, j])
    
    r, ratings = "", np.array(l[len(l) - 1])
    
    for i in range(len(l) - 1):
        r += ("" if r == "" else ", ") + "%d" % pearson( np.array(l[i]), ratings )
    
    f = io.open("correlations", "w+")
    f.write(unicode(r))
    f.close()
    
    print("\nCompared and written.")

# Computing to file the predicted ratings from each of the four layers and actual ones from 30 * 31 / 2 rock comparisons
def compute():
    
    C.flush_layers("session0/training_data.ckpt")
    xs, ys = loader.load_samples("data/samples/items", log=True, table=loader.tvals)
    sims = loader.get_nosofsky_similarities()
    
    a, b, r = [], [], ""
    
    for i in range(len(xs)):
        ax = loader.apply_transform(xs[i])
        
        for j in range(i + 1, len(xs)):
            print("[%d, %d]" % (i + 1, j + 1))
            
            bx = loader.apply_transform(xs[j])
            r += ("" if r == "" else "\n") + "%d, %d, %g, %g" % (i + 1, j + 1, C.push([ np.square(ax - bx) ])[0], sims[i, j])
    
    f = io.open("data_30way", "w+")
    f.write(unicode(r))
    f.close()
    
    return
    
    C.flush_layers("session0/training_data.ckpt")
    
    xs, ys = loader.load_samples("data/samples/items", log=True, table=loader.tvals)
    sims = loader.get_nosofsky_similarities()
    
    r = ""
    
    for i in range(len(xs)):
        ax = loader.apply_transform(xs[i])
        
        for j in range(i + 1, len(xs)):
            bx = loader.apply_transform(xs[j])
            
            print("[%d, %d]" % (i + 1, j + 1)),
            line = ""
            
            for layer in C.hidden_layers:
                l1 = layer.eval( C.apply(x=[ax], p=1.0) )
                l2 = layer.eval( C.apply(x=[bx], p=1.0) )
                
                line += ("" if line == "" else ", ") + str(dot( l1['h'], l2['h'] ))
            
            r += ("" if r == "" else "\n") + "%d, %d, %s, %g" % (i + 1, j + 1, line, sims[i, j])
    
    f = io.open("dots_30way", "w+")
    f.write(unicode(r))
    f.close()
    
    print("\nComputed and written.")

def train():
    global C
    
    sims = loader.get_nosofsky_similarities()
    _xs, _ys = loader.load_samples("data/samples/items", log=True, table=loader.tvals[:15])
    xs, ys = [], []
    for i in range(15):
        a = loader.apply_transform(_xs[i])
        for j in range(i, 15):
            b = loader.apply_transform(_xs[j])
            
            xs.append( np.square(a - b) )

            yl, am = [], 10; step = 1 / am
            
            for i in range(am - 1):
                yl.append(0 if sims[i, j] < step * i + step else 1)
            
            ys.append(yl)
            
    ys = np.reshape(ys, [-1, 9])
    
    C.train(xs, ys, .5, 30, None, 100, False,
            'squared_error', 'logit', 1e-3, True, "session0/training_data.ckpt",
            save_every=1000, save_where="session0/training_data.ckpt")
    
correlate()
