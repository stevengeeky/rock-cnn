# To compute g(x)

from PIL import Image
import random as rand
import io, obsolete
import loader

import network as n
import numpy as np
import tensorflow as tf

vals = [6, 14, 34, 39, 59, 69, 73, 95, 102, 119, 127, 144, 151, 164, 172, 188, 201, 213, 218, 237, 250, 263, 267, 284, 300, 309, 314, 336, 348, 352]
tvals = [x - 1 for x in vals]

def get_nosofsky_similarities(sparse=False):
    if sparse:
        file = io.open("sparse_comparisons.txt", "r")
        s = file.read()
        file.close()

        lines = s.split("\n")
        vals = np.zeros([360, 360])
        for i in range(360):
            sp = lines[i].split(",")
            for j in range(360):
                vals[i, j] = np.float(sp[j]) / 10
        return vals
    else:
        file = io.open("nosofsky.txt", "r")
        s = file.read()
        file.close()
        
        lines = s.split("\n")
        vals = np.zeros([30, 30])
        for i in range(30):
            sp = lines[i + 2][4:].split(" ")
            for j in range(30):
                vals[i, j] = np.float(sp[j]) / 10
        return vals

def get_transformed_ratings(table=None):
    file = io.open("transformedratings.txt", "r")
    s = file.read()
    file.close()
    
    file = io.open("colmatch.txt", "r")
    c = file.read()
    file.close()
    
    lines = s.split("\n"); pieces = []
    clines = c.split("\n")
    
    first = True
    table = range(360) if table == None else table
    for i in table:
        line = lines[i]
        cline = clines[i]
        if line != "":
            cs = cline.split(" ")[3:]
            ls = line.split(" ")[3:]
            l = ls[:6] + cs[:2] + [0]
            if first:
                print(cs, l, len(l))
                first = False
            
            pieces.append(np.array(l).astype(np.float32) / 10)
    return pieces

# Network layers
C = n.cnn()

C.device("gpu")

C.input_layer([None, loader.limit_size ** 2, 4], [-1, loader.limit_size, loader.limit_size, 4])
C.layer('tanh', [5, 5], 32, (2, 2))
C.layer('tanh', [5, 5], 64, (3, 3))
C.layer('tanh', [5, 5], 128, (5, 5))
C.dense_layer(7 * 7 * 128, 'tanh', 256)
C.output_layer([None, 9], 'sigmoid')

def compare():
    nsky = get_nosofsky_similarities()
    rats = get_nosofsky_similarities(sparse=True)
    r = ""
    
    for i in range(30):
        for j in range(30):
            r_val = rats[tvals[i], tvals[j]]
            r += ("" if r == "" else "\n") + ("%d, %d, %g, %g" % (i + 1, j + 1, nsky[i, j], r_val))
    
    file = io.open("sparsity_displacement", "w+")
    file.write(unicode(r))
    file.close()
    
    print("Wrote")

def save():
    global C
    C.flush_layers("session0/training_data.ckpt")

    table = tvals
    xs, dut_ys = loader.load_samples("data/samples/items", log=True, table=table)
    rats = get_transformed_ratings(table=table)
    
    print("==> Loaded.")
    
    prop_xs = []
    for i in range(len(table)):
        result = C.push( [xs[i]] )
        l = []
        for j in range(len(result[0])):
            l.append( result[0, j] )
        
        if (i == 0):
            print(result[0])
        prop_xs.append(l)

    np.save("prop_xs", np.array(prop_xs))

def train():
    global C
    
    xs, dut_ys = loader.load_samples("data/samples/items", log=True, table=tvals)
    ys = []
    
    rats = get_transformed_ratings(table=tvals)
    
    for i in range(30):
        ys.append( np.array(rats[i]) )

    C.train(xs, ys, .5, 30, 10000000, 500, 'squared_error', 'adam', 1e-5, True, True, save_every=1000, save_where="session0/training_data.ckpt")

train()

#print("\nSet 2 (Half Set) --> ")
#network.train(rxs, rys, .5, 30, 10000, 30, 'cross_entropy', 'adam', 1e-4, True, "session, save_every=50, save_where="session/half_training_data.ckpt")

#network.save("session/training_data")
