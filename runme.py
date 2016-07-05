from PIL import Image
import random as rand
import io, obsolete
import loader

import network as n
import numpy as np
import tensorflow as tf

vals = [6, 14, 34, 39, 59, 69, 73, 95, 102, 119, 127, 144, 151, 164, 172, 188, 201, 213, 218, 237, 250, 263, 267, 284, 300, 309, 314, 336, 348, 352]
tvals = [x - 1 for x in vals]

def get_nosofsky_similarities():
    file = io.open("nosofsky.txt", "r")
    s = file.read()
    file.close()
    
    lines = s.split("\n")
    vals = np.zeros([30, 30])
    for i in range(30):
        sp = lines[i + 2][4:].split(" ")
        for j in range(30):
            vals[i, j] = np.float(sp[j])
    return vals

def get_transformed_ratings(table=None):
    file = io.open("transformedratings.txt", "r")
    s = file.read()
    file.close()

    lines = s.split("\n"); pieces = []
    table = range(360) if table == None else table
    for i in table:
        line = lines[i]
        if line != "":
            ls = np.array(line.split(" ")[3:] + [0]).astype(np.float32)
            for j in range(6):
                ls[j] /= 10
            for j in range(12, len(ls)):
                ls[j] /= 10
            pieces.append(ls)
    return pieces

# Network layers
C = n.cnn()

C.device("gpu")

C.input_layer([None, loader.limit_size ** 2, 4], [-1, loader.limit_size, loader.limit_size, 4])
C.layer('softsign', [5, 5], 128, (2, 2))
C.layer('softsign', [5, 5], 256, (3, 3))
C.dense_layer(7 * 7 * 256, 'softsign', 512)
C.output_layer([None, 16], 'sigmoid')

xs, dut_ys = loader.load_samples("data/samples/items", log=True, table=tvals)
ys = []

nsky = get_nosofsky_similarities()
rats = get_transformed_ratings(table=tvals)

for i in range(30):
    ys.append( np.array(rats[i]) )

print("==> Loaded.")

#C.flush_layers()

C.train(xs, ys, .5, 30, 10000000, 100,
    "squared_error",
    'adam', 1e-6, True, "session0/training_data.ckpt",
    save_every=10000, save_where="session0/training_data.ckpt")

#print("\nSet 2 (Half Set) --> ")
#network.train(rxs, rys, .5, 30, 10000, 30, 'cross_entropy', 'adam', 1e-4, True, "session, save_every=50, save_where="session/half_training_data.ckpt")

#network.save("session/training_data")
