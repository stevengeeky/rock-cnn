# rock-cnn

&rarr; Aims to reasonably compare two rocks in accordance with that of a human using convolutional networks with tensorflow.

Rocks and human ratings granted by Robert Nosofsky.
Theory by Steven O'Riley.
See ['An Explication of Large Entities'](https://docs.google.com/document/d/1WpAlT9FFR2_7rEWqicd9v34EIlS_dnuNM8uViNFMGII/edit?usp=sharing) for explanation of _S&larr;f<sub>a&rarr;b</sub>(g(a), g(b))_

# Installation Instructions
To start testing (or training) for your own use, you'll first require a few dependencies.

For network involvement and mathematics, Install [TensorFlow](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#pip-installation) (and numpy, which is included in the tensorflow installation)

For loading resources, install the latest version of [Pillow](https://pypi.python.org/pypi/Pillow) (make sure PIL is uninstalled)

Also don't forget, whenever testing or running anything on your GPU, to `export LD_LIBRARY_PATH=/usr/local/cuda/lib64` in shell (or terminal)

Now you're ready to test and run with rock-cnn.
Simply `git clone https://github.com/stevengeeky/rock-cnn.git` and `cd` into the directory.

runme.py is utilised for training and calculating results from g(x) (image pre-processing and dimensional rating), and runme2.py is utilised for training and calculating the results from f(u, v) (similarity ratings).
