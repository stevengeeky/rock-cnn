# rock-cnn

&rarr; Aims to reasonably compare two rocks in accordance with that of a human using convolutional neural networks.

Rocks and human ratings granted by Robert Nosofsky.
Theory by Steven O'Riley.
See ['An Explication of Large Entities'](https://docs.google.com/document/d/1WpAlT9FFR2_7rEWqicd9v34EIlS_dnuNM8uViNFMGII/edit?usp=sharing) for explanation of _S&larr;f<sub>a&rarr;b</sub>(g(a), g(b))_

# Installation Instructions
To start testing (or training) for your own use, you'll first require a few dependencies.  (And, for developmental notice, I am using Python 2.7.11)

For network involvement and mathematics, install [TensorFlow](https://www.tensorflow.org/versions/r0.9/get_started/os_setup.html#pip-installation) (and numpy, which is included in the tensorflow installation)

For loading resources, install the latest version of [Pillow](https://pypi.python.org/pypi/Pillow) (make sure PIL is uninstalled)
Also, for new Ubuntu installs, don't forget to install libjpeg (`sudo apt-get install libjpeg8-dev`) and zlib (`sudo apt-get install zlib1g-dev`)

Also don't forget, whenever testing or running anything on your GPU, to `export LD_LIBRARY_PATH=/usr/local/cuda/lib64` in shell (or terminal)

Now you're ready to test and run with rock-cnn.
Simply `git clone https://github.com/stevengeeky/rock-cnn.git` and `cd` into the directory.

The project is being revised, but thus far training for g(x) has proven to be hindered on the fact that finding a smooth representation of the dataset is quite difficult.

*A Python/TensorFlow Project by Steven O'Riley*
