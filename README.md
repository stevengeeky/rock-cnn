# rock-cnn

> This project utilizes TensorFlow

&rarr; This project applies a recurrent neural network (RNN) to the replication of superficial characteristics as quantified by human test subjects, and qualitatively applies these observations to each layer within the RNN before pooling is applied. After _K_ layers, the output targets are then directly applied onto each other in order to recreate a human-like comparison between the features of some original two images, _A_ and _B_.

> It is important to note that the output targets are **not** linearly transformed, meaning the study is not hindered by 'optimization through regulation.'

(Another study done on this topic was able to achieve almost unbelievable accuracy, but is virtually irreplicable due to the fact that the output targets are a linearly regressed isomorphic subspace transformed from a linearly regressed 4096x4096 target space. In other words, that particular study is illegitimate.)

For this project, a group of 80 subjects were presented with two images of rocks (of which there were 30 to choose from), and asked to rate the similarity between the two images on a scale from 1 to 10 (1 indicating that the rocks have no similarity, and 10 indicating the the rocks are absolutely the same). We decided to use rocks for comparison because the observations that subjects derive from them are less likely to be based on outside information (for example, were subjects asked to compare the image of a schoolbus with the image of a monkey, the immediate fact that a monkey is not a schoolbus would define a factor of difference which would be less based upon physical appearance alone, and more based upon the fact that the two objects are classifiably different). The overall goal of this project is then to see how well an RNN can replicate the 30 * 29 / 2 human similarity ratings between any two given distinct rocks.

All rock images and human similarity ratings were recorded and shared by Robert Nosofsky.

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
