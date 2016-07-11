from PIL import Image
import numpy as np
import random
import io, os, sys

limit_size = 2 * 3 * 5 * 7

def check_file(filename):
    """Returns whether or not a file is considered a valid image"""
    ext = filename.split(".")[-1].lower()
    return ext == "jpg" or ext == "png" or ext == "jpeg"

def reset(filename):
    file = io.open(filename, "w+")
    file.write(unicode(""))
    file.close()

def rand_arr(r=range(100)):
    """Generates an array randomly and uniquely containing all numbers within range r"""
    temp = [x for x in r]
    res = []
    for i in r:
        index = int(rand.random() * len(temp))
        res.append(temp.pop(index))
    return res

def make_one_hot(x, y, one_hot_index=0):
    """Instantiates a one hot vector of dimensional size y scaled across dimension x"""
    val = 1 if x == -1 or x == None else x
    m = np.zeros([val, y])
    
    for i in range(val):
        m[i, one_hot_index] = 1
    return m[0] if x == -1 or x == None else m

def num_to_mod_tri(n, xy):
    """Attempts to find a unique a, b, and c such that for a number N = x * a + y * b + c, and 0 <= n < N"""
    items = np.array(xy)
    items.sort()
    a = int((n - n % items[1]) / items[1])

    sub = n % items[1]

    b = int((sub - sub % items[0]) / items[0])
    c = int(sub - b * items[0])
    return a, b, c

def load_image(fn, preserve_data=False, to_xyz=True, flip=0):
    """Loads and assembles an image with filename fn"""
    global limit_size
    img = Image.open(fn)
    pix = img.load()
    for y in range(img.height):
        for x in range(img.width):
            if pix[x, y][3] < 255:
                pix[x, y] = (0, 0, 0, 0)
    
    img = img.convert("RGB")
    
    return img

def name_nicely(fd, preface="item"):
    """Renames all images within a directory, recursively, in a nicely labeled fashion"""
    count = 0
    for root, dirs, files in os.walk(fd, topdown=False):
        nfiles = []
        for name in files:
            if check_file(name):
                newname = token() + name[name.rindex("."):]
                nfiles.append(newname)
                os.rename(os.path.join(root, name), os.path.join(root, newname))
        for name in nfiles:
            if check_file(name):
                newname = preface + " " + str(count) + ".png"
                os.rename(os.path.join(root, name), os.path.join(root, newname))
                count += 1

def load_images(fd, to_xyz=True, flip=0):
    """Loads all images recursively within directory fd"""
    res = []
    for root, dirs, files in os.walk(fd, topdown=False):
        for name in files:
            if check_file(name):
                file = os.path.join(root, name)
                image = load_image(file, False, to_xyz, flip)
                res.append(image)
    return res

def load_scrambled_images(fd):
    """Loads all images recursively within directory fd, then scrambles the order of them"""
    choose = []
    for root, dirs, files in os.walk(fd, topdown=False):
        for name in files:
            if check_file(name):
                choose.append(os.path.join(root, name))

    l = len(choose)
    for i in range(l):
        swapind = int(random.random() * l)
        temp = choose[i]
        choose[i] = choose[swapind]
        choose[swapind] = temp
    return choose

def save_sample(savefilename, imagefilename, one_hot_token):
    if not os.path.isfile(savefilename):
        reset(savefilename)
    """Appends a single sample (x_list, y_) to file with filename 'savefilename'"""
    if not check_file(imagefilename):
        return
    file = io.open(savefilename, "r")
    content = file.read()
    file.close()
    
    r = ("" if content == "" else "\n") + imagefilename + "\t" + one_hot_token
    file = io.open(savefilename, "a+")
    file.write(unicode(r))
    file.close()

def save_samples(savefilename, fd, one_hot, log=True):
    """Saves several samples within the file with the filename 'savefilename'"""
    t = token(); imfn = savefilename + "_" + t
    np.save(imfn, one_hot)
    
    for root, dirs, files in os.walk(fd):
        for name in files:
            if check_file(name):
                if log:
                    print("Writing datasets for " + os.path.join(root, name))
                save_sample(savefilename, os.path.join(root, name), imfn)

def load_samples(savefilename, log=True, amount=-1, start=0, table=None, savewhere=""):
    """Loads all samples within the file with filename 'savefilename'"""
    file = io.open(savefilename, "r")
    content = file.read()
    file.close()
    
    lines = content.split("\n")[start:]
    xs, ys = [], []

    count = 0
    table = range(len(lines)) if table == None else table
    for i in table:
        line = lines[i]
        if amount != None and amount >= 0 and count >= amount:
            break
        sp = line.split("\t")
        if log:
            print("Loading datasets for " + sp[0])

        if savewhere != "":
            im = load_image(sp[0], preserve_data=True)
            im.save(os.path.join(savewhere, "rock %d.png" % (count + 1) ))
        xs.append(load_image(sp[0]))
        ys.append(np.load(sp[1] + ".npy"))
        count += 1
    return xs, ys

trans_count = 0
def apply_transform(image, translation='CENTER', rotation=0, flip=0, to_xyz=True):
    global trans_count
    aspect = float(image.width) / float(image.height)
    
    if rotation != 0:
        image = image.rotate(rotation)
    if flip == 1 or flip == 3:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if flip == 2 or flip == 3:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    
    if image.width > image.height:
        image = image.resize((limit_size, int(limit_size / aspect)))
    elif image.width < image.height:
        image = image.resize((int(limit_size * aspect), limit_size))
    
    backing = Image.new('RGB', (limit_size, limit_size), (0, 0, 0))
    #backing.paste( im, (int((backing.width - im.width) / 2), int((backing.height - im.height) / 2), im.width, im.height) )
    
    left, top = 0, 0
    if translation == 'CENTER':
        left = int((backing.width - image.width) / 2); top = int((backing.height - image.height) / 2)
    elif translation == 'RANDOM':
        left = int(random.random() * (backing.width - image.width)); top = int(random.random() * (backing.height - image.height))
    backing.paste(image, (left, top, image.width + left, image.height + top))
    
    #if (trans_count % 10 == 0):
    #    print("(Showing)")
    #    backing.show()
    #trans_count += 1
    
    if (to_xyz):
        rgb2xyz = (
            0.412453, 0.357580, 0.180423, 0,
            0.212671, 0.715160, 0.072169, 0,
            0.019334, 0.119193, 0.950227, 0 )
        backing = backing.convert("RGB", rgb2xyz)
    
    return np.mat(backing.getdata()) / 255

def scramble_samples(xs, ys, indices=False, rand_transform=True):
    """Scrambles a sample of respective (x_list, y_)s, then returns them in a tuple"""
    global limit_size
    
    l = len(xs)
    for i in range(l):
        swapind = int(random.random() * l)
        tempx, tempy = xs[i], ys[i]
        
        xs[i] = xs[swapind]
        ys[i] = ys[swapind]
        xs[swapind] = tempx
        ys[swapind] = tempy
        
        if indices:
            temp = indices[i]
            indices[i] = indices[swapind]
            indices[swapind] = temp
    
    if (rand_transform):
        for i in range(l):
            xs[i] = apply_transform(xs[i], 'RANDOM', int(random.random() * 360), int(random.random() * 4), True)
    
    if not indices:
        return xs, ys
    return xs, ys, indices

def token(length=64):
    """Generates a random token of length 'length'"""
    s = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"
    r = ""
    for i in range(length):
        r += s[int(random.random() * len(s))]
    return r

def layer_to_image(arr):
    return Image.fromarray((np.array(arr) * 255).astype(np.uint8)).convert('RGBA')
