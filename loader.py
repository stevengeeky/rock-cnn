from PIL import Image
import numpy as np
import random
import io, os, sys

limit_size = 2 * 3 * 5 * 7

# Rocks used for testing (numbered 1-360)
vals = [6, 14, 34, 39, 59, 69, 73, 95, 102, 119, 127, 144, 151, 164, 172, 188, 201, 213, 218, 237, 250, 263, 267, 284, 300, 309, 314, 336, 348, 352]

# Correction of the hitherto numbers to clamp them within 0-359 for array index calling later
tvals = [x - 1 for x in vals]

# Resource Loading
def get_nosofsky_similarities(sparse=False):
    """If sparse is false, this returns the 30x30 similarity ratings in an [n, k] provided array (n | 0 <= n <= 29, k | 0 <= k <= 29)
        If sparse is true, this returns the 360x360 similarity ratings
        All thanks to Nosofsky for the rating list, 270 subjects for the sparse list and 80 for the 30x30"""
    if sparse:
        file = io.open("resources/sparse_comparisons.txt", "r")
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
        file = io.open("resources/nosofsky.txt", "r")
        s = file.read()
        file.close()
        
        lines = s.split("\n")
        vals = np.zeros([30, 30])
        for i in range(30):
            sp = lines[i + 2][4:].split(" ")
            for j in range(30):
                vals[i, j] = np.float(sp[j]) / 10
        return vals

def get_transformed_ratings(table=tvals):
    """Returns a modified list of specific similarity judgements (of attributes) of shape (360, 9)"""
    file = io.open("resources/transformedratings.txt", "r")
    s = file.read()
    file.close()
    
    file = io.open("resources/colmatch.txt", "r")
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
                #print(cs, l, len(l))
                first = False
            
            pieces.append(np.array(l).astype(np.float32) / 10)
    return pieces

def check_file(filename):
    """Returns whether or not a file is considered a valid image"""
    ext = filename.split(".")[-1].lower()
    return not "._" in filename and (ext == "jpg" or ext == "png" or ext == "jpeg")

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

def load_image(fn, preserve_data=False, rgba=True):
    """Loads and assembles an image with filename fn"""
    global limit_size
    img = Image.open(fn)
    pix = img.load()
    
    if len(np.shape(pix)) == 4:
        for y in range(img.height):
            for x in range(img.width):
                if pix[x, y][3] == 0:
                    pix[x, y] = (0, 0, 0, 0)
    
    if not rgba:
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

def load_images(fd, rgba=True):
    """Loads all images recursively within directory fd"""
    res = []
    for root, dirs, files in os.walk(fd, topdown=False):
        for name in files:
            if check_file(name):
                file = os.path.join(root, name)
                image = load_image(file, False, rgba)
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
    """Appends a single sample (x_list, y_) to file with filename 'savefilename'"""
    if not os.path.isfile(savefilename):
        reset(savefilename)
    if not check_file(imagefilename):
        return
    file = io.open(savefilename, "r")
    content = file.read()
    file.close()
    
    r = ("" if content == "" else "\n") + imagefilename + "\t" + one_hot_token
    file = io.open(savefilename, "a+")
    file.write(unicode(r))
    file.close()

def save_samples(savefilename, fd, one_hot, log=True, index_increase=0, amount=0):
    """Saves several samples within the file with the filename 'savefilename'"""
    if amount > 0:
        for root, dirs, files in os.walk(fd):
            if (len(files) == 1 and files[0].lower() == '.ds_store'):
                continue
            t = token(); imfn = savefilename + "_" + t
            oh = make_one_hot(None, amount, index_increase % amount)
            index_increase += 1
            np.save(imfn, oh)
            
            for name in files:
                if check_file(name):
                    if log:
                        print("Writing datasets for " + os.path.join(root, name) + " (%d)" % (index_increase - 1))
                    save_sample(savefilename, os.path.join(root, name), imfn)
                
    else:
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
        print(line)
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
def apply_transform(image, translation='CENTER', rotation=0, flip=0, to_xyz=False, pad=True, rgba=True):
    """Applies a variation of transformations on a given image, including translation, rotation, flipping, xyz-ing, and padding (as opposed to cropping)"""
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
    
    backing = Image.new('RGBA' if rgba else 'RGB', (limit_size, limit_size), (0, 0, 0, 0) if rgba else (0, 0, 0))
    
    #backing.paste( im, (int((backing.width - im.width) / 2), int((backing.height - im.height) / 2), im.width, im.height) )
    
    if pad:
        left, top = 0, 0
        if translation == 'CENTER':
            left = int((backing.width - image.width) / 2); top = int((backing.height - image.height) / 2)
        elif translation == 'RANDOM':
            left = int(random.random() * (backing.width - image.width)); top = int(random.random() * (backing.height - image.height))
        backing.paste(image, (left, top, image.width + left, image.height + top))
    else:
        left, top = 0, 0
        if translation == 'CENTER':
            mxw = max(backing.width, image.width); mnw = min(backing.width, image.width)
            mxh = max(backing.height, image.height); mnh = min(backing.height, image.height)
            
            left = int((mxw - mnw) / 2); top = int((mxh - mnh) / 2)
        elif translation == 'RANDOM':
            left = int(random.random() * (backing.width - image.width)); top = int(random.random() * (backing.height - image.height))
        
        backing = image.crop(( left, top, left + backing.width, top + backing.height ))
    
    #if (trans_count % 10 == 0):
    #    print("(Showing)")
    #    backing.show()
    #trans_count += 1
    
    if not rgba and to_xyz:
        rgb2xyz = (
            0.412453, 0.357580, 0.180423, 0,
            0.212671, 0.715160, 0.072169, 0,
            0.019334, 0.119193, 0.950227, 0 )
        backing = backing.convert("RGB", rgb2xyz)
    
    return np.mat(backing.getdata()) / 255

def scramble_samples(xs, ys, indices=False, rand_transform=True, input_images=False):
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
    
    if input_images:
        if rand_transform:
            for i in range(l):
                xs[i] = apply_transform(xs[i], 'RANDOM', int(random.random() * 360), int(random.random() * 4), True, True, True)
        else:
            for i in range(l):
                xs[i] = apply_transform(xs[i])
    
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
