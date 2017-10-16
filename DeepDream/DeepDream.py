import numpy as np
import tensorflow as tf
import urllib.request
from functools import partial
import matplotlib.pyplot as plt
import PIL.Image
import os
import zipfile


def main():

    # STEP 1
    # Download the Google's pre-trained neural network
    url = 'http://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip'

    # create a data directory that we'll extract it to
    data_dir = '../data/'

    # then use os module to retrieve the model name
    model_name = os.path.split(url)[-1]

    # then create a local zip file path
    local_zip_file = os.path.join(data_dir, model_name)

    # if there is nothing at that path we can download it(zip file of NN) using urllib module
    if not os.path.exists(local_zip_file):
        # Download
        model_url = urllib.request.urlopen(url)
        with open(local_zip_file, 'wb') as output:
            output.write(model_url.read())  # write the downloaded data to it

        # then we will extract that using ZipFile module
        with zipfile.ZipFile(local_zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

    # start with a gray image with a little noise
    img_noise = np.random.uniform(size=(224, 224, 3)) + 100.0

    # Step 2

    # Now we can Create our TensorFlow Session

    # Load our inception graph into model_fn
    model_fn = 'tensorflow_inception_graph.pb'

    # Initialise graph using tensorflow function called Graph()
    graph = tf.Graph()

    # Now initialise session using that graph
    sess = tf.InteractiveSession(graph=graph)
    # We will open our save-in-session graph using the FastGFile() function
    with tf.gfile.FastGFile(os.path.join(data_dir, model_fn), 'rb') as f:

        # Once we've opened it, we can read that Graph and parse it accordingly
        # using ParseFromString() method of Tensorflow Graph Definition module
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # We need to define our input, So we will create input tensor using the placeholder() method called input
    # with the size of 32 bits
    t_input = tf.placeholder(np.float32, name='input')  # define input tensor
    imagenet_mean = 117.0  # define imagenet mean value of pixels in an image as 117

    # By removing the above from our image it will help us with feature learning
    # So we'll subtract it from our input tensor and store the value in our preprocesses variable
    t_preprocessed = tf.expand_dims(t_input-imagenet_mean,0)

    # Then we'll load graph_def variable we initialized as the newly processed tensor
    tf.import_graph_def(graph_def,{'input':t_preprocessed})

    # Now we've got our TensorFlow Model, we've downloaded it from the INTERNET
    # And we've loaded into our Session as a Graph with a bunch of layers !!! Yeah..BOOM BOOM !!!

    # -----It's a Convolutional Neural Network , The Type of Neural Network that helps to recognize images----- #

    # Let's load all those layers into an array and store it into our 'layers' object
    # So for every Tensorflow operation in our graph if its a Convolutional layer, load it into our array
    layers = [op.name for op in graph.get_operations() if op.type =='Conv2D' and 'import/' in op.name]

    # So each of our Convolutional layer outputs 10's or 100's of feature channels to pass data in the graph
    # And we can collect them all and store them into feature_nums variable
    feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]

    # Helper functions for TF Graph visualization
    #pylint: disable=unused-variable
    def strip_consts(graph_def, max_const_size=32):
        """Strip large constant values from graph_def."""
        strip_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = strip_def.node.add() #pylint: disable=maybe-no-member
            n.MergeFrom(n0)
            if n.op == 'Const':
                tensor = n.attr['value'].tensor
                size = len(tensor.tensor_content)
                if size > max_const_size:
                    tensor.tensor_content = "<stripped %d bytes>"%size
        return strip_def

    def rename_nodes(graph_def, rename_func):
        res_def = tf.GraphDef()
        for n0 in graph_def.node:
            n = res_def.node.add() #pylint: disable=maybe-no-member
            n.MergeFrom(n0)
            n.name = rename_func(n.name)
            for i, s in enumerate(n.input):
                n.input[i] = rename_func(s) if s[0]!='^' else '^'+rename_func(s[1:])
        return res_def

    def showarray(a):
        a = np.uint8(np.clip(a, 0, 1)*255)
        plt.imshow(a)
        plt.show()

    def visstd(a, s=0.1):
        '''Normalize the image range for visualization'''
        return (a-a.mean())/max(a.std(), 1e-4)*s + 0.5

    def T(layer):
        '''Helper for getting layer output tensor'''
        return graph.get_tensor_by_name("import/%s:0"%layer)

    def render_naive(t_obj, img0=img_noise, iter_n=20, step=1.0):
        t_score = tf.reduce_mean(t_obj) # defining the optimization objective
        t_grad = tf.gradients(t_score, t_input)[0] # behold the power of automatic differentiation!

        img = img0.copy()
        for _ in range(iter_n):
            g, _ = sess.run([t_grad, t_score], {t_input:img})
            # normalizing the gradient, so the same step size should work
            g /= g.std()+1e-8         # for different layers and networks
            img += g*step
        showarray(visstd(img))

    def tffunc(*argtypes):
        '''Helper that transforms TF-graph generating function into a regular one.
        See "resize" function below.
        '''
        placeholders = list(map(tf.placeholder, argtypes))
        def wrap(f):
            out = f(*placeholders)
            def wrapper(*args, **kw):
                return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
            return wrapper
        return wrap

    def resize(img, size):
        img = tf.expand_dims(img, 0)
        return tf.image.resize_bilinear(img, size)[0,:,:,:]
    resize = tffunc(np.float32, np.int32)(resize)

    def calc_grad_tiled(img, t_grad, tile_size=512):
        '''Compute the value of tensor t_grad over the image in a tiled way.
        Random shifts are applied to the image to blur tile boundaries over
        multiple iterations.'''
        sz = tile_size
        h, w = img.shape[:2]
        sx, sy = np.random.randint(sz, size=2)
        img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
        grad = np.zeros_like(img)
        for y in range(0, max(h-sz//2, sz),sz):
            for x in range(0, max(w-sz//2, sz),sz):
                sub = img_shift[y:y+sz,x:x+sz]
                g = sess.run(t_grad, {t_input:sub})
                grad[y:y+sz,x:x+sz] = g
        return np.roll(np.roll(grad, -sx, 1), -sy, 0)

    #  We will start by defining our optimization objective
    # which is to reduce the mean of our input layer

    def render_deepdream(t_obj, img0=img_noise, iter_n= 10, step= 1.5, octave_n= 8, octave_scale=1.4):
        t_score = tf.reduce_mean(t_obj) # defining optimization objective

        # the gradients() function lets us compute the symbolic gradient of our optimized tensor
        #  with respect to our input tensor
        t_grad = tf.gradients(t_score, t_input)[0]

        # split the image into number of octaves
        img = img0
        octaves = []

        for _ in range(octave_n-1):
            hw = img.shape[:2]
            lo = resize(img, np.int32(np.float32(hw)/octave_scale))
            hi = img-resize(lo, hw)
            img = lo
            octaves.append(hi)

        # generate details octave by octave
        # by iterating through each octave
        for octave in range(octave_n):
            if octave>0:
                hi = octaves[-octave]
                img = resize(img, hi.shape[:2])+ hi
            # random shapes are applied to the image to blur tile boundaries over multiple iterations using
            # calc_grad_tiled() function
            for _ in range(iter_n):
                g = calc_grad_tiled(img, t_grad)
                img += g*(step/ (np.abs(g).mean()+1e-7))
            # Step 5 Output deep dreamed image
            showarray(img/255.0)
        # We essentially applied gradient ascent to maximize the loss function
        # which merges our saved representation in this layer with our input image more and more every iteration



    # Let's print them out and visualize what we've got
    print('Number of layers', len(layers))
    print('Total number of Feature  channels:', sum(feature_nums))

    # Step 3

    # Let's now pick a layer from our model that we are going to enhance
    # We'll pick a lower-level layer and pick a channel to visualize.
    layer = 'mixed4d_3x3_bottleneck_pre_relu'
    channel = 139  # picking some feature channel to visualize

    # It's time for using the Pillow image sub module's open() method and store it in our image variable.
    img0 = PIL.Image.open('friends.jpg')

    # We'll format it accordingly using NumPy and perform DeepDream on it with our render_deepdream() function
    # with the focus on the layer we selected earlier
    img0 = np.float32(img0)

    # Step 4
    # Apply Gradient Ascent to that layer
    # we can see a couple of predefined hyperparameters
    render_deepdream(T(layer)[:,:,:,139], img0)

if __name__ == "__main__":
    main()






