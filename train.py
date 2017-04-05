import numpy as np
import os, sys
import argparse
from PIL import Image
import tensorflow as tf
import time

from net import *
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "./"))
from custom_vgg16 import *


# gram matrix per layer
def gram_matrix(x):
    assert isinstance(x, tf.Tensor)
    b, h, w, ch = x.get_shape().as_list()
    features = tf.reshape(x, [b, h*w, ch])
    # gram = tf.batch_matmul(features, features, adj_x=True)/tf.constant(ch*w*h, tf.float32)
    gram = tf.matmul(features, features, adjoint_a=True)/tf.constant(ch*w*h, tf.float32)
    return gram

# total variation denoising
def total_variation_regularization(x, beta=1):
    assert isinstance(x, tf.Tensor)
    wh = tf.constant([[[[ 1], [ 1], [ 1]]], [[[-1], [-1], [-1]]]], tf.float32)
    ww = tf.constant([[[[ 1], [ 1], [ 1]], [[-1], [-1], [-1]]]], tf.float32)
    tvh = lambda x: conv2d(x, wh, p='SAME')
    tvw = lambda x: conv2d(x, ww, p='SAME')
    dh = tvh(x)
    dw = tvw(x)
    tv = (tf.add(tf.reduce_sum(dh**2, [1, 2, 3]), tf.reduce_sum(dw**2, [1, 2, 3]))) ** (beta / 2.)
    return tv

parser = argparse.ArgumentParser(description='Real-time style transfer')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--dataset', '-d', default='dataset', type=str,
                    help='dataset directory path (according to the paper, use MSCOCO 80k images)')
parser.add_argument('--style_image', '-s', type=str, required=True,
                    help='style image path')
parser.add_argument('--batchsize', '-b', type=int, default=1,
                    help='batch size (default value is 1)')
parser.add_argument('--input', '-i', default=None, type=str,
                    help='input model file path without extension')
parser.add_argument('--output', '-o', default='out', type=str,
                    help='output model file path without extension')
parser.add_argument('--lambda_tv', '-l_tv', default=10e-4, type=float,
                    help='weight of total variation regularization according to the paper to be set between 10e-4 and 10e-6.')
parser.add_argument('--lambda_feat', '-l_feat', default=1e0, type=float)
parser.add_argument('--lambda_style', '-l_style', default=1e1, type=float)
parser.add_argument('--epoch', '-e', default=2, type=int)
parser.add_argument('--lr', '-l', default=1e-3, type=float)
parser.add_argument('--checkpoint', '-c', default=0, type=int)
args = parser.parse_args()

data_dict = loadWeightsData('./vgg16.npy')
batchsize = args.batchsize

n_epoch = args.epoch
lambda_tv = args.lambda_tv
lambda_f = args.lambda_feat
lambda_s = args.lambda_style
output = args.output

fpath = os.listdir(args.dataset)
imagepaths = []
for fn in fpath:
    base, ext = os.path.splitext(fn)
    if ext == '.jpg' or ext == '.png':
        imagepath = os.path.join(args.dataset,fn)
        imagepaths.append(imagepath)
n_data = len(imagepaths)
print ('num traning images:', n_data)
n_iter = int(n_data / batchsize)
print (n_iter, 'iterations,', n_epoch, 'epochs')

style_ = np.asarray(Image.open(args.style_image).convert('RGB').resize((224,224)), dtype=np.float32)
styles_ = [style_ for x in range(batchsize)]

if args.gpu > -1:
    device_ = '/gpu:{}'.format(args.gpu)
    print(device_)
else:
    device_ = '/cpu:0'

with tf.device(device_):
    
    model = FastStyleNet()
    saver = tf.train.Saver(restore_sequentially=True)
    saver_def = saver.as_saver_def()

    inputs = tf.placeholder(tf.float32, shape=[batchsize, 224, 224, 3])
    target = tf.placeholder(tf.float32, shape=[batchsize, 224, 224, 3])
    outputs = model(inputs)

    # style target feature
    # compute gram maxtrix of style target
    vgg_s = custom_Vgg16(target, data_dict=data_dict)
    feature_ = [vgg_s.conv1_2, vgg_s.conv2_2, vgg_s.conv3_3, vgg_s.conv4_3, vgg_s.conv5_3]
    gram_ = [gram_matrix(l) for l in feature_]

    # content target feature 
    vgg_c = custom_Vgg16(inputs, data_dict=data_dict)
    feature_ = [vgg_c.conv1_2, vgg_c.conv2_2, vgg_c.conv3_3, vgg_c.conv4_3, vgg_c.conv5_3]

    # feature after transformation 
    vgg = custom_Vgg16(outputs, data_dict=data_dict)
    feature = [vgg.conv1_2, vgg.conv2_2, vgg.conv3_3, vgg.conv4_3, vgg.conv5_3]

    # compute feature loss
    loss_f = tf.zeros(batchsize, tf.float32)
    for f, f_ in zip(feature, feature_):
        loss_f += lambda_f * tf.reduce_mean(tf.subtract(f, f_) ** 2, [1, 2, 3])

    # compute style loss
    gram = [gram_matrix(l) for l in feature]
    loss_s = tf.zeros(batchsize, tf.float32)
    for g, g_ in zip(gram, gram_):
        loss_s += lambda_s * tf.reduce_mean(tf.subtract(g, g_) ** 2, [1, 2])

    # total variation denoising
    loss_tv = lambda_tv * total_variation_regularization(outputs)

    # total loss
    loss = loss_s + loss_f + loss_tv

    # optimizer
    train_step = tf.train.AdamOptimizer(args.lr).minimize(loss)

# for calculating time
s_time = time.time()

var_list={}
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:

    model_directory = './models/'
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    # training
    tf.global_variables_initializer().run()

    if args.input:
        saver.restore(sess, args.input)
        print ('restoring model....')

    for epoch in range(n_epoch):
        print ('epoch', epoch)
        imgs = np.zeros((batchsize, 224, 224, 3), dtype=np.float32)
        for i in range(n_iter):
            for j in range(batchsize):
                p = imagepaths[i*batchsize + j]
                imgs[j] = np.asarray(Image.open(p).convert('RGB').resize((224, 224)), np.float32)
            feed_dict = {inputs: imgs, target:styles_}
            loss_, _= sess.run([loss, train_step,], feed_dict=feed_dict)
            print('(epoch {}) batch {}/{}... training loss is...{}'.format(epoch, i, n_iter-1, loss_[0]))
    saver.save(sess, model_directory + args.output+'.model', write_meta_graph=False)

    for var in tf.global_variables():
        var_list[var.name] = var.eval()


Model = tf.Graph()
with Model.as_default():
    with tf.device(device_):
        inputs = tf.placeholder(tf.float32, shape=[1, 224, 224, 3], name='input')
        # feed dictionary into Transform Net, "train=False" would save all values as constants.
        transform = FastStyleNet(train=False, data_dict=var_list)
        outputs = transform(inputs)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:

        save_path = './graphs/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print('saving pb...')
        tf.train.write_graph(sess.graph_def, save_path, args.output + '.pb', as_text=False)
