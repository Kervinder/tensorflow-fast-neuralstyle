import numpy as np
import argparse
import tensorflow as tf
import os
from PIL import Image

parser = argparse.ArgumentParser(description='Real-time style transfer image generator')
parser.add_argument('--input', '-i', type=str, help='content image')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--style', '-s', default=None, type=str, help='style model name')
parser.add_argument('--ckpt', '-c', default=-1, type=int, help='checkpoint to be loaded')
parser.add_argument('--out', '-o', default='stylized_image.jpg', type=str, help='stylized image\'s name')
parser.add_argument('--pb', '-pb', dafault=False, type=bool, help='load with pb')

args = parser.parse_args()

if not os.path.exists('./images/output/'):
        os.makedirs('./images/output/')

outfile_path = './images/output/' + args.out
content_image_path = args.input
style_name = args.style
ckpt = args.ckpt
load_with_pb = args.pb

original_image = Image.open(content_image_path).convert('RGB')

img = np.asarray(original_image.resize((224, 224)), dtype=np.float32)
shaped_input = img.reshape((1,) + img.shape)

if args.gpu > -1:
    device_ = '/gpu:{}'.format(args.gpu)
    print(device_)
else:
    device_ = '/cpu:0'


with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    if ckpt < 0:
        checkpoint = tf.train.get_checkpoint_state('./ckpts/{}/'.format(style_name))
        input_checkpoint = checkpoint.model_checkpoint_path
    else:
        input_checkpoint = './ckpts/{}/{}-{}'.format(style_name, style_name, ckpt)
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta')
    saver.restore(sess, input_checkpoint)

    graph = tf.get_default_graph()
    input_image = graph.get_tensor_by_name('input:0')
    output = graph.get_tensor_by_name('output:0')
    out = sess.run(output, feed_dict={input_image: shaped_input})
    
out = out.reshape((out.shape[1:]))
im = Image.fromarray(np.uint8(out))

im = im.resize(original_image.size, resample=Image.LANCZOS)
im.save(outfile_path)
    