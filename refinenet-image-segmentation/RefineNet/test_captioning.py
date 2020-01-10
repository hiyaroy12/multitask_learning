import re
import cv2
import time
import os,shutil
import sys
from PIL import Image
import pickle
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

sys.path.append(os.getcwd())
from nets import model as model
from matplotlib import pyplot as plt

from utils.arroyo import arroyo_segmentation_lut as pascal_segmentation_lut
from utils.visualization import visualize_segmentation_adaptive
tf.app.flags.DEFINE_string('test_data_path', '/home3/hiya/Data/jpl/images' , '') #   'demo'
tf.app.flags.DEFINE_string('gpu_list', '1,3', '')
tf.app.flags.DEFINE_integer('num_classes', 21, '')
# tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints_arroyo', '')
#tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints_arroyo/RefineNet_step_32001.ckpt', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'checkpoints_arroyo/RefineNet_step_46001.ckpt', '')
tf.app.flags.DEFINE_string('result_path', 'result/', '')

FLAGS = tf.app.flags.FLAGS


def get_images():
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print ('Find {} images'.format(len(files)))
    return files

def resize_image(im, size=32, max_side_len=500):
    h, w, _ = im.shape
    resize_w = w
    resize_h = h
    #if max(resize_h, resize_w) > max_side_len:
    ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    #else:
    #    ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)
    resize_h = resize_h if resize_h % size == 0 else (resize_h // size) * size
    resize_w = resize_w if resize_w % size == 0 else (resize_w // size) * size
    im = cv2.resize(im, (int(resize_w), int(resize_h)))
    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)
    return im, (ratio_h, ratio_w)

def process_seg(x):
    x = x.astype('int') + 1
    x[x==256]=0
    return x



def main(argv=None):
    import os
    if os.path.exists(FLAGS.result_path):
        shutil.rmtree(FLAGS.result_path)
    os.makedirs(FLAGS.result_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    pascal_voc_lut = pascal_segmentation_lut()
    
    import glob
    im_fn_list = glob.glob('/home3/hiya/workspace/SCOTI/codes/images/test_resized/*')
    #print(len(all_files))
    
    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        logits, f = model.model_feats(input_images, is_training=False)
        pred = tf.argmax(logits, dimension=3)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        # saver = tf.train.Saver(variable_averages.variables_to_restore())
        saver = tf.train.Saver(tf.global_variables())
        

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            # model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))

            # ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            # restore_step=int(ckpt.split('.')[0].split('_')[-1])

            model_path = FLAGS.checkpoint_path

            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)


            feat_list = []
            print(im_fn_list)
            for im_fn in im_fn_list:
                # im = cv2.imread(im_fn)[:, :, ::-1]
                im = np.array(Image.open(im_fn))
                im_resized = cv2.resize(im, (256,256))
                start = time.time()
                feats = sess.run(f, feed_dict={input_images: [im_resized]})
                feats = feats[-4]
                feat_list.append(feats)
            
           
            np_feats = np.concatenate(feat_list, axis = 0)
            np.save('/home3/hiya/Data/jpl/annotations/scoti/resnet101_arroyo_caption_test.npy', np_feats)
            
            #import ipdb; ipdb.set_trace()
            


if __name__ == '__main__':
    tf.app.run()