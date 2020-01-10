#Source code: https://github.com/eragonruan/refinenet-image-segmentation


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

def color_map(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

def return_overlayed_img(image, seg):
    target = process_seg(np.array(seg))[:, :, np.newaxis]
    cmap = color_map()[:, np.newaxis, :]
    new_im = np.dot(target == 0, cmap[0])
    for i in range(1, cmap.shape[0]):
        new_im += np.dot(target == i, cmap[i])
    new_im = Image.fromarray(new_im.astype(np.uint8))
    blend_image = Image.blend(image, new_im, alpha=0.2)
    return (np.array(blend_image)*255.).astype('uint8')

def main(argv=None):
    import os
    if os.path.exists(FLAGS.result_path):
        shutil.rmtree(FLAGS.result_path)
    os.makedirs(FLAGS.result_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    pascal_voc_lut = pascal_segmentation_lut()

    with tf.get_default_graph().as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        logits = model.model(input_images, is_training=False)
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

            ###############################
            pkl_file = 'data/arroyo_seg.pkl'
            with open(pkl_file, 'rb') as f:
                obj = pickle.load(f)
            im_fn_list, anno_files = obj

            # im_fn_list = get_images()
            for im_fn, seg_fn in zip(im_fn_list, anno_files):
                # im = cv2.imread(im_fn)[:, :, ::-1]
                im = np.array(Image.open(im_fn))
                seg = np.array(Image.open(seg_fn))
                im_resized, (ratio_h, ratio_w) = resize_image(im, size=32)
                # import ipdb; ipdb.set_trace()
                
                start = time.time()
                pred_re = sess.run([pred], feed_dict={input_images: [im_resized]})
                pred_re = np.array(np.squeeze(pred_re))

                seg[seg==255]=0   

                img=visualize_segmentation_adaptive(pred_re, pascal_voc_lut)
                img_seg=visualize_segmentation_adaptive(seg, pascal_voc_lut)

                # import ipdb; ipdb.set_trace()
                #img_true=return_overlayed_img(Image.fromarray(img), Image.fromarray(seg))
                #img_pred=return_overlayed_img(Image.fromarray(img), Image.fromarray(pred_re))
                
                
                
                _diff_time = time.time() - start
                cv2.imwrite(os.path.join(FLAGS.result_path, os.path.basename(im_fn)), np.hstack((img, img_seg)))    

                print('{}: cost {:.0f}ms'.format(im_fn, _diff_time * 1000))



if __name__ == '__main__':
    tf.app.run()
