# coding: utf-8
pascal_root = '/home3/hiya/Data/pascal_voc_seg/VOCdevkit/VOC2012'

from utils.arroyo import get_augmented_pascal_image_annotation_filename_pairs
from utils.tf_records import write_image_annotation_pairs_to_tfrecord

# Returns a list of (image, annotation) filename pairs (filename.jpg, filename.png)
overall_train_image_annotation_filename_pairs, overall_val_image_annotation_filename_pairs =get_augmented_pascal_image_annotation_filename_pairs()
# pairs = [(x,y) for x,y in overall_train_image_annotation_filename_pairs]
# import ipdb; ipdb.set_trace()
# You can create your own tfrecords file by providing
# your list with (image, annotation) filename pairs here

write_image_annotation_pairs_to_tfrecord(filename_pairs=overall_val_image_annotation_filename_pairs,
                                         tfrecords_filename='pascal_val.tfrecords', arroyo=True)

write_image_annotation_pairs_to_tfrecord(filename_pairs=overall_train_image_annotation_filename_pairs,
                                         tfrecords_filename='pascal_train.tfrecords', arroyo=True)

