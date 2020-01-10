import os
import json
import glob
import pickle
from tqdm import tqdm
from difflib import get_close_matches
from tqdm import tqdm

def arroyo_segmentation_lut():
    """Return look-up table with number and correspondng class names
    for PASCAL VOC segmentation dataset. Two special classes are: 0 -
    background and 255 - ambigious region. All others are numerated from
    1 to 20.
    
    Returns
    -------
    classes_lut : dict
        look-up table with number and correspondng class names
    """

    # class_names = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
    #                'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    #                'dog', 'horse', 'motorbike', 'person', 'potted-plant',
    #                'sheep', 'sofa', 'train', 'tv/monitor', 'ambigious']

    # class_names = ['background' 'gravel', 'sand', 'boulder', 'artifacts', 
    #                'vegetation', 'fragments', 'biotic-substances', 'sky', 'ambigious'] # 9

    class_names = ['gravel', 'sand', 'boulder', 'artifacts', 
                   'vegetation', 'fragments', 'biotic-substances', 'sky', 'blank', 'blank', 'blank', 'blank',
                   'blank', 'blank', 'blank', 'blank', 'blank',
                   'blank', 'blank', 'blank', 'blank', 'ambigious']
    
    #0: nothing (most) 1: gravel (most) 2: sand (most) 3: boulder (rare) 
    #4: artifacts (medium) 5: vegetation (medium) 6: fragments (most) 7: biotic substances (most) 8: sky (rare)
    
    enumerated_array = enumerate(class_names[:-1])
    
    classes_lut = list(enumerated_array)
    
    # Add a special class representing ambigious regions
    # which has index 255.
    classes_lut.append((255, class_names[-1]))
    
    classes_lut = dict(classes_lut)

    return classes_lut


def get_augmented_pascal_image_annotation_filename_pairs():
    pkl_file = 'data/arroyo_seg.pkl'
    if not os.path.exists(pkl_file):
        img_dir = os.path.expanduser("~/Data/jpl/images")
        anno_dir = os.path.expanduser("~/Data/jpl/annotations/spoc")
        img_list = [f for f in glob.glob(img_dir + "**/*", recursive=True)]
        anno_list = [f for f in glob.glob(anno_dir + "**/*", recursive=True)]
        image_files = []
        anno_files = []
        for im in tqdm(img_list):
            image_files.append(im)
            anno_files.append(get_close_matches(im, anno_list)[0])

        with open(pkl_file, 'wb') as f:
            pickle.dump([image_files, anno_files], f)

    else:
        with open(pkl_file, 'rb') as f:
            obj = pickle.load(f)
        image_files, anno_files = obj
    

    n_train = int(0.9 * len(image_files))

    # import ipdb; ipdb.set_trace()

    train_img_files = image_files[:n_train]
    train_anno_files = anno_files[:n_train]

    val_img_files = image_files[n_train:]
    val_anno_files = anno_files[n_train:]

    overall_train_image_annotation_filename_pairs = zip(train_img_files, train_anno_files)
    overall_val_image_annotation_filename_pairs = zip(val_img_files, val_anno_files)
    # import ipdb; ipdb.set_trace()
    return overall_train_image_annotation_filename_pairs, overall_val_image_annotation_filename_pairs