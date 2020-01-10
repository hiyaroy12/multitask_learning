# multitask_learning
Segmentation and captioning

##SPOC: 
##Introduction:
this is a tensorflow implement of the paper: RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation [arxiv:1611.06612](https://arxiv.org/abs/1611.06612)


##requirements:
• python2/3 compatible
• tensorflow 1.2
• some dependencies like cv2, numpy etc. 
• recommend to install Anaconda


##data preparation:
• arroyo images and corresponding SPOC and SCOTI labels used are available at: 

• More details about dataset:
  utils/arroyo.py 
  	- class_names = ['gravel', 'sand', 'boulder', 'artifacts', 'vegetation', 'fragments', 'biotic-substances', 'sky', 'blank', 'blank', 'blank', 'blank','blank', 'blank', 'blank', 'blank', 'blank', 'blank', 'blank', 'blank', 'blank', 'ambigious'] to match Pascal VOC convention. Class ‘ambiguous’ refers to as the 255-labeled pixels, we ignore it for both training and inference.
	- Put the pkl_file under 'data/arroyo_seg.pkl' available: https://www.dropbox.com/s/tilrro30pv7eabj/arroyo_seg.pkl?dl=0
	- Change the img_dir and anno_dir
	- Data split: available at utils/arroyo.py 
• download the pretrain model of resnet_v1_101.ckpt, you can download it from https://www.dropbox.com/s/188uyrunr32f14c/resnet_v1_101.ckpt?dl=0


##training:
• run convert_arroyo_to_tfrecords.py to convert training data into .tfrecords or you can download it here: 
https://www.dropbox.com/s/0ag1ywvq0t3nqf9/pascal_train_arroyo.tfrecords?dl=0 https://www.dropbox.com/s/8qd34kgqjujvm0p/pascal_val_arroyo.tfrecords?dl=0
• run python RefineNet/multi_gpu_train.py 

##validation:
• if you want to download the model I trained and used for validation: 'checkpoints_arroyo/RefineNet_step_46001.ckpt', its available here:  https://www.dropbox.com/s/sz12b6caemphy2z/RefineNet_step_46001.zip?dl=0
• put images in demo/ and run python RefineNet/demo.py
