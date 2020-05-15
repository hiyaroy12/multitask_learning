multitask_learning (segmentation and captioning)

SPOC:
--------------------------------------------------------------------------------------------------------------------

Introduction:
--------------------------------------------------------------------------------------------------------------------
this is a tensorflow implementation of the paper: RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation [arxiv:1611.06612](https://arxiv.org/abs/1611.06612)

requirements:
--------------------------------------------------------------------------------------------------------------------
-	python2/3 compatible
-	tensorflow 1.2
-	some dependencies like cv2, numpy etc. 
-	recommend to install Anaconda

data preparation:
--------------------------------------------------------------------------------------------------------------------
-	Arroyo images that I have used are located here: fornat1:/home/maars/data/arroyo_img_labels/images>
        Corresponding labels for spoc and scoti: fornat1:/home/maars/data/arroyo_img_labels/label> 
-	More details about dataset: 
	utils/arroyo.py 
  	- Class ‘ambiguous’ refers to as the 255-labeled pixels, we ignore it for both training and inference.
	- Put the pkl_file under 'data/arroyo_seg.pkl' available: 
	https://www.dropbox.com/s/tilrro30pv7eabj/arroyo_seg.pkl?dl=0
	- Change the img_dir and anno_dir
	- Data split: available at utils/arroyo.py 

training:
--------------------------------------------------------------------------------------------------------------------
-	run convert_arroyo_to_tfrecords.py to convert training data into .tfrecords or you can download it here: 
https://www.dropbox.com/s/0ag1ywvq0t3nqf9/pascal_train_arroyo.tfrecords?dl=0 https://www.dropbox.com/s/8qd34kgqjujvm0p/pascal_val_arroyo.tfrecords?dl=0
-	run python RefineNet/multi_gpu_train.py 
	- you can download the pretrain model of resnet_v1_101.ckpt here:		         https://www.dropbox.com/s/188uyrunr32f14c/resnet_v1_101.ckpt?dl=0


validation:
--------------------------------------------------------------------------------------------------------------------
-	if you want to download the model I trained and used for validation: 'checkpoints_arroyo/RefineNet_step_46001.ckpt', its available here:  https://www.dropbox.com/s/sz12b6caemphy2z/RefineNet_step_46001.zip?dl=0
-	put images in demo/ and run python RefineNet/demo.py



--------------------------------------------------------------------------------------------------------------------
SCOTI-captioning:
--------------------------------------------------------------------------------------------------------------------
This is a TensorFlow implementation of Show, Attend and Tell: Neural Image Caption Generation with Visual Attention which introduces an attention based image caption generator. https://arxiv.org/abs/1502.03044 

In case of our proposed “universal encoder” we use the features from RefineNet and use it for training the LSTM module of Show, Attend and Tell model.
-	pycocoevalcap is expected to be in SCOTI/codes/mica/

data preparation
--------------------------------------------------------------------------------------------------------------------
- SCOTI/codes/load_json_for_arroyo_images.ipynb generates ‘img_caption_list.json’ file.

training:
--------------------------------------------------------------------------------------------------------------------
- RefineNet/demo-captioning.py generates the feature vector ‘resnet101_arroyo_features.npy’ available at: https://www.dropbox.com/s/v00i6werrbkkl5z/resnet101_arroyo_features.npy?dl=0
	- To run this file make sure your pwd is refinenet-image-segmentation/ and run python RefineNet/demo_captioning.py to generate features on your own.

- To train the image captioning model, run SCOTI/codes/mica/ train_arroyo.py

	- To run this code: change the path of the ‘annotations_file’ and ‘all_feats’ path.
	- This code saves at model model_path='../models/lstm/
	- Or you can use the trained lstm ‘model-500’ which can be downloaded https://www.dropbox.com/s/3fmvyoyhwtnuxm9/model-500.zip?dl=0
		- In that case for similar convention with this code put ‘model-500’ under codes/models/lstm/model-500'

validation:
--------------------------------------------------------------------------------------------------------------------
-	Put test images to generate captions at: SCOTI/codes/images/test_resized/
-	Run refinenet-image-segmentation/RefineNet/test_captioning.py that saves the image features as ‘resnet101_arroyo_caption_test.npy'. Specify your desired path to save this file. 
	- To run this file make sure your pwd is refinenet-image-segmentation/ and run python RefineNet/test_captioning.py 
-	Run SCOTI/codes/mica/test_arroyo.py to generate captions. 
	- ‘all_feats’ file path has to be changed.


