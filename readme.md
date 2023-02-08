
## How to run our code?

# Set Up Environment
Our code is built on top of HTCN. We follow the paper in our implementation details and Faster-RCNN respository to setup the 
environment. We use python 3.8, pytorch 1.7.1 and a single 3090 for training. You can set up your own 
environment according to your GPU computing power and you can install other packages by running 'pip install -r requirements.txt'
(this should be done when the environment is activated and Faster-RCNN has already been set up.)
You can run:'python setup.py build develop' to set up faster-rcnn. If you have problems setting up, you may find solutions in 
issues of HTCN or Faster-RCNN respository and use provided setting up files in lib/build.
You may need this command: 'export TORCH_CUDA_ARCH_LIST="8.0"' to help you make your GPU compatible with your CUDA version.


# Dataset Preparation
For double blind principle, we have removed all external links. We will add the links of datasets and models in 
our open-source version to help readers implement our code more easily. Please download the dataset from official 
dataset websites. All codes are written to fit for the format of PASCAL_VOC. Please prepare the annotations in VOC format.  
Please download vgg16_caffe.pth and change the path in lib/model/utils/config.py and change the data path in lib/datasets/config_dataset.py.
Also change your path in lib/model/utils/parser_func.py in line 416 to change cfg_path.
After preparation ,your dataset should looks like:
cityscape_t/
|--Annotations(VOC format)
|--ImageSets
   |--Main
      |--train.txt
      |--val.txt
      |--trainval.txt
|--JPEGImages


# Run Scripts
We take cityscapes -> foggy-cityscapes as an example. We have 2975 images for training and 500 images for test for both
cityscapes and foggy-cityscapes, respectively. We name the folders of these two datasets as cityscapes_s and cityscapes_t.
This is a four stage training process:

# Stage1. First pretrain your model using cityscapes training dataset:
CUDA_VISIBLE_DEVICES="your gpu id" python trainval_pretrain_adv.py --dataset cs --net vgg16 --log_ckpt_name "your own path" --save_dir "your own path"

# Stage2. Select source_similar data and source_dissimilar data and construct dataset cityscapes_similar and cityscapes_disimilar
# Duplicate the folder cityscape_t to two folders cityscapes_t_similar and cityscapes_t_disimilar 
# Change line 224 and 231 in select_by_uncertainty.py to localize the path of train.txt in foggy_cityscapes_similar(disimilar)/ImageSets/Main
CUDA_VISIBLE_DEVICES="your gpu id" python select_by_uncertainty.py --dataset_t cs_fg --net vgg16 --log_ckpt_name "your own path" --save_dir "your own path" --load_name "path of checkpoint of stage1"

# Stage3. TCD adversarial training with mean teacher
CUDA_VISIBLE_DEVICES="your gpu id" python trainval_adv_mt.py --epochs 30 --dataset cs_fg_similar --dataset_t cs_fg_disimilar --net vgg16 --log_ckpt_name "your own path" --save_dir "your own path" --load_name "path of checkpoint of stage1"

# Stage4. Fine-tune with False negative simulation 
CUDA_VISIBLE_DEVICES="your gpu id" python trainval_adv_mosaic_mt.py --epochs 20 --dataset cs_fg --dataset_t cs_fg_similar --net vgg16 --log_ckpt_name "your own path" --save_dir "your own path" --load_name "path of checkpoint of stage3"

# Stage5. Test the result
CUDA_VISIBLE_DEVICES="your gpu id" python test_net_adv.py --dataset_t cs_fg --net vgg16 --log_ckpt_name "your own path" --save_dir "your own path" --load_name "path of checkpoint of stage4"

