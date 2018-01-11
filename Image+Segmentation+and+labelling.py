
# coding: utf-8

#  # Image Segmentation 
#  
#  Author: Rajya Laxmi Yellajosyula
#  
#  Below tasks are performed by setting up AWS and successfully running the below code in this notebook.
#  

# # What is segmentation?
# 
# Segmentation is the task of "labeling" groups of pixels in an image to identify certain objects.
# 
# In the early years, research on segmentation was focused on "foreground-background" segmentation; marking only those pixels that comprise the "background" of an image (in the image below, the background is marked in blue).
# 
# <div>
# <img src="http://www.eyeshalfclosed.com/images/cat.jpg" width=500/>
# </div>
# 
# In recent years, sophisticated deep-learning models have enabled complex multi-label segmentation, such as in the images below.
# 
# <tr>
# <td>
# <img src="http://www.eyeshalfclosed.com/images/sheep.png" width=500/>
# </td>
# <td>
# <img src="http://www.eyeshalfclosed.com/images/street.png" width=500/>
# </td>
# </tr>
#    

# # Prerequisites
# 
# Below tasks are run on an AWS GPU instance with the specifications listed below;
# 
# **Machine.**
# 
#    - Used the [Ubuntu Deep Learning AMI](https://aws.amazon.com/marketplace/pp/B077GCH38C).
#    - Used a p2.xlarge instance.
#    - Allocated at least 80GB of disk space.
#    - Used the `conda_tensorflow_p36` Conda environment: `source activate conda_tensorflow_p36`
#    - Created a security group and open all inbound/outbound ports to 0.0.0.0/0.
# 
# All commands below assume the aforementioned Conda environment is active.
# 
# **Run Jupyter.** `jupyter notebook --ip=* --no-browser`
# 
# You may move Jupyter to the background by: CTRL-Z, then `bg`, then `disown`. You can access Jupyter using your public DNS; it will look something like `ec2-54-84-36-171.compute-1.amazonaws.com:8888`. Figure out how you can find this out.
# 
# **Data downloads.** All downloads must go into the same directory as this notebook. Unzip files after download. *This will take time.*
# 
#    * Download the [trained model weights](https://github.com/matterport/Mask_RCNN/releases/download/v1.0/mask_rcnn_coco.h5) (~250MB).
# 
#    * Download the [training images](http://images.cocodataset.org/zips/train2014.zip) (13GB).
#    
#    * Download the [validation images](http://images.cocodataset.org/zips/val2014.zip) (6GB).
#    * Download the [training image annotations](https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0).
#    * Download the [test image annotations](https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0).
# 
# Now create a new folder named `2014`, then move the `train2014`, `val2014` folders into `2014/`.
# 
# Create a new `2014/annotations/` folder and move the train and test annotation JSON files into it
# 
# Your directory structure should look like:
# ```
# 2014/
#    /annotations/
#        /annotations/instances_minival2014.json
#        /annotations/instances_valminusminival2014.json
#    /train2014/
#        /train2014/*.jpg
#    /val2014/
#        /val2014/*.jpg
# ```
# 
# **Package installation.**
# 
#    * Install Cython: ``pip install cython
#    * Install Tensorflow: `pip install tensorflow==1.3.0 tensorflow-gpu==1.3.0`
#    * Install Keras and image tools: `pip install keras scikit-image pillow h5py`
#    * Install OpenCV: `pip install opencv-python`
#    * Install pycoco:
#    
# `pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"`
#    
# **GPU.** Ensure Keras/TensorFlow can see your GPU with the following Python code (run in the `conda_tensorflow_p36 environment` after installing all the required packages). You should see a GPU in one of the devices listed.

# In[24]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# # Testing a pre-trained model on small data
# 
# We will first load a pre-trained convolutional neural network model and test it on a small dataset of images. These images are stored in the `/images/` folder.
# 
# The model was trained by annotating each image with the objects it contains. Annotations are in the following format:
# 
# ```
# annotation{
#     "id" : int,
#     "image_id" : int,
#     "category_id" : int,
#     "segmentation" : RLE or [polygon],
#     "area" : float,
#     "bbox" : [x,y,width,height],
#     "iscrowd" : 0 or 1,
# }
# 
# categories[{
#     "id" : int,
#     "name" : str,
#     "supercategory" : str,
# }]
# ```
# 
# To understand the annotations and how they are connect to images  look at [section 4 on this page](http://cocodataset.org/#download). You may ignore the `iscrowd` variable.

# ## Set up the environment

# In[25]:


import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
from model import log
import visualize
from config import Config
from shapes import ShapesDataset

from pycocotools.coco import COCO

get_ipython().magic('matplotlib inline')

ROOT_DIR = os.getcwd()
MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
IMAGE_DIR = os.path.join(ROOT_DIR, "images")


# ## Model configuration
# 
# These lines specify how many GPUs to use, and how many images to process in parallel on each GPU.

# In[26]:


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
#config.display()


# ## Load the pre-trained model
# 
# This is actually a Keras model wrapped along with some helpful functions. The model may be loaded in two modes: `training` and `inference` (testing) mode. `model_dir` points towards a directory to save logs and trained weights, which we have set above as the `/logs` directory.

# In[27]:


get_ipython().run_cell_magic('time', '', 'model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)\nmodel.load_weights(COCO_MODEL_PATH, by_name=True)')


# ## Hard-code object classes
# 
# For the small dataset of images we are using, we define our own list of class names and class indices for each object. These are of various types: for example, "car", "bicycle", etc..

# In[28]:


# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


# ## Load and visualize a random image
# 
# Below code loads and visualizes a random image

# In[29]:


file_names = next(os.walk(IMAGE_DIR))[2]
image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
plt.imshow(image)
plt.show()


# ## Test the pre-trained model
# 
# We now call the `detect` function of the model on the list of images we want to be segmented. This returns a `result` object; inspect this object to see what it contains.
# 
# The `visualize` helper module provides useful functions to visualize our segmentation results. Understand how this function works (SHIFT+TAB in Jupyter is useful, as well as looking at the code in `visualize.py` directly).

# In[30]:


get_ipython().run_cell_magic('time', '', "results = model.detect([image], verbose=1)\nr = results[0]\nvisualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], \n                            class_names, r['scores'])")


# # Training from scratch
# 
# Now that we understand what a properly trained model should do, we consider training a model from scratch.

# ## Load the data
# 
# Load the annotations for the training images into memory.

# In[31]:


get_ipython().run_cell_magic('time', '', 'config = coco.CocoConfig()\nCOCO_DIR = "2014"\ndataset = coco.CocoDataset()\ndataset.load_coco(COCO_DIR, "minival")\ndataset.prepare()')


# Now load the same for the test images.

# In[32]:


get_ipython().run_cell_magic('time', '', 'dataset_val = coco.CocoDataset()\ndataset_val.load_coco(COCO_DIR, "val35k")\ndataset_val.prepare()')


# ## List a few object classes

# In[33]:


print("Image Count: {}".format(len(dataset.image_ids)))
print("Class Count: {}".format(dataset.num_classes))
for i, info in enumerate(dataset.class_info):
    print("{:3}. {:50}".format(i, info['name']))
    if i > 10:
        break


# ## Visualize a random image and its annotations

# In[34]:


# Load random image and mask.
image_id = random.choice(dataset.image_ids)
image = dataset.load_image(image_id)
mask, class_ids = dataset.load_mask(image_id)
bbox = utils.extract_bboxes(mask)

# Display image and additional stats
print("image_id ", image_id, dataset.image_reference(image_id))
log("image", image)
log("mask", mask)
log("class_ids", class_ids)
log("bbox", bbox)

# Display image and instances
visualize.display_instances(image, bbox, mask, class_ids, dataset.class_names)


# ## Training configuration
# 
# See the default configuration values in `config.py`.

# In[35]:


class TrainConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 60

config = TrainConfig()
config.display()


# ## Create a new model in training mode 

# In[36]:


# Create model in training mode

model_new = modellib.MaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)


# ## Initialize the model weights with the weights learned on COCO [5 points]
# 
# Call `load_weights` as before, but add the following argument in the call to the function:
# 
# ```
# exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
#           "mrcnn_bbox", "mrcnn_mask"]
# ```

# In[37]:


# your code here
model_new.load_weights(COCO_MODEL_PATH, by_name=True,exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
          "mrcnn_bbox", "mrcnn_mask"])


# ## Train the model for 10 epochs 
# 
# Look up the documentation or code for the train function to figure out its arguments.
# 
# Pass the following additional arguments to the `train` function:
# 
#    - `layers="heads"` to only train the weights that were not pre-loaded.
#    - `learning_rate=config.LEARNING_RATE` to set the learning rate.
#    - `epochs=10`.
#    
# This will take ~10 minutes on a p2.xlarge GPU instance with 1 GPU.

# In[38]:


get_ipython().run_cell_magic('time', '', "# call to train\nmodel_new.train(dataset, dataset_val, layers='heads',  learning_rate=config.LEARNING_RATE,   epochs=10 )")


# ## Test model 

# Load the model in inference (testing) mode.

# In[39]:


# load model

class InferenceConfig(coco.CocoConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

# Recreating the model in inference mode
model_inf = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)


# Load the last trained model weights.

# In[40]:


model_path = model_inf.find_last()[1] # use the last trained weights
model_inf.load_weights(model_path, by_name=True)


# Visualize the true annotations of a random test image.

# In[41]:


image_id = random.choice(dataset_val.image_ids)
original_image, image_meta, gt_class_id, gt_bbox, gt_mask =    modellib.load_image_gt(dataset_val, inference_config, 
                           image_id, use_mini_mask=False)

log("original_image", original_image)
log("image_meta", image_meta)
log("gt_class_id", gt_bbox)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)

visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            dataset.class_names, figsize=(8, 8))


# Visualize the predicted annotations for this image

# In[42]:


# predicted annotations visualization
result = model_inf.detect([original_image], verbose=1)

p = result[0]
visualize.display_instances(original_image, p['rois'], p['masks'], p['class_ids'], 
                            dataset_val.class_names, p['scores'])

