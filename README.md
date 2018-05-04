# SSD: Single Shot MultiBox Detector

[![Build Status](https://travis-ci.org/weiliu89/caffe.svg?branch=ssd)](https://travis-ci.org/weiliu89/caffe)
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

By [Wei Liu](http://www.cs.unc.edu/~wliu/), [Dragomir Anguelov](https://www.linkedin.com/in/dragomiranguelov), [Dumitru Erhan](http://research.google.com/pubs/DumitruErhan.html), [Christian Szegedy](http://research.google.com/pubs/ChristianSzegedy.html), [Scott Reed](http://www-personal.umich.edu/~reedscot/), [Cheng-Yang Fu](http://www.cs.unc.edu/~cyfu/), [Alexander C. Berg](http://acberg.com).

### Introduction

SSD is an unified framework for object detection with a single network. You can use the code to train/evaluate a network for object detection task. For more details, please refer to our [arXiv paper](http://arxiv.org/abs/1512.02325) and our [slide](http://www.cs.unc.edu/~wliu/papers/ssd_eccv2016_slide.pdf).

<p align="center">
<img src="http://www.cs.unc.edu/~wliu/papers/ssd.png" alt="SSD Framework" width="600px">
</p>

| System | VOC2007 test *mAP* | **FPS** (Titan X) | Number of Boxes | Input resolution
|:-------|:-----:|:-------:|:-------:|:-------:|
| [Faster R-CNN (VGG16)](https://github.com/ShaoqingRen/faster_rcnn) | 73.2 | 7 | ~6000 | ~1000 x 600 |
| [YOLO (customized)](http://pjreddie.com/darknet/yolo/) | 63.4 | 45 | 98 | 448 x 448 |
| SSD300* (VGG16) | 77.2 | 46 | 8732 | 300 x 300 |
| SSD512* (VGG16) | **79.8** | 19 | 24564 | 512 x 512 |


<p align="left">
<img src="http://www.cs.unc.edu/~wliu/papers/ssd_results.png" alt="SSD results on multiple datasets" width="800px">
</p>

_Note: SSD300* and SSD512* are the latest models. Current code should reproduce these results._

### Citing SSD

Please cite SSD in your publications if it helps your research:

    @inproceedings{liu2016ssd,
      title = {{SSD}: Single Shot MultiBox Detector},
      author = {Liu, Wei and Anguelov, Dragomir and Erhan, Dumitru and Szegedy, Christian and Reed, Scott and Fu, Cheng-Yang and Berg, Alexander C.},
      booktitle = {ECCV},
      year = {2016}
    }

### Contents
1. [Prerequisites](#Prerequisites)
2. [Installation](#installation)
3. [Bug Fixes](#bug fixes)
4. [Preparation](#preparation)
5. [Train/Eval](#traineval)
6. [Models](#models)

### Note: This is installation process is only tested on Ubuntu 16.04.  

### Prerequisites
1. python2.7 Ubuntu 16.04 comes with python2.7 installed.
2. Install Protocol Buffers by google verison 2.6.1. Run the script given below.
```Shell
#! /bin/bash
wget https://github.com/google/protobuf/releases/download/v2.6.1/protobuf-2.6.1.tar.gz
tar xzf protobuf-2.6.1.tar.gz
cd protobuf-2.6.1
sudo apt-get update
sudo apt-get install build-essential
sudo ./configure
sudo make
sudo make check
sudo make install
sudo ldconfig
protoc --version
```
Note: If you have protobuf already installed, please remove that isntalled version.

3. Install anaconda for python2.7
Steps can be found [here](https://conda.io/docs/user-guide/install/linux.html)

4. Install the following following necessary libraries
```Shell
  sudo apt-get install libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
  sudo apt-get install --no-install-recommends libboost-all-dev
```

5. Install python [opencv](https://pypi.org/project/opencv-python/) module

6. Install python numpy module
```Shell
  pip install numpy
```

### Installation
1. Get the code. We will call the directory that you cloned Caffe into `$SDDC_ROOT`
  ```Shell
  git clone https://github.com/weiliu89/caffe.git
  cd caffe
  git checkout ssd
  ```

2. Install more python packets [Inspired from caffe](http://caffe.berkeleyvision.org/installation.html)
  ```Shell
  cd $HOME/python
  for req in $(cat requirements.txt); do pip install $req; done
  # Change $CAFFE_ROOT to the root directory location of repo
  export PYTHONPATH="${PYTHONPATH}:$SDDC_ROOT/python"
```

3. Build the project
  ```Shell
  make all -j8
  # Make sure to include $CAFFE_ROOT/python to your PYTHONPATH.
  make py
  make test -j8
  # (Optional but very important to see everything is installed properly)
  make runtest -j8
  make pytest -j8
  ```

### Bug Fixes
1. Common make all build fail issues can be found [here](https://github.com/BVLC/caffe/wiki/Commonly-encountered-build-issues).
2. make py command requires numpy library. Make sure the numpy library location in the Makefile.config is correct.
3. Make more the protobuf verison is 2.6.1. Can be checked by the following commands
```Shell
  protoc -version
```
4. For everything else google is your best friend. Most of these issues come from different envirnoment issues.

### Preparation
1. Download [fully convolutional reduced (atrous) VGGNet](https://gist.github.com/weiliu89/2ed6e13bfd5b57cf81d6). By default, we assume the model is stored in `$CAFFE_ROOT/models/VGGNet/`

2. Download VOC2007 and VOC2012 dataset. By default, we assume the data is stored in `$HOME/data/`
  ```Shell
  # Download the data.
  cd $HOME/data
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  # Extract the data.
  tar -xvf VOCtrainval_11-May-2012.tar
  tar -xvf VOCtrainval_06-Nov-2007.tar
  tar -xvf VOCtest_06-Nov-2007.tar
  ```

3. Create the LMDB file.
  ```Shell
  cd $SDDC_ROOT
  # Create the trainval.txt, test.txt, and test_name_size.txt in data/VOC0712/
  ./data/VOC0712/create_list.sh
  # You can modify the parameters in create_data.sh if needed.
  # It will create lmdb files for trainval and test with encoded original image:
  #   - $HOME/data/VOCdevkit/VOC0712/lmdb/VOC0712_trainval_lmdb
  #   - $HOME/data/VOCdevkit/VOC0712/lmdb/VOC0712_test_lmdb
  # and make soft links at examples/VOC0712/
  ./data/VOC0712/create_data.sh
  ```

### Train/Eval
1. Train your model and evaluate the model on the fly.
  ```Shell
  # It will create model definition files and save snapshot models in:
  #   - $CAFFE_ROOT/models/VGGNet/VOC0712/SSD_300x300/
  # and job file, log file, and the python script in:
  #   - $CAFFE_ROOT/jobs/VGGNet/VOC0712/SSD_300x300/
  # and save temporary evaluation results in:
  #   - $HOME/data/VOCdevkit/results/VOC2007/SSD_300x300/
  # It should reach 77.* mAP at 120k iterations.
  python examples/ssd/ssd_pascal.py
  ```
  If you don't have time to train your model, you can download a pre-trained model at [here](https://drive.google.com/open?id=0BzKzrI_SkD1_WVVTSmQxU0dVRzA).

2. Evaluate the most recent snapshot.
  ```Shell
  # If you would like to test a model you trained, you can do:
  python examples/ssd/score_ssd_pascal.py
  ```

3. Test your model using a webcam. Note: press <kbd>esc</kbd> to stop.
  ```Shell
  # If you would like to attach a webcam to a model you trained, you can do:
  python examples/ssd/ssd_pascal_webcam.py
  ```
  [Here](https://drive.google.com/file/d/0BzKzrI_SkD1_R09NcjM1eElLcWc/view) is a demo video of running a SSD500 model trained on [MSCOCO](http://mscoco.org) dataset.

4. Check out [`examples/ssd_detect.ipynb`](https://github.com/weiliu89/caffe/blob/ssd/examples/ssd_detect.ipynb) or [`examples/ssd/ssd_detect.cpp`](https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_detect.cpp) on how to detect objects using a SSD model. Check out [`examples/ssd/plot_detections.py`](https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/plot_detections.py) on how to plot detection results output by ssd_detect.cpp.

5. To train on other dataset, please refer to data/OTHERDATASET for more details. We currently add support for COCO and ILSVRC2016. We recommend using [`examples/ssd.ipynb`](https://github.com/weiliu89/caffe/blob/ssd/examples/ssd_detect.ipynb) to check whether the new dataset is prepared correctly.

### Models
We have provided the latest models that are trained from different datasets. To help reproduce the results in [Table 6](https://arxiv.org/pdf/1512.02325v4.pdf), most models contain a pretrained `.caffemodel` file, many `.prototxt` files, and python scripts.

1. PASCAL VOC models:
   * 07+12: [SSD300*](https://drive.google.com/open?id=0BzKzrI_SkD1_WVVTSmQxU0dVRzA), [SSD512*](https://drive.google.com/open?id=0BzKzrI_SkD1_ZDIxVHBEcUNBb2s)
   * 07++12: [SSD300*](https://drive.google.com/open?id=0BzKzrI_SkD1_WnR2T1BGVWlCZHM), [SSD512*](https://drive.google.com/open?id=0BzKzrI_SkD1_MjFjNTlnempHNWs)
   * COCO<sup>[1]</sup>: [SSD300*](https://drive.google.com/open?id=0BzKzrI_SkD1_NDlVeFJDc2tIU1k), [SSD512*](https://drive.google.com/open?id=0BzKzrI_SkD1_TW4wTC14aDdCTDQ)
   * 07+12+COCO: [SSD300*](https://drive.google.com/open?id=0BzKzrI_SkD1_UFpoU01yLS1SaG8), [SSD512*](https://drive.google.com/open?id=0BzKzrI_SkD1_X3ZXQUUtM0xNeEk)
   * 07++12+COCO: [SSD300*](https://drive.google.com/open?id=0BzKzrI_SkD1_TkFPTEQ1Z091SUE), [SSD512*](https://drive.google.com/open?id=0BzKzrI_SkD1_NVVNdWdYNEh1WTA)

2. COCO models:
   * trainval35k: [SSD300*](https://drive.google.com/open?id=0BzKzrI_SkD1_dUY1Ml9GRTFpUWc), [SSD512*](https://drive.google.com/open?id=0BzKzrI_SkD1_dlJpZHJzOXd3MTg)

3. ILSVRC models:
   * trainval1: [SSD300*](https://drive.google.com/open?id=0BzKzrI_SkD1_a2NKQ2d1d043VXM), [SSD500](https://drive.google.com/open?id=0BzKzrI_SkD1_X2ZCLVgwLTgzaTQ)

<sup>[1]</sup>We use [`examples/convert_model.ipynb`](https://github.com/weiliu89/caffe/blob/ssd/examples/convert_model.ipynb) to extract a VOC model from a pretrained COCO model.
