# Hand Detection and Orientation Estimation
This project utilizes a modified MobileNet in company with the [SSD](https://github.com/weiliu89/caffe/tree/ssd) framework to achieve a robust and fast detection of hand location and orientation. 
Our implementation is adapted from [the PyTorch version of SSD](https://github.com/amdegroot/ssd.pytorch) and [MobileNet](https://github.com/ruotianluo/pytorch-mobilenet-from-tf).

[Original repo](https://github.com/yangli18/hand_detection.git)

<img src="https://github.com/yangli18/hand_detection/blob/master/data/results/demo/010174_hand.svg" height=226><img src="https://github.com/yangli18/hand_detection/blob/master/data/results/demo/010061_hand.svg" height=226><img src="https://github.com/yangli18/hand_detection/blob/master/data/results/demo/010210_hand.svg" height=226>

### Contents
1. [Preparation](#preparation)
2. [Training](#training)
3. [Evaluation](#Evaluation)


### Preparation

What to install:
- PyTorch (tested 1.7)
- torchvision
- opencv (3.x or 4.x is all acceptable)


### Training

Train the detection model on the Oxford hand dataset. 
```Shell
python train.py 2>&1 | tee log/train.log
```
* Trained with data augmentation v2, our detection model reaches over **86%** average precision (AP) on the Oxford hand dataset. We provide the trained model in the `weights` dir.
* For the convenience of training, a pre-trained MobileNet is put in the `weights` dir. 
You can also download it from [here](https://github.com/ruotianluo/pytorch-mobilenet-from-tf).
 

### Evaluation

1. Evaluate the trained detection model.
    ```Shell
    python eval.py --trained_model weights/ssd_new_mobilenet_FFA.pth --version ssd_new_mobilenet_FFA
    ```
    * ***Note***: For a fair comparison, the evaluation code of the Oxford hand dataset should be used to get the exact mAP (mean Average Precision) of hand detection. 
    The above command should return a similar result, but not exactly the same.
    
2. Evaluate the average detection time.
    ```Shell
    python eval_speed.py --trained_model weights/ssd_new_mobilenet_FFA.pth --version ssd_new_mobilenet_FFA
    ```   

### Interface
```
from interface import SSD_Interface

# Load model weight
model = SSD_Interface(weight_path)

# Simple calling warper by process function
output = model.process("data/sample/input.jpg")

import json
json.dumps(output, indent=4)
```
