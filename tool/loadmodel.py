
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.transforms as transforms
from tool.darknet2pytorch import Darknet
cfgfile='yolov4-tiny_training.cfg'
weightfile='yolov4-tiny_training_last.weights'
model = Darknet(cfgfile)
model.print_network()
model.load_weights(weightfile)
print('Loading weights from %s... Done!' % (weightfile))