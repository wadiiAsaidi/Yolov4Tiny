from tool.darknet2pytorch import Darknet
#---------------------------------------load model
cfgfile='yolov4-tiny_training.cfg'
weightfile='yolov4-tiny_training_last.weights'

model = Darknet(cfgfile)
model.print_network()
model.load_weights(weightfile)


print('Loading weights from %s... Done!' % (weightfile))
#---------------------------------------
import os
import PIL
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch.utils.data.sampler as sampler
from torch import nn, optim
import torch.nn.functional as F
from box import post_processing ,boxes_cv2
import time
frame=cv2.imread('00001.jpg')
start =time.time()

with torch.no_grad():

        image=np.copy(frame)
        sized = cv2.resize(image, (608, 608))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(sized.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)

        image = torch.autograd.Variable(image)
        model.eval()
        output = model(image)
        
        boxes=post_processing(frame,0.4, 0.6,output)

        list=boxes_cv2(frame,boxes[0]) 

#-------------------------------------------
end =time.time()


print(end-start)