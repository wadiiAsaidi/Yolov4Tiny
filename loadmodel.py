
from box import boxes_cv2,post_processing
import torch 
from CnnWithStn import TrafficSignNet
import torch.nn.functional as F
import cv2
import numpy as np
import torchvision.transforms as transforms
#device = torch.device("cuda")
TrafficSignNet = TrafficSignNet()#.to(device)
check_point = torch.load('model.pt',map_location=torch.device('cpu'))
TrafficSignNet.load_state_dict(check_point)
TrafficSignNet.eval()
import matplotlib.pyplot as plt
from tool.darknet2pytorch import Darknet
import time
cfgfile='yolov4-tiny_training.cfg'
weightfile='yolov4-tiny_training_last.weights'
model = Darknet(cfgfile)
model.print_network()
model.load_weights(weightfile)
model.eval()
print('Loading weights from %s... Done!' % (weightfile))


#-------------------
start=time.time() 
frame=cv2.imread('00001.jpg')
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
        ts=[]  
        for tuple in list:  
          x1=tuple[0]
          y1=tuple[1]
          x2=tuple[2]
          y2=tuple[3]

          ts.append(frame[y1:y2,x1:x2])
        print(len(ts))
end=time.time()         
print(end-start)