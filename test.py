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

#---------------------------------------load model
cfgfile='yolov4-tiny_training.cfg'
weightfile='yolov4-tiny_training_last.weights'
model = Darknet(cfgfile)
#model.print_network()
model.load_weights(weightfile)
#print('Loading weights from %s... Done!' % (weightfile))
#---------------------------------------

import pandas as pd
import time
start =time.time()
video = cv2.VideoCapture('result (56).mp4')
# Writer that will be used to write processed frames
writer = None

# Variables for spatial dimensions of the frames
h, w = None, None

f=0
while True:
    # Capturing frames one-by-one
    ret, frame = video.read()
    f+=1
    # If the frame was not retrieved
    if not ret:
        break
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
              for i in range(len(ts)):

                #print(ts[i].shape) #  (16, 16, 3)  (41, 43, 3)
                w=ts[i].shape[0]
                h=ts[i].shape[1]
                dif=w-h
                if  (dif<5 and (w in range(15,42) and (h in range(15,42))) ) :

                      transformer=transforms.ToTensor()
                      tensor=transformer(ts[i])
                      resize_tensor=transforms.Resize(size=(32,32))
                      resized_tensor=resize_tensor(tensor).unsqueeze(1)
                      
                      #print(resized_tensor.shape[0])

                      TrafficSignNet=TrafficSignNet(resized_tensor)
                      labels_df = pd.read_csv('prohibitory.csv')
                      co,pred = torch.max(TrafficSignNet, 1)
                      #print(pred[0].item())
                      #print(labels_df['SignName'][str(pred[0].item())])
                      plt.imshow(ts[i])
                      plt.show()

end=time.time()
print(end-start)
print(f)