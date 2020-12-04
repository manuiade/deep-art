from __future__ import division

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)

import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import pandas as pd
import random 
import pickle as pkl
import argparse
import ntpath
import matplotlib.pyplot as plt
import glob

from darknet_utils import *
from darknet import Darknet
from sort import *
from pipeline_utils import *



class ObjectModel():
    def __init__(self, net_numclasses, name):
        super(ObjectModel, self).__init__()
        self.net_numclasses = net_numclasses
        self.names = load_classes("cfg/coco/{name}.names".format(name=name))
        self.cfgfile = "cfg/net/{name}.cfg".format(name=name)
        self.weights = "cfg/weights/{name}.weights".format(name=name)
        self.model = Darknet(self.cfgfile)
        self.model.load_weights(self.weights)
        #Prepare trackers
        self.tracker = Sort()
    
def arg_parse():
    parser = argparse.ArgumentParser(description='VCS Project')
   
    parser.add_argument("--video", dest = 'video', help = 
                        "Video to run detection upon",
                        default = "video.avi", type = str)
    parser.add_argument("--skip-frames", dest = 'skip', help = 
                        "Number of frames advance after each cycle",
                        default = 1, type = int)


    return parser.parse_args()

if __name__ == '__main__':

    #Set CUDA
    CUDA = torch.cuda.is_available()
    torch.cuda.empty_cache()

    #Parse parameters
    args = arg_parse()
    videofile = args.video
    skip = args.skip

    #YOLO parameters 
    confidence = 0.3
    nms_thresh = 0.4
    resolution = 416

    #Load parameters for painting detection
    print("Loading network for painting detection.....")
    painting = ObjectModel(1,"painting")
    print("Network successfully loaded")
    
    #Load parameters for person detection
    print("Loading network for people detection.....")
    person = ObjectModel(80,"person")
    print("Network successfully loaded")
    
    #Prepare color palette
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]
    
    #Check if resolution is ok
    painting.model.net_info["height"] = resolution
    person.model.net_info["height"] = resolution
    inp_dim = int(painting.model.net_info["height"])
    assert inp_dim % 32 == 0, 'Resolution should be multiple of 32'
    assert inp_dim > 32, 'Resolution should be > 32'

    #Prepare model for evaluation mode
    print("Preparing models for evaluation.....")
    if CUDA:
        painting.model.cuda()
        person.model.cuda()
    painting.model.eval()
    person.model.eval()
    print("Done.")

    print("Loading painting database..")
    images, img_list, dataframe = load_database()
    print("Database loaded succesfully")

    #Open video file
    print("Opening video file...")
    cap = cv2.VideoCapture(videofile)
    assert cap.isOpened(), 'Cannot capture source'
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = 0
    print("Done.")
    

    #ORB detector for paiting retrieval  
    orb = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE)

    #Crop value to correct image cropping
    crop_factor = 10
    
    print("Starting pipeline..")

    while cap.isOpened():
        cap.set(1, frames)
        room = [0,"","",0]
        print("####################################################")
        #input("Press Enter to continue...")
        print("###### Processing frame {frames} of {length} ######".format(frames=frames,length=length))   
        ret, frame = cap.read()
        if ret:          
            #Prepare frame for YOLO detection
            f_height, f_width = frame.shape[0:2]
            img, orig_im, dim = prep_image(frame, inp_dim)         
            im_dim = torch.FloatTensor(dim).repeat(1,2)                                  
            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()

            curr_frame = orig_im.copy()

            ########## 1) --> PAINTING DETECTION ##########
            painting_detect, im_dim_painting = detect(painting.model,painting.net_numclasses,img,im_dim,CUDA,confidence,nms_thresh,inp_dim)
            painting_tracked = None

            if painting_detect is not None:
                for i in range(painting_detect.shape[0]):
                    painting_detect[i, [1,3]] = torch.clamp(painting_detect[i, [1,3]], 0.0, im_dim_painting[i,0])
                    painting_detect[i, [2,4]] = torch.clamp(painting_detect[i, [2,4]], 0.0, im_dim_painting[i,1])
                painting_detect = painting_detect[:,1:]
                #Tracking paintings between frames
                painting_tracked = painting.tracker.update(painting_detect.cpu())
                index = 0
                
                #Operations for each painting found
                for x1, y1, x2, y2, obj_id, cls_pred in painting_tracked:
                    x1, box_w, y1, box_h = calculate_bb(x1,x2,y1,y2,f_height,f_width,crop_factor)                   
                    label = "{0}".format(painting.names[int(painting_detect[index,-1])]) + "-" + str(int(obj_id))
                    print("### Detected {label} ###".format(label=label))
                    print("Bounding boxes for {label} -> x: {x1}, y: {y1}, w: {box_w}, h: {box_h}".format(label=label,x1=x1,y1=y1,box_w=box_w,box_h=box_h))                    
                    cropped = curr_frame[y1:y1+box_h,x1:x1+box_w]                   
                    cv2.imwrite('data/output/frame_{frames}_out{index}.png'.format(frames=frames,index=int(obj_id)),cropped)
                    
                    ########## 2) --> PAINTING RECTIFICATION ##########
                    rectified = rectify_image(cropped)
                    if rectified is not None:
                        print("Saving rectified image for {label} to data/output/".format(label=label))
                        cv2.imwrite('data/output/frame_{frames}_out{index}_Rectified.png'.format(frames=frames,index=int(obj_id)),rectified)

                        ########## 3) --> PAINTING RETRIEVAL ##########           
                        room_tmp = retrieval(rectified,images,img_list,orb,dataframe)                       
                        #Keep the most probable room based on most accurate painting retrieval
                        if room_tmp is not None:
                            ##Print five highest percentage
                            print("Highest similarities for {label}:".format(label=label))
                            for i in range(len(room_tmp)):
                                if i > 4:
                                    break
                                print("{index}) -> similarity of {sim:.2f}% for file {filename} -> {title}".format(index=i+1, sim=room_tmp[i][0],filename=room_tmp[i][1], title=room_tmp[i][2]))
                                
                            if room_tmp[0][0] > room[0]:
                                room = room_tmp[0]

                    index+=1
                    draw_bb(colors,label,obj_id,x1,y1,box_w,box_h,orig_im)
                    
                    
            ########## 4) --> PEOPLE DETECTION ##########
            person_detect, im_dim_person = detect(person.model,person.net_numclasses,img,im_dim,CUDA,confidence,nms_thresh,inp_dim)
            if person_detect is not None:
                person_detect = person_detect[person_detect[:,7] == 0]
                cut_shape = person_detect.shape[0]

                #Discard person detected inside painting bounding boxes
                if painting_detect is not None:
                    for i in range(person_detect.shape[0]):
                        inside = False
                        for j in range(painting_detect.shape[0]):
                            if person_detect[i,1] >= painting_detect[j,0] - 50 and person_detect[i,2] >= painting_detect[j,1] -50  and person_detect[i,3] <= painting_detect[j,2] + 50 and person_detect[i,4] <= painting_detect[j,3] + 50:
                                inside = True
                                break
                        if inside == False:
                            person_detect = torch.cat((person_detect, person_detect[i,:].unsqueeze(0)), 0)

                if person_detect.shape[0] > cut_shape:
                    person_detect = person_detect[cut_shape:,:]
                else:
                    person_detect = None

                person_tracked = None

            if person_detect is not None:
                for i in range(person_detect.shape[0]):
                    person_detect[i, [1,3]] = torch.clamp(person_detect[i, [1,3]], 0.0, im_dim_person[i,0])
                    person_detect[i, [2,4]] = torch.clamp(person_detect[i, [2,4]], 0.0, im_dim_person[i,1])
                person_detect = person_detect[:,1:]
                #Tracking paintings between frames
                person_tracked = person.tracker.update(person_detect.cpu())
                index = 0

                #Operations for each person found
                for x1,y1,x2,y2,obj_id, cls_pred in person_tracked:                  
                    x1, box_w, y1, box_h = calculate_bb(x1,x2,y1,y2,f_height,f_width,crop_factor)                  
                    label = "{0}".format(person.names[int(person_detect[index,-1])]) + "-" + str(int(obj_id))
                    print("### Detected {label} ###".format(label=label))
                    print("Bounding boxes for {label} -> x: {x1}, y: {y1}, w: {box_w}, h: {box_h}".format(label=label,x1=x1,y1=y1,box_w=box_w,box_h=box_h))

                    ########## 5) --> PEOPLE LOCALIZATION ##########
                    if room[3] > 0:
                        print("Detected person {obj_id} in room {room_number}".format(obj_id=obj_id,room_number=room[3]))
                        room_Map(room[3])
                    
                    else:
                        print("Not enough information to localize person {obj_id}".format(obj_id=obj_id))
                        cv2.destroyWindow("room")
                    index+=1
                    draw_bb(colors,label,obj_id,x1,y1,box_w,box_h,orig_im)

            #Show current frame with bounding boxes for both paintings and people
            cv2.imshow("frame", orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += skip
            
        else:
            if frames + skip >= length:
                print("No more frames to read, press Enter to terminate program")
                input()
                break
            else:
                print("Cannot read current frame, skipping..")
                frames += skip
        
    torch.cuda.empty_cache()





    
    

    
