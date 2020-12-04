import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch 
import torch.nn as nn
from torch.autograd import Variable
from darknet_utils import *
import glob
import ntpath
import pandas as pd

#### Detection utils ####

def prep_image(img, inp_dim):
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def detect(model,num_classes,img,im_dim, CUDA,confidence,nms_thresh,inp_dim):
    with torch.no_grad():   
        detected = model(Variable(img), CUDA)

    detected = write_results(detected, confidence, num_classes,  nms_conf = nms_thresh)
    if type(detected) == int:
        return None, None      
    im_dim = im_dim.repeat(detected.size(0), 1)
    scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)  
    detected[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
    detected[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2 
    detected[:,1:5] /= scaling_factor
    
    return detected, im_dim

#### Painting Rectification utils ####

def order_points(pts):
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    width_factor = (br[1] - tr[1])/(bl[1]-tl[1])
    if width_factor < 1:
        width_factor = 1/width_factor
    maxWidth = int(maxWidth * width_factor)    
    height_factor = (br[0] - bl[0])/(tr[0]-tl[0])
    if height_factor < 1:
        height_factor = 1/height_factor
    maxHeight = int(maxHeight * height_factor)
    dst = np.array([[0, 0],[maxWidth - 1, 0],[maxWidth - 1, maxHeight - 1],[0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def rectify_image(image):
    thresh = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.medianBlur(thresh,5)
    thresh= cv2.GaussianBlur(thresh, (5,5), 15)
    thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11, 2)
    _, thresh = cv2.threshold(thresh, 175, 255, cv2.THRESH_OTSU)
    thresh= cv2.GaussianBlur(thresh, (5,5), 15)
    #thresh = cv2.dilate(thresh, None, iterations=3)
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max([c for c in cnts], key=lambda x: cv2.contourArea(x))
    hull = cv2.convexHull(cnt)
    hull = [tuple(p[0]) for p in hull]

    tr1 = max(hull, key=lambda x: 2*x[0] - x[1])
    tl1 = min(hull, key=lambda x: 2*x[0] + x[1])
    br1 = max(hull, key=lambda x: 2*x[0] + x[1])
    bl1 = min(hull, key=lambda x: 2*x[0] - x[1])

    tr2 = max(hull, key=lambda x: x[0] - 2*x[1])
    tl2 = min(hull, key=lambda x: x[0] + 2*x[1])
    br2 = max(hull, key=lambda x: x[0] + 2*x[1])
    bl2 = min(hull, key=lambda x: x[0] - 2*x[1])

    tr = (max(tr1[0],tr2[0]),min(tr1[1],tr2[1]))
    tl = (min(tl1[0],tl2[0]),min(tl1[1],tl2[1]))
    br = (max(br1[0],br2[0]),max(br1[1],br2[1]))
    bl = (min(bl1[0],bl2[0]),max(bl1[1],bl2[1]))

    pts = np.array([tr,tl,br,bl], dtype = "float32")
    warped = four_point_transform(image, pts)

    return warped

#### Painting Retrieval utils ####

def retrieval(rectified,images,img_list,orb,dataframe):

    out_perc = []
    out_filename = []
    out_title = []   
    out_room = []
    rectified = cv2.cvtColor(rectified, cv2.COLOR_RGB2GRAY)
    kp1, des1 = orb.detectAndCompute(rectified,None)
    for j, im in enumerate(images):
        im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        good_points = []
        kp2, des2 = orb.detectAndCompute(im,None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = bf.knnMatch(des1,des2, k=2)
        for m,n in matches:
            if m.distance < 0.8*n.distance:
                good_points.append(m)
        number_keypoints = 0
        if len(kp1) <= len(kp2):
            number_keypoints = len(kp1)
        else:
            number_keypoints = len(kp2)
        if number_keypoints == 0:
            percentage = 0
        else:
            percentage = len(good_points) / (number_keypoints) * 100
        if percentage > 2:
            #print("similarity: " + str(percentage) + " for image " + img_list[j])
            title = dataframe.loc[dataframe['Image'] == img_list[j]]
            room = dataframe.loc[dataframe['Image'] == img_list[j]]
            #print(room["Room"])
            #print(detected["Title"])
            #results = cv2.drawMatches(rectified, kp1, im, kp2, good_points, None)
            out_perc.append(percentage)
            out_filename.append(img_list[j])
            out_title.append(title["Title"].values[0])
            out_room.append(room["Room"].values[0])
    out = sorted(zip(out_perc, out_filename, out_title,out_room), key=lambda x: x[0], reverse=True)
    
    if len(out) != 0:
        return out[0:5]
    else:
        return None



#### Painting database utils ####

def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def sort(lst): 
    lst.sort(key = str) 
    return lst 
 
def load_database():
    img_list = glob.glob("data/database/paintings_db/*.png")
    img_list = sort(img_list) 
    
    images = [cv2.imread(img) for img in img_list ]
    img_list = [path_leaf(path) for path in img_list]
    
    dataframe = pd.read_csv("data/database/data.csv")
    ntpath.basename("a/b/c")

    return images, img_list, dataframe


#### Bounding boxes utils ####

def calculate_bb(x1,x2,y1,y2,f_height,f_width,crop_factor):
    if int(x1) > crop_factor:   
        x1 = int(x1) - crop_factor
    elif int(x1) < 0:
        x1 = 0
    else:
        x1 = int(x1)

    if int(x2) < f_width - crop_factor:
        box_w = int(x2 - x1) + crop_factor
    else:
        box_w = int(x2 - x1)

    if int(y1) >  crop_factor:
        y1 = int(y1) - crop_factor
    elif int(y1) < 0:
        y1 = 0
    else:
        y1 = int(y1)

    if int(y2) < f_height - crop_factor:
        box_h = int(y2 - y1) + crop_factor
    else:
        box_h = int(y2 - y1)

    return x1, box_w, y1, box_h

def draw_bb(colors,label,obj_id,x1,y1,box_w,box_h,img):
    color = colors[int(obj_id) % len(colors)]
    color = [i * 255 for i in color]
    cv2.rectangle(img, (x1,y1), (x1+box_w,y1+box_h),color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    cv2.rectangle(img, (x1, y1), (x1+t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
    cv2.putText(img, label , (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)


#### Person localization utils ####
def room_Map(room):
    list =[]

    f = open("data/room_number/"+str(room)+".txt","r")
    img = cv2.imread("data/map.png")
    lines =  f.readlines()

    for line in lines:
        list.append(int(line))

    cv2.rectangle(img, (list[0], list[1]),
                 (list[2], list[3]), (255, 255, 0), 1)

    cv2.putText(img,"Here", (list[0] -20 ,list[1]-10),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("room", img)
  
    key = cv2.waitKey(2)
    if key & 0xFF == ord('w'):
        return
