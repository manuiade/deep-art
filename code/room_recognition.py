import cv2
import numpy as np
from pytesseract import Output
import pytesseract
import re
import math
k=0

bounding_box =[]
img =  cv2.imread("map.png")
points_index=[]


def distance(ptA, ptB):
	return float((math.sqrt((ptA[0] - ptB[0]) ** 2 + (ptA[1] - ptB[1]) ** 2)))

ret, threshed_img = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                127, 255, cv2.THRESH_BINARY)
# find contours and get the external one

contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    if cv2.contourArea(c) < 500 and cv2.contourArea(c) > 100:
        # get the bounding rect
        [x, y, w, h]= cv2.boundingRect(c)
        #x, y, w, h= cv2.boundingRect(c)
        #print(i)
        #k+=1
        #bounding_box[k] = cv2.boundingRect(c)
        bounding_box.append([x,y,w,h])
        # draw a green rectangle to visualize the bounding rect

        #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1)
        #cv2.rectangle(img, (bounding_box[i][0], bounding_box[i][1]),
        #                    (bounding_box[i][0]+bounding_box[i][2], bounding_box[i][1]+bounding_box[i][3]), (0, 255, 0), 1)

lenght= len(bounding_box)
idx=0

for i in range(lenght):
    for j in range(lenght):
        if i != j:

            if distance((bounding_box[i][0],bounding_box[i][1]),(bounding_box[j][0],bounding_box[j][1])) < 100:
                print(distance((bounding_box[i][0],bounding_box[i][1]),(bounding_box[j][0],bounding_box[j][1])))
                if bounding_box[i][0]-bounding_box[j][0] <0 and bounding_box[i][0]-bounding_box[j][0] > -36:
                    #cv2.rectangle(img, (bounding_box[i][0], bounding_box[i][1]),
                    #                (bounding_box[j][0]+bounding_box[j][2], bounding_box[j][1]+bounding_box[j][3]), (255, 255, 0), 3)
                    points_index.append(i)
                    points_index.append(j)
                    new_img=img[bounding_box[i][1]:bounding_box[j][1]+bounding_box[j][3],bounding_box[i][0]:bounding_box[j][0]+bounding_box[j][2]]
                    cv2.imwrite('./Room/'+str(idx)+'.png',new_img) #stores the new image

                    f = open('./Room/'+str(idx)+'.txt',"w")
                    f.write(str(bounding_box[i][0])+"\n", ) ## x
                    f.write(str(bounding_box[i][1])+"\n") ## y
                    f.write(str(bounding_box[j][0]+bounding_box[j][2])+"\n") ## x + w
                    f.write(str(bounding_box[j][1]+bounding_box[j][3])+"\n") ## y + h
                    idx+=1
                    #cv2.line(img, (bounding_box[i][0], bounding_box[i][1]), (bounding_box[j][0], bounding_box[j][1]),(0, 255, 0), thickness=3)


                #else:
                     #cv2.rectangle(img, (bounding_box[j][0], bounding_box[j][1]),
                     #                (bounding_box[i][0]+bounding_box[i][2], bounding_box[i][1]+bounding_box[i][3]), (255, 0, 0), 1)
                    
                    #cv2.line(img, (bounding_box[i][0], bounding_box[i][1]), (bounding_box[j][0], bounding_box[j][1]),(0,0, 0), thickness=3)

#print(bounding_box[1]&bounding_box[0])

#print(len(contours))
#cv2.drawContours(img, contours, -1, (255, 255, 0), 1)

for i in range(lenght):
    if i not in points_index:
        #cv2.rectangle(img, (bounding_box[i][0], bounding_box[i][1]),
        #                    (bounding_box[i][0]+bounding_box[i][2], bounding_box[i][1]+bounding_box[i][3]), (0, 255, 0), 3)
        new_img = img[bounding_box[i][1]-10:bounding_box[i][1]+bounding_box[i][3]+10,bounding_box[i][0]-10:bounding_box[i][0]+bounding_box[i][2]+10]
        cv2.imwrite('./Room/'+str(idx)+'.png',new_img) #stores the new image
        f = open('./Room/'+str(idx)+'.txt',"w")
        f.write(str(bounding_box[i][0])+"\n", ) ## x
        f.write(str(bounding_box[i][1])+"\n") ## y
        f.write(str(bounding_box[i][0]+bounding_box[i][2])+"\n") ## x + w
        f.write(str(bounding_box[i][1]+bounding_box[i][3])+"\n") ## y + h
        idx+=1
Cropped_loc=cv2.imread('./Room/12.png') #the filename of cropped image
cv2.imshow("cropped",Cropped_loc)
text=pytesseract.image_to_string(Cropped_loc, lang='eng',
           config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789') #converts image characters to string
print("Number is:" ,text)

if text == str(1):
    print("sono io")

cv2.imshow("contours", img)

while True:
    key = cv2.waitKey(1)
    if key == 27: #ESC key to break
        break

cv2.destroyAllWindows()
