import random as r
import numpy as np
import cv2
from PIL import Image
import os

#from matplotlib import pyplot as plt

dirname = 'G:\\Backup\\Dataset\\subtracted\\average\\try\\'
dire='G:\\Backup\\Dataset\\subtracted\\average\\1\\'
for img in os.listdir(dire):
    image = img   
    count=10 #no of iteration the images is processed
    img =cv2.imread(str(dire)+image)
    rows,cols,_=img.shape
    while count!=0:
        a=r.randint(1,7)
        if a==1:
            x=r.randint(1,350)
            #M = cv2.getRotationMatrix2D((cols/2,rows/2),x,1)
            img =Image.open(str(dire)+str(image))
            dst =img.rotate(x)
            
            #dst = cv2.warpAffine(img,M,(cols,rows))
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            count=count-1
            dst.save(os.path.join(dirname,str(image)))            
            
        elif a==2:
            x=r.randint(-10,5)
            y=r.randint(-5,5)
            M=np.float32([[1,0,x],[0,1,y]])
            dst = cv2.warpAffine(img,M,(cols,rows))           
            count-=1
            cv2.imwrite(os.path.join(dirname,str(image)),dst)           
        elif a==3:
            continue           
        elif a==4:
            blur= cv2.blur(img,(5,5))           
            #img = blur
            count-=1
            cv2.imwrite(os.path.join(dirname,image),blur)           
        elif a==5:
            gblur = cv2.GaussianBlur(img,(5,5),0)
            #img = gblur
          
            count-=1
            cv2.imwrite(os.path.join(dirname,str(image)),gblur)
           
        elif a==6:
            mblur = cv2.medianBlur(img,5)
            #img = mblur
            count-=1
            cv2.imwrite(os.path.join(dirname,str(image)),mblur)
                    
        elif a==7:
            bfilter= cv2.bilateralFilter(img,9,75,75)
            #img = bfilter
            count-=1
            cv2.imwrite(os.path.join(dirname,str(image) ),bfilter)
         
        
            
        

