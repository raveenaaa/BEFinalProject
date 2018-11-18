import cv2
import numpy as np
import os


directory = 'E:\\Project\\Dataset\\subtracted\\2\\'

for img in os.listdir(directory):
  image = img
  img = cv2.imread(str(directory)+str(img),cv2.IMREAD_COLOR)    
  mask = np.zeros(img.shape[:2],np.uint8)
  height, width = img.shape[:2]  
  bgm =np.zeros((1,65),np.float64)
  fgm =np.zeros((1,65),np.float64)

  rect=(5,5,width-5,height-5)
  try:      
    cv2.grabCut(img,mask,rect,bgm,fgm,10,cv2.GC_INIT_WITH_RECT)
    cv2.grabCut(img,mask,rect,bgm,fgm,10,cv2.GC_INIT_WITH_MASK)
  except:
    cv2.grabCut(img,mask,rect,bgm,fgm,10,cv2.GC_INIT_WITH_RECT) 
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img =img*mask2[:,:,np.newaxis]
    cv2.imwrite('E:\\Project\\Dataset\\subtracted\\try\\'+str(image),img)
    continue
  else: 
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img =img*mask2[:,:,np.newaxis]
    cv2.imwrite('E:\\Project\\Dataset\\subtracted\\try\\'+str(image),img)
  
