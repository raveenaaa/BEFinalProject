# resize an image using the PIL image library
# free from:  http://www.pythonware.com/products/pil/index.htm
# tested with Python24        vegaseat     11oct2005

from PIL import Image

for i in range(114,128):
    imageFile="a ("+str(i+1)+").jpg"
    im1 = Image.open(imageFile)
    # adjust width and height to your needs
    width = 512
    height = 512
    im2 = im1.resize((width, height), Image.ANTIALIAS)      # use nearest neighbour
    ext = ".jpg"
    im2.save(str(i+1) + ext)
 


# optional image viewer ...
# image viewer  i_view32.exe   free download from:  http://www.irfanview.com/
# avoids the many huge bitmap files generated by PIL's show()
#import os
#os.system("d:/python24/i_view32.exe %s" % "BILINEAR.jpg")
