import numpy as np
import cv2
import os
import pytesseract as tess

tess.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'

##def loadImg(path="drive-download"):  #function to parse through all images
##    temp=os.listdir(path)            #temp has the address of subfolder in which images are stored
##    images=[]                        # images is a list which contains the final path of the images
##    for i in temp:                   #i has name of 1st images in the subfolder
##        images.append(os.path.join(path,i)) #Final path= path/image.jpg
##   
##    return images
##
##filenames=loadImg()
##images=[]
##for f in filenames:
##    images.append(cv2.imread(f,cv2.IMREAD_UNCHANGED)) #cv2.IMREAD_UNCHANGED wouldnt change the image at all,
##                                                      #It wouldnt even change RGB configuration and all
##
##
##for i in images:
    




img=cv2.imread("drive-download/txt_mudit_b8_1_513.jpg")
if img.shape[1]>810 and img.shape[0]>600:
    img=cv2.resize(img,(800,600),interpolation=cv2.INTER_AREA)

gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
adapt=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)

kernel=np.ones((2,2),np.uint8)
adapclose=cv2.morphologyEx(adapt,cv2.MORPH_CLOSE,kernel)

blur=cv2.blur(adapclose,(2,2))
gaussianBlur=cv2.GaussianBlur(adapclose,(3,3),0)
bilateralBlur=cv2.bilateralFilter(adapclose,9,75,75)

text=tess.image_to_string(gaussianBlur)
print(text)

####cv2.imshow("AdaptiveThreshold",adapt)
####cv2.imshow("adapclose",adapclose)
cv2.imshow("gaussianBlur",gaussianBlur)
##cv2.imshow("blur",blur)
##cv2.imshow("bilateralBlur",bilateralBlur)
cv2.waitKey(0)
cv2.destroyAllWindows()






