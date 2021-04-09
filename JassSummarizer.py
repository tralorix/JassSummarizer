from enum import Enum
from PIL import Image
from IPython.core.display import display, HTML
import pandas as pd
import numpy as np
import cv2                                                              # computer vision python library see README.md dependencies
import io
import base64
import skimage.color
import skimage.filters
import skimage.io

# Identifies the type of card (0-3) and includes also the mode of playinf (0-5)
Jassmode=Enum("Jassmode","Eicheln,Schellen,Schilten,Rosen,Topdown,Downup",start=0)

# Identifies the names (value) of the cards
JassCN=Enum("JassCN","N6,N7,N8,N9,Banner,Bauer,Ober,Koenig,Ass",start=0)

# Identifies each available Card
JassCN2=Enum("JassCN2","A0,A1,A2,A3,A4,A5,A6,A7,A8,B0,B1,B2,B3,B4,B5,B6,B7,B8,C0,C1,C2,C3,C4,C5,C6,C7,C8,D0,D1,D2,D3,D4,D5,D6,D7,D8",start=0)

# stores the values the cards have depending on their node they run
JassRegular=[0,0,0,0,10,2,3,4,11]
JassTrumpf=[0,0,0,14,10,20,3,4,11]
JassTopdown=[0,0,8,0,10,2,3,4,11]
JassDownup=[11,7,8,0,10,2,3,4,0]

# function to get the name by cardId
def jassCardName(cardId):
    name=Jassmode(ord(cardId[0])-65).name                               # A=0, B=1,C=2,D=3 Ascii code-65 for the first char
    typ=JassCN(ord(cardId[1])-48).name                                  # Ascii code-48 for the 2nd char to get the typ
    return f"{name} {typ}"
                                                                        
# short routine for image scaling
def scaleImage(img,scaleFactor):
    width  = int(img.shape[1] * scaleFactor / 100)
    height = int(img.shape[0] * scaleFactor / 100)
    dim    = (width, height)  
    return cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

# highlighter within pandas tables
def highlight_min(data, color='yellow'):
    attr = 'background-color: {}'.format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
        is_min = data == data.min()
        return [attr if v else '' for v in is_min]
    else:  # from .apply(axis=None)
        is_min = data == data.min().min()
        return pd.DataFrame(np.where(is_min, attr, ''),index=data.index, columns=data.columns)

# https://www.mathworks.com/help/matlab/ref/rgb2gray.html
def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

# Main routine to analyze the image
# 1st return parameter image Original
# 2nd return parameter mask
# 3rd return parameter image with countour drawn
# 4th originalmaked image baut with full black background
# 5th croped imaged^with largest boundingbox
# boundingbox dimension datas (x,y,w,h)
def analyzeScan(frame, kThreshold = 0.4):
    frameOrg=np.copy(frame)                                             # make a copy from input image
    x=y=w=h=0                                                           # initialize values, will be all zero if no contours where found
    src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                  # opencv prefered color channel BlueGreenRed converting to grayscale before thresholding
    #blur = cv2.blur(src_gray,(5,5))                                    # bluring the image with opencv lib which was not as good as the next line
    #cv2.GaussianBlur(src_gray,(5,5),0)                                 # not verified yet, take next line as preferred
    blur = skimage.filters.gaussian(src_gray, sigma=2)                  # <== would be nice to have a solution in opencv
    mask = blur > kThreshold                                            # apply threshold
    
    sel = np.zeros_like(frame)                                          # generating black image in the size of the input image
    sel[mask] = frame[mask]                                             # the frame with the image cut out by the mask goes on top of the black image
    (thresh, blackAndWhiteImage) = cv2.threshold(sel, 0, 255, cv2.THRESH_BINARY)
    mT=cv2.cvtColor(sel, cv2.COLOR_BGR2GRAY)                            # converting to gray 
    # opencv function to find countours on the grayed picture , RETR_EXTERNAL option select only the outer contours see 
    # https://docs.opencv.org/3.4/d9/d8b/tutorial_py_contours_hierarchy.html
    contours, hierarchy=cv2.findContours(mT, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE )
    mask = np.full(frame.shape[0:2],255, np.uint8)                      # build mask in white coloe
    cv2.drawContours(mask, contours,-1, (0,0,0),thickness=cv2.FILLED)   # build new mask with contour information in black color    
    res = cv2.bitwise_and(frame,frame,mask = 255 - mask)                # combine new mask with frameData

    mask2 = 255 - mask
    contours, hierarchy = cv2.findContours(mask2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0,255,0), 3)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    if len(contour_sizes)!=0:                                           # if no countour is there/found, we do not have to find the bigest available contur/boundingbox
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]     # generate bounding box out of available contours
        x,y,w,h = cv2.boundingRect(biggest_contour)                     # store size of boundingbox for later use
        img_rect = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)     # resize to size of boundingbox
    else:
        img_rect =frame                                                 # in case of no countours found use original image data 
        
    crop_img = res[y:y+h, x:x+w]
    d=[x,y,w,h]                                                         # store dimensions into one parameter
    return frameOrg,mask,img_rect,res,crop_img,d                        # return all analyzed image data and dimenion

# function to rotate an image by degree
def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

# Inline Image generation with html
def display_imageTable(data):
    html = "<table>"
    for row in data:
        html += "<tr>"
        for field in row:
            buff = io.BytesIO()
            Image.fromarray(field).save(buff, format="JPEG")
            new_image_string = base64.b64encode(buff.getvalue()).decode("utf-8")
            html += f'<td><img src="data:image/jpeg;base64,{new_image_string}"></td>'
        html += "</tr>"
    html += "</table>"
    display(HTML(html))
    
# routine to fill black pixels with randomzied colors
def backgroundDiffuser(imgArray,seed=0):
    if seed!=0:
        np.random.seed(seed)
    for x in range(imgArray.shape[0]):
        for y in range(imgArray.shape[1]):
            if imgArray[x,y].sum() ==0:
                imgArray[x,y][0]=int(np.random.random()*255)
                imgArray[x,y][1]=int(np.random.random()*255)
                imgArray[x,y][2]=int(np.random.random()*255)
    return imgArray

# routine to fill black pixels with randomzied colors
def backgroundDiffuser2(imgArray,seed=0):
    if seed!=0:
        np.random.seed(seed)
    for x in range(imgArray.shape[0]):
        for y in range(imgArray.shape[1]):
            if imgArray[x,y].sum() ==0:
                imgArray[x,y][0]=int(np.random.random()*128)+64
                imgArray[x,y][1]=int(np.random.random()*128)+64
                imgArray[x,y][2]=int(np.random.random()*128)+64
    return imgArray

# routine to fill black pixels with randomzied colors
def backgroundDiffuser3(imgArray,seed=0):
    if seed!=0:
        np.random.seed(seed)
    for x in range(imgArray.shape[0]):
        for y in range(imgArray.shape[1]):
            if imgArray[x,y].sum() ==0:
                imgArray[x,y][0]=255-int(np.random.random()*36)
                imgArray[x,y][1]=255-int(np.random.random()*36)
                imgArray[x,y][2]=255-int(np.random.random()*36)
    return imgArray

# takes the cardId and the current Jassmode to caluclate score
def calculateScore(card, mode):
    cardValue=int(card[-1])
    if mode == Jassmode.Topdown:
        return JassTopdown[cardValue]        
    elif mode == Jassmode.Downup:
        return JassDownup[cardValue]
    else:
        cardType=Jassmode(ord(card[0])-65).name
        if cardType== mode.name:
            return JassTrumpf[cardValue]
    return JassRegular[cardValue]
