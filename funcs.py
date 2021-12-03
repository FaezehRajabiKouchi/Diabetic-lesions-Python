import math
import numpy as np
from skimage.transform import rescale
from skimage import segmentation
import skimage
import matplotlib.pyplot as plt


def resizeretina ( retinaRGB, x, y ):
    # Resize an RGB image of retina
    a = np.shape(retinaRGB)[0]
    b = np.shape(retinaRGB)[1]
    scaleFactor = (math.sqrt(x * y / (a * b)), math.sqrt(x * y / (a * b)),1)
    retinaRGB = rescale(retinaRGB, scaleFactor, preserve_range= True).astype(np.uint8)
    return retinaRGB

def superpixels(im,N):
    L = segmentation.slic(im, n_segments=N, start_label = 1)
    NumLabels = L.max()
    
    return L, NumLabels

def label2idx(label_arr):
    return [np.where(label_arr.ravel() == i)[0]
        for i in range(1, np.max(label_arr) + 1)]


def Regionprops(Mask, IG, mr):
    while True:
        label_img = skimage.measure.label(Mask)
        s = skimage.measure.regionprops_table(label_img, properties=['centroid', 'area', 'bbox'])
        ar = s['area']
        idM = np.argmax(ar) 
        BC = [s['bbox-2'][idM]-s['bbox-0'][idM],s['bbox-3'][idM]-s['bbox-1'][idM]]
        SS = BC[0]*BC[1]
        n, m = np.shape(Mask)
        S = n*m
        nm = np.count_nonzero(Mask)
        if SS>27000 or nm>61714:
            mr = mr + 0.05
            Mask = IG> mr*IG.max()
            
        else:
            break
       
    se = skimage.morphology.disk(1)
    Mask = skimage.morphology.opening(Mask, se)
    se = skimage.morphology.disk(2)
    Mask = skimage.morphology.closing(skimage.morphology.dilation(Mask, se), se)
    label_img = skimage.measure.label(Mask, connectivity=Mask.ndim)
    s = skimage.measure.regionprops_table(label_img, properties=['centroid', 'area', 'bbox',
                                                                 'convex_area','extent', 'coords'])
    return Mask, s
    
    


def regiongrowing(I,x,y,reg_maxdist):
# % This function performs "region growing" in an image from a specified
# % seedpoint (x,y)
# %
# % J = regiongrowing(I,x,y,t) 
# % 
# % I : input image 
# % J : logical output image of region
# % x,y : the position of the seedpoint (if not given uses function getpts)
# % t : maximum intensity distance (defaults to 0.2)
# %
# % The region is iteratively grown by comparing all unallocated neighbouring pixels to the region. 
# % The difference between a pixel's intensity value and the region's mean, 
# % is used as a measure of similarity. The pixel with the smallest difference 
# % measured this way is allocated to the respective region. 
# % This process stops when the intensity difference between region mean and
# % new pixel become larger than a certain treshold (t)
# %
# % Example:
# %
# % I = im2double(imread('medtest.png'));
# % x=198; y=359;
# % J = regiongrowing(I,x,y,0.2); 
# % figure, imshow(I+J);
# %
# % Author: D. Kroon, University of Twente
    L1 = I[x][y]
    if L1 == 0:
        I[x][y] = 1

    J = np.zeros(np.shape(I))#output
    n , m = np.shape(I) #% Dimensions of input image
    reg_mean = I[x,y] # The mean of the segmented region
    reg_size = 1    #% Number of pixels in region
    ###% Free memory to store neighbours of the (segmented) region
    neg_free = 10000
    neg_pos = 0
    neg_list = np.zeros((neg_free, 3))
    
    
    pixdist = 0 #% Distance of the region newest pixel to the regio mean
    #% Neighbor locations (footprint)
    neigb = [[-1,0],[1,0],[0,-1],[0,1]]
# % Start regiogrowing until distance between regio and posible new pixels become
# % higher than a certain treshold
    while pixdist<reg_maxdist and reg_size < n*m:
        ##% Add new neighbors pixels
        for j in range(0,len(neigb)):
            #% Calculate the neighbour coordinate
            xn =int( x + neigb[j][0])
            yn = int(y + neigb[j][1])
            #% Check if neighbour is inside or outside the image
            ins= xn>=0 and yn>=0 and xn<=n-1 and yn<=m-1
            #% Add neighbor if inside and not already part of the segmented area
            if ins and J[xn][yn]==0:
                neg_list[neg_pos][:] = [xn, yn, I[xn][yn]]
                neg_pos = neg_pos + 1
                J[xn][yn]= 1
                
        #% Add a new block of free memory
        # if neg_pos+10>neg_free:
        #     neg_free = neg_free+10000
        #     neg_list[neg_pos+1:neg_free+1][:]=0 
        b = neg_list[0:neg_pos][2]
        
        ##% Add pixel with intensity nearest to the mean of the region, to the region
        dist = np.abs(b - reg_mean)
        pixdist = dist.min()
        indx = np.argmin(dist) 
        J[x][y]=2
        reg_size = reg_size+1
        #% Calculate the new mean of the region
        reg_mean= (reg_mean*reg_size + neg_list[indx][2])/(reg_size+1)
        #% Save the x and y coordinates of the pixel (for the neighbour add proccess)
        x = int(neg_list[indx][0])
        y = int(neg_list[indx][1])
        #% Remove the pixel from the neighbour (check) list
        neg_list[indx][:] = neg_list[neg_pos][:]
        neg_pos = neg_pos-1
        
    #% Return the segmented area as logical matrix'
    J = J>1
        

    return J


def clearDisk(JJ,s):


    s['StdPixel']=[]
    for props in s['coords']:
        sA = np.std(props, axis=0)
        s['StdPixel'].append(sA)

        

    id = np.argmax(s['area'])
    if s['area'][id]/s['convex_area'][id] > 0.55:
        a = s['coords'][id]
        x = a[:,0]
        y = a[:,1]
        
    else:
        idM =np.argsort(s['area'])[-2]
        x, y = s['coords'][idM]
    
   
    JJ[x,y]=0
    JJ = JJ.astype(bool) 

    return JJ , s
        
         
        
        
    