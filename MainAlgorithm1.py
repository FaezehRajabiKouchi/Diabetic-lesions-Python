import numpy as np
import funcs as fun
import matplotlib.pyplot as plt
from skimage import exposure
import skimage
from scipy.io import savemat


################################% Read Image

k = 0

for num in range(1, 90):
    print(num)
    ##### In this section we should add the path of the dataset                      
    path = r'Directory of the dataset\Diabetic-lesions-Python\resources\images\ddb1_fundusimages\image'+str(num)+'.png'
    IM = (plt.imread(path)*255).astype(np.uint8)
    #print(np.unique(IM[:,:,0]))
    #plt.imshow(IM)
################################% Resize image

    im = fun.resizeretina(IM, 576,750)
    #print(np.unique(im[:,:,0]))
    n, m, d = np.shape(im)
    im1 = im
    #plt.imshow(im)
################################% Pre-Procces
    #Display the superpixel boundaries overlaid on the original image.
    
    L,N = fun.superpixels(im,3000)
    #print(N)
    #plt.imshow(mark_boundaries(im, L))
    I1= np.zeros((3,576,750))
    I = np.zeros(np.shape(im), np.uint8)
    
    idx = fun.label2idx(L)
    
    numRows = n
    numCols = m
    for labelVal in range(0,N):
        I1[0].ravel()[idx[labelVal]] = im[:,:,0].ravel()[idx[labelVal]].max()
        I1[1].ravel()[idx[labelVal]] = 2* im[:,:,1].ravel()[idx[labelVal]].max()
        I1[2].ravel()[idx[labelVal]] = 2* im[:,:,2].ravel()[idx[labelVal]].max()
    I[:,:,0] =I1[0]
    I[:,:,1] =I1[1]
    I[:,:,2] =I1[2]
    #plt.imshow(I)
    IG= (I[:,:,1].astype(float) + I[:,:,2].astype(float))/(2*255)
    IG = exposure.equalize_adapthist(IG, clip_limit=0.03)
    #plt.imshow(IG)
    
    
###############################% segmentation
    mr = 0.54
    Mask = IG>mr*IG.max()
    se = skimage.morphology.disk(2)
    Mask = skimage.morphology.closing(Mask, se)
    Mask , s = fun.Regionprops(Mask, IG, mr)
    JJ = np.zeros((n, m))
    reg_maxdist = 0.05
    XYC = [s['centroid-0'].astype(int), s['centroid-1'].astype(int)]
    
    for i in range(0, len(XYC[0])):
        J = fun.regiongrowing(Mask,XYC[0][i],XYC[1][i],reg_maxdist)
        JJ = JJ + J
    
    JJ = JJ.astype(bool)
    
    
    
    label_img = skimage.measure.label(JJ, connectivity=JJ.ndim)
    s = skimage.measure.regionprops_table(label_img, properties=['centroid', 'area', 'bbox',
                                                                 'extent', 'coords', 'convex_area'
                                                                 ])
      
    J,s = fun.clearDisk(Mask,s)
    
    
    
    J = J.astype(float)
    JJ = np.zeros(np.shape(im), np.uint8)
    JJ[:,:,0] = J
    JJ[:,:,1] = J
    JJ[:,:,2] = J
    IM = im*np.uint8(JJ)
    XYC = [s['centroid-0'].astype(int), s['centroid-1'].astype(int)]
    
    s['label'] = []
    s['name'] = []
    s['I']=[]
    s['J']=[]
    for i in range(0, len(XYC[0])):
        xmin = XYC[0][i]-29
        xmax = XYC[0][i]-29+59
        ymin = XYC[1][i]-25
        ymax = XYC[1][i]-25+49
                
        s['I'].append(im[xmin:xmax, ymin:ymax,:])
        s['J'].append(IM[xmin:xmax, ymin:ymax,:])
        plt.figure()
        plt.subplot(2,2,1)
        plt.imshow( im)
        plt.plot(XYC[0][i] , XYC[1][i],'og')
        plt.subplot(2,2,2)
        plt.imshow(J)
        plt.subplot(2,2,3)
        plt.imshow(s['J'][i])
        plt.subplot(2,2,4)
        plt.imshow(s['I'][i])
        plt.show()
        im1 = im1 + 255*np.uint8(JJ)
        s['name'].append(['image',str(num)])
        Cammand = input('What is Diabet?(Diabet = 0)(Normal = 1)(else 2)')
        if Cammand == 0:
            s['label'].append(-1)
        elif Cammand == 1:
            s['label'].append(1)
        else:
            s['label'].append(0)
    
            
    
    k = k + 1
    ############ In this section we should add the path of the savemat                       
    fileName = r'F:\APPLY\project2\resources\Result\i'+str(num)+'.mat'
    savemat(fileName,s)

    
    
        
        
    
    





    

