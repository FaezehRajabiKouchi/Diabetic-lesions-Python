from scipy.io import loadmat
import numpy as np
import FSfuncs as fun

#load data
data = loadmat('AllDataRegion.mat')
Label = data['ss']['Label']

#############################%region charactristic
##%Returns a scalar that specifies the actual number of pixels in the region.
Area = data['ss']['Area'] 
##%Returns a scalar that specifies the ratio of pixels in the region to pixels in the total bounding box. Computed as the Area divided by the area of the bounding box.
Extend = data['ss']['Extent'] 
##%Returns a scalar specifying the proportion of the pixels in the convex hull that are also in the region. Computed as Area/ConvexArea
Solidity = data['ss']['Solidity'] 
##%eturns a scalar that specifies the number of pixels in 'ConvexImage'.
ConvexArea = data['ss']['ConvexArea']
StdPixel = data['ss']['StdPixel']

Feature1 = np.concatenate((Area,Extend,Solidity,ConvexArea,StdPixel),axis = 1)
Feature2 = np.zeros((len(Area), 27))
##%% Colour Feature
for i in range(0,len(Area)):
    I = data['ss']['I'][i][0]/255
    R = I[:,:,0].astype(float)
    G = I[:,:,1].astype(float)
    B = I[:,:,2].astype(float)
    SDR = np.std(R)
    MeanR = np.mean(R)
    SDG = np.std(G)
    MeanG = np.mean(G)
    SDB = np.std(B)
    MeanB = np.mean(B)
    J = data['ss']['J'][i][0].astype(float)/255
    n, m = np.shape(R)
    l = 0
    JJ = np.zeros((n*m, 3))
    for k in range(0,n):
        for j in range(0,m):
            if J[k,j,0]>0.1:
                JJ[l][:] = I[k,j,:]
    
    MRGB = np.mean(JJ, axis =0)
    SDRGB = np.std(JJ, axis =0)
    H, bin_edge = np.histogram(G.ravel(), bins =15)
    Feature2[i] = [SDR,MeanR,SDG,MeanG,SDB,MeanB,MRGB[0],MRGB[1], MRGB[2],
                   SDRGB[0],SDRGB[1],SDRGB[2],
                   H[0], H[1], H[2], H[3], H[4], H[5], H[6], H[7], H[8], H[9],
                   H[10], H[11], H[12], H[13], H[14]]
Feature = np.concatenate((Feature2, Feature1), axis = 1)    
#############################% classification
A = np.concatenate((Feature, Label), axis = 1)
Z = fun.EvalSVM(A)
Z = [round(num, 4) for num in Z]
Z3 = fun.EvalMLP(A)
Z3 = [round(num, 4) for num in Z3]


#############################% FDA
N = 10
Y = fun.Fisher(Feature, Label, N)
A = np.concatenate((Y, Label), axis = 1)
Z1 = fun.EvalSVM(A)
Z1 = [round(num, 4) for num in Z1]


#############################%MLP 

Z2 = fun.EvalMLP(A)
Z2 = [round(num, 4) for num in Z2]
#############################%print Result
print( '                      Specificity|Sensitivity|Accuracy')
print('SVM with ',str(np.shape(Feature)[1]),' Feature: ',str(Z[0]), ' | ',str(Z[1]),' | ', str(Z[2]))

print('SVM With ',str(N),' Feature: ',str(Z1[0]),' | ', str(Z1[1]),' | ', str(Z1[2]))

print('MLP With ',str(np.shape(Feature)[1]),' Feature: ',str(Z3[0]),' | ', str(Z3[2]), ' | ', str(Z3[1]))

print('MLP With ',str(N),' Feature: ',str(Z2[0]),' | ', str(Z2[1]),' | ', str(Z2[2]));
    
