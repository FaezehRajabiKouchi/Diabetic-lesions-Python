import numpy as np 
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy import linalg as la
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf


def Normalize_Fcn(x, xmin, xmax):
    xN = (x - xmin) / (xmax - xmin)*2-1
    return xN

def EvalSVM(a):
    m, nn = np.shape(a)
    idx = list(np.argwhere(a[:,-1] == 1))
    class1 = np.zeros((len(idx), nn))
    for i in range(0,len(idx)):
        class1[i,:] = a[idx[i],:]
    b = np.delete(a, idx, axis=0)

    class2 = b
    d1 = len(idx)
    d2 = m - d1
    kn = 30
    result = np.zeros((kn,3))
    for Q in range(0,kn):
        d = np.random.permutation(d1)
        class1 =class1[d][:]
        d = np.random.permutation(d2)
        class2 = class2[d][:]
        
        class1 = np.concatenate((class1[:,0:nn-1], np.ones((d1,1)), -np.ones((d1,1))), axis = 1)
        class2 = np.concatenate((class2[:,0:nn-1], -np.ones((d2,1)), np.ones((d2,1))), axis = 1)
        
        w = d1 - d2
        m1 = d1
        m2 = d2
        if w>0:
            train = np.concatenate((class1[0:int(0.7*m2)][:], class2[0:int(0.7*m2)][:]), axis =0)
            test= np.concatenate((class1[int(0.7*m2):m1,:],class2[int(0.7*m2):m2,:]), axis = 0)
        else:
            train = np.concatenate((class1[0:int(0.7*m1)][:], class2[0:int(0.7*m1)][:]), axis =0)
            test= np.concatenate((class1[int(0.7*m1):m1,:],class2[int(0.7*m1):m2,:]), axis = 0)
        m3, n3 = np.shape(train)
        d = np.random.permutation(m3)
        train = train[d][:]
        
        sample = test[:,0:nn-1]
        training = train[:, 0:nn-1]
        group = train[:,-1].astype(int)
        t = make_pipeline(StandardScaler(), svm.LinearSVC(max_iter= 10000))
        t.fit(training, group)
        ypred = t.predict(sample)
        ytf = test[:, -1].astype(int)
        c2 = len(ytf)
        yfh = ypred
        TP1 = 0
        TN1 = 0
        FP1 = 0
        FN1 = 0
        d = ytf-yfh
        for n in range(0,len(d)):
            if abs(d[n])>0:
                if ytf[n]==1:
                    FN1 = FN1+1
                else:
                    FP1 = FP1+1
            if abs(d[n])==0:
                if ytf[n]==1:
                    TP1 = TP1+1
                else:
                    TN1 = TN1+1
        SENSITIVITY1=100*(TP1/(TP1+FN1))
        SPECIFICITY1=100*(TN1/(TN1+FP1))
        ACC1=100*((TP1+TN1)/(TP1+TN1+FP1+FN1))
    
        result[Q,:]=[SENSITIVITY1 , SPECIFICITY1, ACC1]
    Z = np.mean(result,  axis = 0)
    return Z
        
        
        

        
    
def EvalMLP(Data):
    
    n , m = np.shape(Data)
    Inputs = Data[:, 0:m-1]
    v, nn = np.shape(Inputs)
    Targets = Data[:, -1].astype(int)
    for kk in range(0,nn):
        Inputs[:,kk] = Normalize_Fcn(Inputs[:,kk],min(Inputs[:,kk]),max(Inputs[:,kk]))
    
    

    x_train, x_test, y_train, y_test = train_test_split(Inputs, Targets,
                                                        test_size = 0.25)   
    x_train = np.asarray(x_train).astype(float)

    y_train = np.asarray(y_train).astype(float)
    x_test = np.asarray(x_test).astype(float)
    model = Sequential()
    model.add(Dense(nn+10 , input_shape =(nn,), 
                    activation = 'linear', name = 'input'))
    model.add(Dense(10 , activation = 'tanh', name = 'h1'))
    model.add(Dense(3 , activation = 'tanh', name = 'h2'))
    model.add(Dense(1 , activation = 'tanh', name = 'output'))
    opt = Adam(learning_rate= 0.001)
    model.compile(optimizer= opt, loss='mse')
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    model.fit(x_train, y_train, verbose= 0, epochs=100, callbacks=[callback])
    out = model.predict(x_train)
    out =np.transpose(np.sign(out)).astype(int)
    

     
    TP1=0
    TN1=0
    FP1=0
    FN1=0
    d = np.transpose(out - y_train)
    for n in range(0, len(d)):
        if abs(d[n])>0:
            if y_train[n] == 1:
                FN1 = FN1+1
            else:
                FP1 = FP1+1
        elif abs(d[n])==0:
            if y_train[n] == 1:
                TP1 = TP1+1
            else:
                TN1 = TN1+1
    
    SENSITIVITY1=100*(TP1/(TP1+FN1))
    SPECIFICITY1=100*(TN1/(TN1+FP1))
    ACC1=100*((TP1+TN1)/(TP1+TN1+FP1+FN1))
    
    Z=[SENSITIVITY1 , SPECIFICITY1,ACC1]
    
    return Z
    
    
    
def Fisher(X, L, N):
    Classes = np.unique(L)
    k = len(Classes)
    n = np.zeros((k,1)).astype(int)
    C = {}
    M = np.mean(X, axis = 0)
    S = {}
    Sw = 0
    Sb = 0
    for j in range(0,k):
        idx = np.argwhere(L==Classes[j])
        idx =idx[:,0]
        Xj = np.zeros((len(idx), np.shape(X)[1]))
        for i in range(0,len(idx)):
            Xj[i,:] = X[idx[i],:]
        
        n[j] = np.shape(Xj)[0]
        C[j] = np.mean(Xj, axis = 0)
        S[j] = 0
        a = np.zeros((32,1))
        for i in range(0, n[j][0]):
            a[:,0] = Xj[i,:]- C[j]
            b = np.matmul(a, np.transpose(a))
            S[j] = S[j] + b
       
        
        Sw = Sw + S[j]
        a = np.zeros((32,1))
        a[:,0] = C[j]-M
        b = np.matmul(a, np.transpose(a))
        Sb = Sb + n[j] * b
        
        
    q = la.eig(Sb, Sw, left=False, right=True)
    W = q[1]
    LAMBDA = q[0] 
    
    
    SortOrder = np.argsort(-LAMBDA , axis =0)
    W = W[:, SortOrder]
    Y = np.matmul(X, W)
    Y = Y[:, 0:N]
    return Y
    
        
        
        
    
    

