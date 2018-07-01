import cv2
import tensorflow as tf
import numpy as np
import random #mini batch를 사용하기 위해서는 Random한 data set을 묶음으로 batch학습을 시켜야하므로 필요하다.
import matplotlib.pyplot as plt #실제 data value에 대한 정보를 matlab의 plot처럼 graph를 도식화하기 위해 불러온다.
import glob
#import ConvMNIST as CM

batch_size = 50

#A = [CM.imageprepare(file) for file in glob.glob("Dataset/BW/0_Te/*.png")]
X0_Te = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY) for file in glob.glob("Dataset/BW/0_Te/*.jpg")]
X0_Te = np.array(X0_Te)
X0_Te = X0_Te.reshape(len(X0_Te),-1)
#X0_Te = cv2.cvtColor(X0_Te, cv2.COLOR_BGR2GRAY)


X1_Te = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY) for file in glob.glob("Dataset/BW/1_Te/*.jpg")]
X1_Te = np.array(X1_Te)
X1_Te = X1_Te.reshape(len(X1_Te),-1)


X5_Te = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY) for file in glob.glob("Dataset/BW/5_Te/*.jpg")]
X5_Te = np.array(X5_Te)
X5_Te = X5_Te.reshape(len(X5_Te),-1)
#X5_Te = [CM.imageprepare(file) for file in glob.glob("Dataset/BW/5_Te/*.png")]



X0_Tr = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY) for file in glob.glob("Dataset/BW/0_Tr/*.jpg")]
X0_Tr = np.array(X0_Tr)
X0_Tr = X0_Tr.reshape(len(X0_Tr),-1)


X1_Tr = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY) for file in glob.glob("Dataset/BW/1_Tr/*.jpg")]
X1_Tr = np.array(X1_Tr)
X1_Tr = X1_Tr.reshape(len(X1_Tr),-1)


X5_Tr = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY) for file in glob.glob("Dataset/BW/5_Tr/*.jpg")]
X5_Tr = np.array(X5_Tr)
X5_Tr = X5_Tr.reshape(len(X5_Tr),-1)



Y0_Tr = np.zeros((1,len(X0_Tr)))
Y1_Tr = np.zeros((1,len(X1_Tr)))
Y1_Tr[Y1_Tr==0] = 1
Y5_Tr = np.zeros((1,len(X5_Tr)))
Y5_Tr[Y5_Tr==0] = 2

Y0_Te = np.zeros((1,len(X0_Te)))
Y1_Te = np.zeros((1,len(X1_Te)))
Y1_Tr[Y1_Tr==0] = 1
Y5_Te = np.zeros((1,len(X5_Te)))
Y5_Te[Y5_Te==0] = 2

#X = [cv2.imread(file) for file in glob.glob("Dataset/BW/0_X/*.png")]
#Y = [cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY) for file in glob.glob("Dataset/BW/0_Y/*.png")]
#A = cv2.imread('Dataset/0_X/IMG_4040.jpg')
#gray_image = cv2.cvtColor(A, cv2.COLOR_BGR2GRAY)
#GRY = [cv2.cvtColor(images, cv2.COLOR_BGR2GRAY) for file in images]
#GRY = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)

#data1 = np.random.multivariate_normal(m1, cov1, 250) #normalized variables 1~3
#data2 = np.random.multivariate_normal(m2, cov2, 180) #Draw random samples from a multivariate normal distribution.
#data3 = np.random.multivariate_normal(m3, cov3, 100) #즉, 3개의 cluster를 생성하기 위해 정규분포를 따르는 랜덤한 points들을 생성
#m1, cov1 = [9, 8], [[1.5, 2], [1, 2]] #data value float type from 8~9
#data1 = np.random.multivariate_normal(m1, cov1, 250) #normalized variables 1~3
    

#X_Tr = np.vstack((X0_Tr,X5_Tr)) #vertical stack

X_Tr = np.vstack((X0_Tr, np.vstack((X1_Tr,X5_Tr)))) #vertical stack

Y_Tr = np.zeros((len(X_Tr),3))
Y_Tr[:len(Y0_Tr[0]),0] = 1 #Y0 Tr matrix
Y_Tr[len(Y0_Tr[0]):(len(Y0_Tr[0]) + len(Y1_Tr[0])),1] = 1 #Y1 Tr matrix
Y_Tr[(len(Y0_Tr[0]) + len(Y1_Tr[0])):(len(Y0_Tr[0]) + len(Y1_Tr[0]) + len(Y5_Tr[0])),2] = 1 #Y5 Tr matrix
#Y_Tr = np.hstack((Y0_Tr,Y5_Tr)) #horizontal stack
#Y_Tr = Y_Tr.T

X_Te = np.vstack((X0_Te, np.vstack((X1_Te,X5_Te))))

Y_Te = np.zeros((len(X_Te),3))
Y_Te[:len(Y0_Te[0]),0] = 1 #Y0 Te matrix
Y_Te[len(Y0_Te[0]):(len(Y0_Te[0]) + len(Y1_Te[0])),1] = 1 #Y1 Te matrix
Y_Te[(len(Y0_Te[0]) + len(Y1_Te[0])):(len(Y0_Te[0]) + len(Y1_Te[0]) + len(Y5_Te[0])),2] = 1 #Y5 Tr matrix
#Y_Te = np.hstack((Y0_Te,Y5_Te))
#Y_Te = Y_Te.T

        #Test & Train random choice = shuffle data
R_Ch_Tr = np.random.choice(len(X_Tr), batch_size)
R_Ch_Te = np.random.choice(len(X_Te), batch_size)




            #append
Mini_Batch_X_Tr = []
Mini_Batch_Y_Tr = []
            
Mini_Batch_X_Te = []
Mini_Batch_Y_Te = []
        
            #data assignments
for i in range(len(R_Ch_Tr)):
    Mini_Batch_X_Tr.append( X_Tr[R_Ch_Tr[i]] )
    Mini_Batch_Y_Tr.append( Y_Tr[R_Ch_Tr[i]] )
            
for i in range(len(R_Ch_Te)):
    Mini_Batch_X_Te.append( X_Te[R_Ch_Te[i]] )
    Mini_Batch_Y_Te.append( Y_Te[R_Ch_Te[i]] )
        
            #array transformation
Mini_Batch_X_Tr = np.array(Mini_Batch_X_Tr)
Mini_Batch_Y_Tr = np.array(Mini_Batch_Y_Tr)
        
Mini_Batch_X_Te = np.array(Mini_Batch_X_Te)
Mini_Batch_Y_Te = np.array(Mini_Batch_Y_Te)




def shuffle():
        #Test & Train random choice = shuffle data
    global R_Ch_Tr
    R_Ch_Tr = np.random.choice(len(X_Tr), batch_size)
    #len(X_Tr) = 528 중 10개
    global R_Ch_Te
    R_Ch_Te = np.random.choice(len(X_Te), batch_size)
    #(len(X_Te) = 90 중 10개
    
def Select(Sel):
            #append
    Mini_Batch_X_Tr = []
    Mini_Batch_Y_Tr = []
            
    Mini_Batch_X_Te = []
    Mini_Batch_Y_Te = []
    
            #data assignments
    for i in range(len(R_Ch_Tr)):
        Mini_Batch_X_Tr.append( X_Tr[R_Ch_Tr[i]] )
        Mini_Batch_Y_Tr.append( Y_Tr[R_Ch_Tr[i]] )
            
    for i in range(len(R_Ch_Te)):
        Mini_Batch_X_Te.append( X_Te[R_Ch_Te[i]] )
        Mini_Batch_Y_Te.append( Y_Te[R_Ch_Te[i]] )
        
            #array transformation
    Mini_Batch_X_Tr = np.array(Mini_Batch_X_Tr)
    Mini_Batch_Y_Tr = np.array(Mini_Batch_Y_Tr)
        
    Mini_Batch_X_Te = np.array(Mini_Batch_X_Te)
    Mini_Batch_Y_Te = np.array(Mini_Batch_Y_Te)

    if(Sel == 'Mini_X_Tr'):
        return Mini_Batch_X_Tr
    elif(Sel == 'Mini_Y_Tr'):
        return Mini_Batch_Y_Tr
    elif(Sel == 'Mini_X_Te'):
        return Mini_Batch_X_Te
    elif(Sel == 'Mini_Y_Te'):
        return Mini_Batch_Y_Te
    elif(Sel == 'X_Te'):
        return X_Te
    elif(Sel == 'Y_Te'):
        return Y_Te
    elif(Sel == 'X_Tr'):
        return X_Tr
    elif(Sel == 'Y_Tr'):
        return Y_Tr
    
#np.random.shuffle(X)
#X[np.random.choice(np.arange(len(X)), K), :] 

'''
def get_mini_batches(X, y, batch_size):
    random_idxs = random.choice(len(y), len(y), replace=False)
    X_shuffled = x[random_idxs, : ]
    y_shuffuled = y[random_idxs]
    mini_batches = [(X_shuffled[i:i+batch_size, :], shuffled[i:i+batch_size]) for i in range(0, len(y), batch_size)]
    return mini_batches

get_mini_batches(X, X, 10)

hsv = cv2.cvtColor(images, cv2.COLOR_BGR2HSV)
lower = np.array((0,0,0))
upper = np.array((255,0,150))
mask = cv2.inRange(hsv, lower, upper)
#mnist = input_data.read_data_sets("Dataset/0_X", one_hot=True)
#images = images.reshape(56, 56)
'''
