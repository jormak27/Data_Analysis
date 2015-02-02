# Jordan Makansi
# 12/13/2014
# CE 263N - Project

import sys
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from pylab import *
import csv

from Save_array import *
from import_tif import *
Node_Locations = Locations
SnowData = snow_array

# Global Variables and Constants
GRANULARITY = 50

def getSnowValues():
    SnowValues = []
    for [x,y] in Node_Locations:
        print [x,y]
        column = x%topleftX
        row = y%(topleftY-1000)
        print 'row and column', row, column 
        value = SnowData[row][column]
        SnowValues = SnowValues +[value] 
    return SnowValues

SnowValues = getSnowValues()
data = zip(Locations,SnowValues)
data = np.array([[x,y,z] for [[x,y],z] in data])
X, Y = data[:,:-1], data[:,-1:]  #Xtrain is the Lat Long, Ytrain is snow depth 

def my_range(start, end, step):
    while start <= end:
        yield start
        start += step

def make_grid(bounding_box, ncell): 
         
        xmax, xmin, ymax, ymin = bounding_box 
        xgrid = np.linspace(xmin, xmax, ncell) 
        ygrid = np.linspace(ymin, ymax, ncell) 
     
        mX, mY = np.meshgrid(xgrid, ygrid) 
        ngridX = mX.reshape(ncell*ncell, 1); 
        ngridY = mY.reshape(ncell*ncell, 1);          
        return np.concatenate((ngridX, ngridY), axis=1) 

class Predictor:

    def covariance(self, X, Z, h):
        d = spatial.distance_matrix(X,Z)
        K = np.exp(-(d**2)/(2*h*h))
        return K

    def multi_Gaussian(self,X,Z,y,h,sigma):
        predict = Predictor()
        K_train_train = predict.covariance(X,X,h)
        K_train_test = predict.covariance(X,Z,h)
        K_test_train = predict.covariance(Z,X,h)
        K_test_test = predict.covariance(Z,Z,h)
        len_identity = len(X)
        identity = np.identity(len_identity)
        noise = (sigma**2)*identity
        K_train_train_noise = K_train_train+noise
        K_train_train_noise_inv = np.linalg.inv(K_train_train_noise)
        m_f = np.dot(np.dot(K_test_train,K_train_train_noise_inv),y)
        return m_f

## Divide data into 4 equal parts (5 points each), and one part of 3 points (23 points total)
k = 4
predict = Predictor()
RMSE_min = 10000000
h_Arr = []
sigma_Arr = []
RMSE_Arr = []
data_num = len(X)
sigma_opt = 0
for h in my_range(5,200,5): 
    for sigma in my_range(0.05,0.05,0.05):
        h_Arr.append(h)
        sigma_Arr.append(sigma)
        RMSE = 0
        rem = data_num%k
        interval = data_num/k
        for index in xrange(interval,data_num+interval,interval):
            if data_num+interval == index:
                X_train = X[:data_num-rem]
                Y_train = Y[:data_num-rem]
                X_test = X[data_num-rem:]
                Y_test = Y[data_num-rem:]
            else:
                X_train = np.append(X[0:index-interval],X[index:],axis=0)
                Y_train = np.append(Y[0:index-interval],Y[index:],axis=0)
                X_test = X[index-interval:index]
                Y_test = Y[index-interval:index]
            m_f_fold = predict.multi_Gaussian(X_train,X_test,Y_train,h,sigma)
            RMSE_fold = np.sqrt(((m_f_fold-Y_test)**2).mean())
            RMSE = RMSE + RMSE_fold
        RMSE_Arr = RMSE_Arr+[RMSE]
        RMSE = RMSE/k
        if RMSE < RMSE_min:
            RMSE_min = RMSE
            h_opt = h
            sigma_opt = sigma
        
print 'OPTIIMAL BANDWIDTH:', h_opt
print 'OPTIMAL SIGMA_2:', sigma_opt   

plt.plot(h_Arr,RMSE_Arr)
plt.savefig('Bandwidth_vs_Error_(Sigma=0.58).png')

print 'finished plot' 
X_test = X

#run predictor on h_opt
predict = Predictor()
m_f = np.absolute(predict.multi_Gaussian(X,X_test,Y,h_opt,sigma_opt))
row = np.append(X_test,m_f,axis=1)

np.savetxt('predictions_on_Zeshi_code.csv',row,delimiter=',')

## feed a grid of points into the predictor in order to generate the simulation

# Xgrid is a grid of points with the same extent, just at resolution 50
topleftY = geo[3]
topleftX = geo[0]
bounding_box = [topleftX+1000, topleftX, topleftY, topleftY-1000]  #this is in meters (not kilometers)
xmax, xmin, ymax, ymin = bounding_box 
lat_min =  topleftX
lng_min = topleftY-1000

bounding_box = [topleftX+1000, topleftX, topleftY, topleftY-1000]
Xgrid = make_grid(bounding_box,GRANULARITY)

print 'plotting points'
Xtrain = X
for i in range(len(Xtrain)):
    plt.scatter(Xtrain[i][0],Xtrain[i][1])
    print Xtrain[i][0]
    
#Compute Simulation using the Optimal H, found from above
m_f = np.absolute(predict.multi_Gaussian(X,Xgrid,Y,h_opt,sigma_opt))
print 'this is m_f', m_f
image = m_f.reshape(GRANULARITY,GRANULARITY)
ax=plt.imshow(image,extent=[topleftX, topleftX+1000, topleftY-1000, topleftY])
plt.colorbar(ax)
plt.savefig('Snow_Plot_and_Nodes_with_H=136.jpg', bbox_inches='tight')

### Compute Error between LIDAR data and predicted values
Error_grid = []
m_f_repeat = np.repeat(image,20,axis=1)
m_f_grid = np.repeat(np.repeat(image,20,axis=1),20,0)
print '\n finished computing the m_f_grid, here it is: ', m_f_grid
print '\n the shape of the m_f_grid should be (1000,1000)', m_f_grid.shape


# Construct grid and plot 
Error_grid = m_f_grid - snow_array
RMSE_grid = np.array([np.sqrt(i**2) for i in Error_grid])
print 'RMSE_grid computed and its shape should be (1000,1000)', RMSE_grid.shape
ax = plt.imshow(RMSE_grid, extent=[topleftX, topleftX+1000, topleftY-1000, topleftY])
plt.savefig('Error_heatmap.jpg', bbox_inches='tight')
np.savetxt('Error.csv',np.array([RMSE_grid.mean()]),delimiter=',')

print 'plotting the node locations on the Error heatmap'
Xtrain = X
for i in range(len(Xtrain)):
    plt.scatter(Xtrain[i][0],Xtrain[i][1])
