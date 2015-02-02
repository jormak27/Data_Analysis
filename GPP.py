## Jordan Makansi


print 'yes idle is working'

import sys
import os
import numpy as np
from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot as pl

from Save_array import *
from import_tif import *
Node_Locations = Locations
SnowData = snow_array

## Xgrid is the grid in meters of the geographic location of data 
## Xtrain is the snow data
def make_grid(bounding_box, ncell): 
         
        xmax, xmin, ymax, ymin = bounding_box 
        xgrid = np.linspace(xmin, xmax, ncell) 
        ygrid = np.linspace(ymin, ymax, ncell) 
     
        mX, mY = np.meshgrid(xgrid, ygrid) 
        ngridX = mX.reshape(ncell*ncell, 1); 
        ngridY = mY.reshape(ncell*ncell, 1);          
        return np.concatenate((ngridX, ngridY), axis=1) 

topleftY = geo[3]
topleftX = geo[0]
bounding_box = [topleftX+1000, topleftX, topleftY, topleftY-1000]  #this is in meters (not kilometers)
Xgrid = make_grid(bounding_box, 50)  # create cells as determined by global variable GRANULARITY

def getSnowValues():
    SnowValues = []
    for [x,y] in Node_Locations:
        column = x%topleftX
        row = y%(topleftY-1000)
        value = SnowData[row][column]
        SnowValues = SnowValues +[value] 
    return SnowValues

SnowValues = getSnowValues()

print SnowValues

######## --------- ##########
##data = np.genfromtxt('trn_data.csv', delimiter=',', skiprows=1)

data = zip(Locations,SnowValues)
data = [[x,y,z] for [[x,y],z] in data]
data = np.array(data[:len(data)])  #truncated the last point to make it easy 

Xtrain, Ytrain = data[:,:-1], data[:,-1:]  #Xtrain is the Lat Long
## Ytrain is the snow depth from LIDAR at those locations 


def f(x):
    """The function to predict."""
    return x * np.sin(x)

#----------------------------------------------------------------------
#  First the noiseless case
##X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
X = Xtrain

# Observations
##y = f(X).ravel()
##print y 
y = Ytrain

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE


##x = np.atleast_2d(np.linspace(0, 10, 1000)).T
x = Xgrid

# Instanciate a Gaussian Process model
gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

print gp

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, MSE = gp.predict(x, eval_MSE=True)
sigma = np.sqrt(MSE)
print sigma
print ' this is y_pred ', y_pred
#sys.exit(1)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE

fig = pl.figure()
pl.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
pl.plot(X, y, 'r.', markersize=10, label=u'Observations')
pl.plot(x, y_pred, 'b-', label=u'Prediction')
pl.fill(np.concatenate([x, x[::-1]]),
        np.concatenate([y_pred - 1.9600 * sigma,
                       (y_pred + 1.9600 * sigma)[::-1]]),
        alpha=.5, fc='b', ec='None', label='95% confidence interval')
pl.xlabel('$x$')
pl.ylabel('$f(x)$')
pl.ylim(-10, 20)
pl.legend(loc='upper left')
pl.savefig('prediction.png')

#----------------------------------------------------------------------
# now the noisy case
X = np.linspace(0.1, 9.9, 20)
X = np.atleast_2d(X).T

# Observations and noise
y = f(X).ravel()
dy = 0.5 + 1.0 * np.random.random(y.shape)
noise = np.random.normal(0, dy)
y += noise

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d(np.linspace(0, 10, 1000)).T

# Instanciate a Gaussian Process model
gp = GaussianProcess(corr='squared_exponential', theta0=1e-1,
                     thetaL=1e-3, thetaU=1,
                     nugget=(dy / y) ** 2,
                     random_start=100)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, MSE = gp.predict(x, eval_MSE=True)
sigma = np.sqrt(MSE)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE
fig = pl.figure()
pl.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x)$')
pl.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label=u'Observations')
pl.plot(x, y_pred, 'b-', label=u'Prediction')
pl.fill(np.concatenate([x, x[::-1]]),
        np.concatenate([y_pred - 1.9600 * sigma,
                       (y_pred + 1.9600 * sigma)[::-1]]),
        alpha=.5, fc='b', ec='None', label='95% confidence interval')
pl.xlabel('$x$')
pl.ylabel('$f(x)$')
pl.ylim(-10, 20)
pl.legend(loc='upper left')

pl.show()

