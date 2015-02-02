#Jordan Makansi
#CEE 263N - Final Project
#12/9/14

#imports
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy import spatial
from sklearn import cross_validation

#Global Variables and constants 
H_MIN = 500.0  #minimum bandwidth tested 
H_MAX = 1500.0 #maximum bandwidth tested 
STEP_SIZE_H = 100.0  #step size for H iterations 
SIGMA_2_MIN = 0.01  # minimum covariance variance tested 
SIGMA_2_MAX = 0.07 # maximumcovariance tested 
STEP_SIZE_SIGMA = .05 # step size for covariance iterations

OPTIMAL_H = H_MIN 
OPTIMAL_SIGMA_2 = SIGMA_2_MIN
GRANULARITY = 50


########## ----------- HIGH LEVEL OPERATIONS ---------------

    #import array of current sensor node locations
    #retrieve snow values for those locations
    #generate GP for those snow values
        #try different bandwidth parameters
        #see if they minimize the error
    #produce heatmap of error given the best possible GP 

## Xgrid is the grid in meters of the geographic location of data 
## Xtrain is the snow data

from Save_array import *
from import_tif import *
Node_Locations = Locations
SnowData = snow_array

def getSnowValues():
    SnowValues = []
    for [x,y] in Node_Locations:
        column = x%topleftX
        row = y%(topleftY-1000)
        value = SnowData[row][column]
        SnowValues = SnowValues +[value] 
    return SnowValues

SnowValues = getSnowValues()

######## --------- ##########
##data = np.genfromtxt('trn_data.csv', delimiter=',', skiprows=1)

data = zip(Locations,SnowValues)
data = [[x,y,z] for [[x,y],z] in data]
data = np.array(data[:len(data)])  #truncated the last point to make it easy 

Xtrain, Ytrain = data[:,:-1], data[:,-1:]  #Xtrain is the Lat Long
## Ytrain is the snow depth from LIDAR at those locations 

##for i in range(len(Xtrain)):
##    Xtrain[i][0]*=10**-3  #conversion from meters to kilometers
##    Xtrain[i][1]*=10**-3

#Get matrices of test locations
######## ---------##########

data = Locations
data = np.array(data)

Xtest = data[:,:]
##for i in range(len(Xtest)):
##    Xtest[i][0]*=10**-3  #conversion from meters to kilometers
##    Xtest[i][1]*=10**-3

#Set up the data grid for the contour plot
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

Xgrid = make_grid(bounding_box, GRANULARITY)  # create cells as determined by global variable GRANULARITY

##for i in range(len(Xgrid)):
##    Xgrid[i][0]*=10**-3  #convert to kilometers 
##    Xgrid[i][1]*=10**-3

'''  From here and below, we only use these matrices:
   0) Xtest   
   1) Xgrid   #Xgrid gives the lat-lon locations of the upper left hand corner of each cell  (in UTM coordinates, Zone 11S)
   2) Xtrain  
   3) Ytrain'''

print 'Xtest', Xtest
print 'Xgrid', Xgrid
print 'Xtrain', Xtrain
print 'Ytrain', Ytrain

def covariance(X, Z, h):
    '''This function computes the covariance matrix with a guassian kernel
    between the two matrices.
        Input: two matrices, and bandwidth(h)
        Output: covariance matrix.'''
    d = spatial.distance_matrix(X,Z)
    K = np.exp(-(d**2) / (2*h*h))
    return K


class Optimization():

    def __init__(self):
        self.OPTIMAL_H = H_MIN 
        self.OPTIMAL_SIGMA_2 = SIGMA_2_MIN
        self.error_min = 10000000000     
        self.h = H_MIN
        self.sigma_square = SIGMA_2_MIN
        self.errors = []
        self.H_s    = []

    def Optimize(self):
        '''Optimizes using k-fold cross-validation'''
        k = 4
        while self.h<=H_MAX:
            
            ## CREATE DICTIONARY OF SAMPLE DATA 
            Xsamples = {}
            Ysamples = {}

            #Divide the matrix into "k" pieces 
            for n in range(1,k+1): 
                index_1 = ((n-1)*len(Xtrain))/k
                index_2 = (n*len(Xtrain))/k
                Xsamples['Xsample_'+str(n)] = Xtrain[index_1:index_2]
                Ysamples['Ysample_'+str(n)] = Ytrain[index_1:index_2]
                #print 'this is Xsamples for ', n, Xsamples['Xsample_'+str(n)]             

            #create dictionary for K matrices (or Covariance matrices) using current iteration bandwidth
            K_n_i = {}
            for n in range(1,k+1):
                for i in range(1,k+1):
                    K_n_i['K_'+str(n)+'_'+str(i)] = covariance(Xsamples['Xsample_'+str(n)],Xsamples['Xsample_'+str(i)],self.h)
                    #print 'this is covariance  for ', n, i, 'with hiked bandwidth'
                    #print Xsamples['Xsample_'+str(n)]
                    #print covariance(Xsamples['Xsample_'+str(n)],Xsamples['Xsample_'+str(i)],self.h)

            self.sigma_square = SIGMA_2_MIN
            
            #create m_f dictionary, which is the predictive mean for each combination of training/test data
            m_f_n_i = {}
            while self.sigma_square <= SIGMA_2_MAX:
                error = 0  #reset the error computation 
                
                for n in range(1,k+1):
                    for i in range(1,k+1):
                        if n==i:
                            pass
                        else:
                            Ysample = Ysamples['Ysample_'+str(n)]
                            Xsample = Xsamples['Xsample_'+str(n)]
                            K_i_n = K_n_i['K_'+str(i)+'_'+str(n)]
                            K_n_n = K_n_i['K_'+str(n)+'_'+str(n)]                            
                            term_2 = self.sigma_square*np.matrix(np.identity(len(Xsample)))
                            term_3 = (Ysample-np.average(Ysample))
                            m_f_n_i['m_f_'+str(n)+'_'+str(i)]=np.average(Ysample)+K_i_n*np.linalg.inv(K_n_n+term_2)*(term_3)
                            
                #Compute errors 
                for n in range(1,k+1):
                    Xsamples_n = Xsamples['Xsample_'+str(n)]
                    for i in range(1,k+1):
                        if n == i:
                            pass
                        else:
                            counter = 0 
                            for x in range(len(Xsamples_n)):
                                m_f = m_f_n_i['m_f_'+str(i)+'_'+str(n)]
                                Yindex = x
                                #print '\n average value of ', n, 'and ', i
                                #print m_f[0]
                                #print 'the average is ', np.average(Ysamples['Ysample_'+str(n)])
                                for b in range(n):
                                    if b == 0:
                                        pass
                                    else:
                                        X = Xsamples['Xsample_'+str(b)]
                                        Yindex = Yindex + len(X)
                                term_1 = m_f[x]-Ytrain[Yindex]
                                term_2 = m_f[x]-Ytrain[Yindex]
                                term_1 = np.matrix(term_1[0,0])
                                term_2 = np.matrix(term_2[0,0])
                                error = error + np.sqrt(np.matrix.item((term_1)*(term_2)))
                                
                #check if error minimum is breached
                self.errors +=[error]
                self.H_s += [self.h]
                if (self.error_min > error):
                    self.OPTIMAL_H = self.h
                    self.OPTIMAL_SIGMA_2 = self.sigma_square
                    self.error_min = error
                self.sigma_square += STEP_SIZE_SIGMA
            self.h = self.h+STEP_SIZE_H
            
        #done :)
        print 'FOUND OPTIMALS : \n', 
        print 'this is the optimal H', self.OPTIMAL_H
        print 'this is the optimal covariance', self.OPTIMAL_SIGMA_2
        print 'minimum error for these parameters is ', self.error_min

        #Writing predictions to a .csv file
        with open('H_vs_Error.csv','w') as f:
            f.write("H,Error\n")
            for i in range(len(self.errors)):
                f.write("%f,%f \n" % (self.H_s[i],self.errors[i]))
        f.close()
        
        return self.OPTIMAL_H, self.OPTIMAL_SIGMA_2
    

class Predictions():
    '''This is part 1.  Finds the Optimal_H and Optimal_sigma_2, computes predictions and writes output to a .csv file'''

    def __init__(self):
        self.Prediction = Optimization()
        Prediction=self.Prediction.Optimize()
        self.OPTIMAL_H, self.OPTIMAL_SIGMA_2 = Prediction

        #computing optimal predictions and storing
        K_trn_trn = covariance(Xtrain,Xtrain, self.OPTIMAL_H)
        K_test_test = covariance(Xtest,Xtest, self.OPTIMAL_H)
        K_trn_test = covariance(Xtrain,Xtest, self.OPTIMAL_H)
        K_test_trn = covariance(Xtest,Xtrain, self.OPTIMAL_H)

        self.m_f = np.average(Ytrain)+K_test_trn*np.linalg.inv(K_trn_trn+self.OPTIMAL_SIGMA_2*np.matrix(np.identity(23)))*(Ytrain-np.average(Ytrain))
        m_f = self.m_f
        
        #Writing predictions to a .csv file
        with open('snow_prediction.csv','w') as f:
            f.write("lat,lng,mm_predicted\n")
            for i in range(len(Xtest)):
                f.write("%f,%f,%f\n" % (Xtest[i][0],Xtest[i][1],m_f[i]))
        f.close()
        

class Simulation():
    '''This is part 2. Calculates one stochastic simulation.'''

    def __init__(self, OPTIMAL_H, OPTIMAL_SIGMA_2):
        self.OPTIMAL_H = OPTIMAL_H
        self.OPTIMAL_SIGMA_2 = OPTIMAL_SIGMA_2
    
        #compute m_f and cov_f
        K_grid_grid = covariance(Xgrid,Xgrid, self.OPTIMAL_H)
        K_trn_grid =  covariance(Xtrain,Xgrid, self.OPTIMAL_H)
        K_grid_trn =  covariance(Xgrid,Xtrain, self.OPTIMAL_H)
        K_trn_trn =   covariance(Xtrain,Xtrain, self.OPTIMAL_H)

        self.m_f_grid = K_grid_trn*np.linalg.inv(K_trn_trn+self.OPTIMAL_SIGMA_2*np.matrix(np.identity(23)))*Ytrain
        self.cov_f_grid=K_grid_grid-(K_grid_trn*(np.linalg.inv((K_trn_trn+np.matrix(np.identity(23))))*K_trn_grid))

        #print self.m_f_grid
        
        self.other = np.linalg.inv(K_trn_trn+self.OPTIMAL_SIGMA_2*np.matrix(np.identity(23)))
        self.Y = Ytrain
        #print 'this is K_grid_trn',  covariance(Xgrid,Xtrain, self.OPTIMAL_H)
        #print 'this is middle part ', self.OPTIMAL_SIGMA_2*np.matrix(np.identity(23))
        #print 'this is ytrain ', Ytrain 
        #print 'this is m_f_grid ', self.m_f_grid
        #print 'this is the other part \n', np.linalg.inv(K_trn_trn+self.OPTIMAL_SIGMA_2*np.matrix(np.identity(23)))
        
        #Simulating one value from the predictions
        gamma = 0.001*np.matrix(np.identity(self.cov_f_grid.shape[0]))
        L = np.linalg.cholesky(self.cov_f_grid + gamma*np.eye(self.cov_f_grid.shape[0]))
        
        u = np.random.randn(self.cov_f_grid.shape[0],1)   # np.random.randn generates guassian noise in a normal distribution (mean is 0, variance is 1)  
        self.f_sim = self.m_f_grid+L*u
        
class Vizualization():
    '''This is part 3. Vizualizes output from part 2.'''
    
    def __init__(self,f_sim):
            
        # set bounds of the visualization to various geo parts 
        topleftX = geo[0]
        topleftY = geo[3]
        upperRightX = topleftX+1000
        lowerLeftY = topleftY-1000
                  
        #Extracting grid points for extent of the plot            
        x = Xgrid[:,0]
        y = Xgrid[:,1]
        
        image = f_sim.reshape(GRANULARITY,GRANULARITY)
        
        #plotting points within the grid
        for i in range(len(Xtrain)):
            if (Xtrain[i][0]<topleftY and Xtrain[i][0]>lowerLeftY and Xtrain[i][1]>topleftX and Xtrain[i][1]<upperRightX):
                print 'point within grid'
                plt.scatter(Xtrain[i][0],Xtrain[i][1])
        for i in range(len(Xtest)):        
            if (Xtest[i][0]<topleftY and Xtest[i][0]>lowerLeftY and Xtest[i][1]>topleftX and Xtest[i][1]<upperRightX):
                print 'point within grid again'
                plt.scatter(Xtest[i][0],Xtest[i][1])
        
        ax=plt.imshow(image,extent=[x.max(), x.min(), y.max(), y.min()])
        plt.colorbar(ax)
        plt.savefig('test_150.jpg', bbox_inches='tight')
        
        import simplekml
        kml = simplekml.Kml()
        ground = kml.newgroundoverlay(name='GroundOverlay')
        ground.icon.href = 'test.jpg'
        ground.latlonbox.north = topleftY
        ground.latlonbox.south = lowerLeftY
        ground.latlonbox.east =  upperRightX
        ground.latlonbox.west =  topleftX
        kml.save('GroundOverlay.kml')

def main():
    prediction = Predictions()
    simulation = Simulation(prediction.OPTIMAL_H,prediction.OPTIMAL_SIGMA_2)
    f_sim = simulation.f_sim
    vizualization = Vizualization(f_sim)
    return prediction.m_f, simulation.m_f_grid, simulation.other, simulation.Y

m_f, m_f_grid, other, Y  = main()


sys.exit(1)
f_sim = main()
print 'this is f_sim \n' ,f_sim
print type(f_sim)
print 'the shape is ', f_sim.shape

##### HELPER FUNCTIONS THAT ARE RARELY USED ####

def getit():
    k=21
    Xsamples = {}
    Ysamples = {} 
    for n in range(1,k+1): 
        index_1 = ((n-1)*len(Xtrain))/k
        index_2 = (n*len(Xtrain))/k
        Xsamples['Xsample_'+str(n)] = Xtrain[index_1:index_2]
        Ysamples['Ysample_'+str(n)] = Ytrain[index_1:index_2]
    return Xsamples,Ysamples

#okay so it works?  
