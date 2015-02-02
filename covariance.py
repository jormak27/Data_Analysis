 # Jordan Covariance understanding 

import numpy as np
import random

some_array = random.normalvariate(0,100)

Asset1_SD = 0.2
Asset2_SD = 0.1
Asset3_SD = 0.15

# you can think of this as three variables
# so you will always get 6 values for covariance
Corr_12 = 0.8
Corr_13 = 0.5
Corr_23 = 0.3


Corrs = [Corr_12,Corr_13,Corr_23]

'''To construct a covariance matrix,
we will multiply a diagonal matrix holding standard deviations
and a matrix holding correlations.'''

#Diagonal matrix holding standard deviations
D = np.diag([0.2,0.1,0.15])

# Correlation matrix (NOT COVARIANCE)
''' You know it is a correlation matrix because there are 1's in
the diagonal.  Asset1 perfectly correlates with itself, as does Asset2, etc.'''

C = np.array([[1,0.8,0.5],
              [0.8,1.0,0.3],
              [0.5,0.3,1.0]])

D = np.matrix(D)
print D
C = np.matrix(C)
print C

# This is the covariance matrix, V
# Note that the covariance of an asset with itself is just the variance
# Remember that the variance is the Standard Deviation squared 

'''
Notice that the diagonal of this matrix, V is just the variances of each variable (standard deviation squared)
'''

V = D*C*D
print V
