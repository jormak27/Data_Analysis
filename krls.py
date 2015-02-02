#!/bin/env python
#
# File name: krls.py
# Copyright: (C) 2010 i2maps
# 
# Kernel Recursive Least Squares implementation.
#
# Initialization parameters:
#   
#   ktype           kernel type         <! probably kernels have to be higher level and a more independent class >  
#   kparam          kernel parameters   <! to allow a user defining an implicit function for a pair of data samples >
#
#   state           a data sample structure (array)
#   target          a target value (float)
#
#   adopt_thresh    approximate linear dependence threshold, [0, 1)
#   maxsize         maximum size of the dictionary 
#   adaptive        indicator if elimination is data-adaptive, True/False
#   forget_rate     frequency rate for forced elimination of the oldest entry, [0, 1]


import numpy as np
#import dictionary 


class krls:  #  aka InfoSphere Streams Killer

     
     def __init__(self, kernel_function, adopt_thresh, state, target, maxsize, adaptive=True, forget_rate=0.):
       
        self.dp = dict.dict(kernel_function, adopt_thresh, state, target, maxsize, adaptive, forget_rate)

        self.P = [[1]]
        self.Alpha = np.dot(self.dp.Kinv, target)


     def update(self, state, target):
        
        # dictionary update preceeds weights update. They are not independent:
        # dictionary needs to know weights Alpha to enable adaptive elimination
        self.dp.update( state, target, self.Alpha )
        
        # now we update the weights using values precomputed in dicionary
        # to enable recursive updates here
        at = self.dp.at
        dt = self.dp.dt
        ktwid = self.dp.ktwid
                    
        if self.dp.addedFlag: # if a new entry was added  to the dictionary
          
          if self.dp.eliminatedFlag: # and an older/unrelevant entry was eliminated to make some room
          
              # I suspect the smart tricks got partly waisted here by matrix multiply
              # cause weights only become dependent on dictionary and not on all incoming samples.
              # Still need to figure out what's happening.
                                 
              self.Alpha = np.dot(self.dp.Kinv, self.dp.Targ)
              
              
          else:  # was enough room, so update as per original paper
          
              self.P = np.vstack( [np.hstack([self.P, np.zeros((self.dp.numel-1,1))]), np.hstack( [np.zeros((1,self.dp.numel-1)), [[1]]] ) ] )
              inno = ( target - np.dot(ktwid.T,self.Alpha) )/dt         
              self.Alpha = np.vstack([self.Alpha - np.dot(at,inno), inno])
              self.Alpha = np.dot(self.dp.Kinv, self.dp.Targ)
             
              self.addedFlag = 1;
        
        else:    # we don't add an entry but update weights not to waste the sample 
                  # kinda smart incremental reduced rank regression.
          
              tmp = np.dot(self.P, at)
              qt = tmp / ( 1 + np.dot(at.T,tmp) )
              self.P = self.P - np.dot(qt,tmp.T)
              self.Alpha = self.Alpha + np.dot(self.dp.Kinv, qt*( target - np.dot(ktwid.T,self.Alpha) ))
              self.addedFlag = 0
          
    
    
     def query(self, sample):
               
        # compute the kernel of the input with the dictionary
        kernvals = self.dp.query(sample)
        # compute the weighted sum
        target =  np.dot(kernvals, self.Alpha)
          
        return target
