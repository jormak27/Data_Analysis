#Jordan Makansi
#CE 263N - Midterm (Bike sharing)
#11 - 4 - 2014

# Imports here:
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import pandas as pd 

#Get training and testing data
def getData():
    trn_data = pd.read_csv('midterm_train.csv')   
    test_data = pd.read_csv('midterm_test.csv')     
    return trn_data, test_data

trn_data_df, test_data_df = getData()

def getCols(dFrame, start_col, finish_col):
    #Get nearest neighbor attributes (numbers 6, 7, etc. correspond to above attributes)
    columns = dFrame.columns[start_col:finish_col]
    # return the dataframe as an array
    print 'this is the DataFrame\n ', dFrame.loc[:,columns]
    return np.array(dFrame.loc[:,columns])

def DTL(Data,max_depth=5,min_samples_leaf=5):
    '''This function takes in the Data, to be analyzed by the Decision tree.
        Input: Data (a 2-D array)  1st dimension is a time series by hour.
        2nd dimension is different variables that affect count of bikers.
        Output: Classifications that explain the most variability'''
        
    #Decisiontree to no importance  
    clf = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    count_trn = trn_data_df.loc[:,'count']
    #print type(count_trn)
    #print count_trn
    #print np.array(count_trn)
    print 'Data is \n', Data
    print 'count_trn is \n', np.array(count_trn)
    clf.fit(Data,np.array(count_trn))
    #prints the importances of each features 
    print clf.feature_importances_
    return clf.feature_importances_
'''
Data = getCols(trn_data_df,7,12)
features = DTL(Data)
print 'the features are ', features
'''

#KNeighbors to predict counts by inorporated nearest neighbor of both 
neigh = KNeighborsClassifier(n_neighbors=10, weights='distance')
trn = trn_data[:,[8,9]]
test = test_data[:,[8,9]]
neigh.fit(trn, count_trn)
#print(neigh.predict(test))

trn = trn_data[:,[11,2]]

for hour in range(24):
    x=[]
    for row in range(len(trn)):
            if trn[row][1]==hour:
                    x.append(trn[row][0])
    k_means = KMeans(init='k-means++', n_clusters=2, n_init=10)
    k_means.fit(np.array(x)[np.newaxis].T)
    print k_means.cluster_centers_

trn = trn_data[:,[8,9,2,3,4]]
clf = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5)
clf.fit(trn,count_trn)
#print clf.feature_importances_
