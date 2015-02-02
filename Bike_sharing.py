#Jordan Makansi
#CE 263N - Bike_share Analysis
# Standard imports

import matplotlib.pyplot as plt
import numpy as np
import csv
import sys 

# Import libraries used 
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn import linear_model
import pandas as pd 

# Global variables 
DEBUG = True  # if set to True, then print statements will inform the user 

#### ----------------------------------- Helper Functions ------------------------------- 

def getData():
    #Get training and testing data
    trn_data = pd.read_csv('midterm_train.csv')   
    test_data = pd.read_csv('midterm_test.csv')     
    return trn_data, test_data

trn_data_df, test_data_df = getData()

print trn_data_df

def getCols(dFrame, start_col, finish_col):
    #Get nearest neighbor attributes (numbers 6, 7, etc. correspond to above attributes)
    columns = dFrame.columns[start_col:finish_col]
    # return the dataframe as an array
    v = dFrame.loc[:,columns]
    return v

def DTL(Data,max_depth=5,min_samples_leaf=5):
    '''This function takes in the Data, to be analyzed by the Decision tree.
        Input: Data (a 2-D array)  1st dimension is a time series by hour.
        2nd dimension is different variables that affect count of bikers.
        Output: Classifications that explain the most variability'''
    #DecisionTree to no importance  
    clf = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    count_trn = getCols(trn_data_df,12,13)
    #Series(count_trn) 
    clf.fit(Data,count_trn)
    #convert count_trn to a Series, and then, an array
    count_Series = pd.Series(count_trn['count'])
    clf.fit(Data,np.array(count_Series))
    return clf.feature_importances_


#### ----------------------------------- Predictions ------------------------------- 


''' The predictions work in a three step process:
- Use DTL to determine which features explain the most variability,
and hence what weighting to assign to each of the 4 classifications of variables:
    1. Year and Month (Regression)
    2. Weather (Nearest Neighbor)
    3. Holiday/Workday (Clustering)
    4. Hour of day (Clustering)
- Conduct predictions. Clustering, Nearest Neighbor or Regression, depending on the variable.
Indicating in parentheses above.'''

#Prediction 1 - Effect of Temperature, Humidity, Windspeed (6,7,8,9,10)
def Prediction_Enviro_Variables():
    '''This function executes the prediction for Environmental variables.
    See Process Flow Diagram for details.'''
    Data = getCols(trn_data_df,7,12)  # This creates a new DataFrame, containing only the range of columns specified
    #...(in this case columns 7 through 11), because DataFrames have an extra 1st column for indexing 
    features = DTL(Data)
    if DEBUG == True:
        print 'the features of Environmental Variables are ', features

    #nearest neighbor prediction, weighted by distance 
    neigh = KNeighborsClassifier(n_neighbors=10, weights='distance')
    trn = getCols(trn_data_df,9,11)  # grab only the columns for "feels like" and "humidity"

    # i.e., only columns 9 and 10, same for test_data:
    test = getCols(test_data_df,9,11)
    count_trn = getCols(trn_data_df,12,13)    
    count_Series = pd.Series(count_trn['count'])

    # conduct NearestNeighbors prediction based on counts from training data  
    neigh.fit(trn,count_Series)
    predict_1 = neigh.predict(test)
    return predict_1

#Prediction 2 - Effect of hour, weekday/weekend (2,4), on counts (11)
def Prediction_Hour():
    '''This function tests the effects of hour and weekday
    (columns 2 and 4 in the DataFrame), on the counts (column 11).
    It outputs the count predictions by executing Kmeans Clustering '''

    # concatenate DataFrames to construct train data 
    trn0 = getCols(trn_data_df,3,4)
    trn1 = getCols(trn_data_df,5,6)
    trn2 = getCols(trn_data_df,12,13)
    
    trn = pd.concat((trn0,trn1,trn2),axis=1)
    trn =np.array(trn)
    
    # concatenate DataFrames to construct test data (this does not include the counts, obviously)
    test0 =  getCols(test_data_df,3,4)
    test1 = getCols(test_data_df,5,6)

    test = pd.concat((test0,test1),axis=1)
    test = np.array(test)

    predict_2 = []
    cluster_centers = []

    for hour in range(24):

        x=[]
        for row in range(len(trn)):
            if trn[row][0]==hour:
                x.append(trn[row][2])
                
        k_means = KMeans(init='k-means++', n_clusters=2, n_init=10)
        k_means.fit(np.array(x)[np.newaxis].T)
        
        cluster_centers.append(k_means.cluster_centers_)
    
    for row in range(len(test)):
        
        for hour in range(24):
           if test[row][0]==hour:
               if test[row][1]==6 or test[row][1]==7:
                   predict_2.append(min(cluster_centers[hour][0],cluster_centers[hour][1]))
               else:
                   predict_2.append(max(cluster_centers[hour][0],cluster_centers[hour][1]))
    return predict_2,cluster_centers
    
def Prediction_Year_Month():
    #Prediction 3 - Effect of year, month (0,1)
    trn0 = getCols(trn_data_df,1,3)
    trn1 = getCols(trn_data_df,12,13)
    trn = pd.concat((trn0,trn1),axis=1)
    trn = np.array(trn)
    
    test =  getCols(test_data_df,1,3)
    test = np.array(test)
    
    x=[]
    y=[]
    predict_3=[]
    for month in range(1,24):
        if (month ==20):
            continue
        count=[]
        for row in range(len(trn)):
            if (trn[row][0]*12+trn[row][1] == month):
                count.append(trn[row][2])
        x.append([month])
        y.append([np.average(count)])

    clf = linear_model.LinearRegression()
    clf.fit(x,y)

    for row in range(len(test)):
        
        if test[row][1]==8:
            predict_3.append(clf.intercept_+clf.coef_*(20))
        elif test[row][1]==12:
            predict_3.append(clf.intercept_+clf.coef_*(24))

    return predict_3,x,y

predict_3,x,y = Prediction_Year_Month()


def Prediction_Holiday():
    #Prediction 4 - Effect of holiday in December (3)

    trn0 = getCols(trn_data_df,12,13)
    trn1 = getCols(trn_data_df,2,3)
    trn2 = getCols(trn_data_df,4,5)
    trn3 = getCols(trn_data_df,6,7)

    
    trn = pd.concat((trn0,trn1,trn2,trn3),axis=1)
    trn = np.array(trn)

    test0 = getCols(test_data_df,2,3)
    test1 = getCols(test_data_df,4,5)
    test2 = getCols(test_data_df,6,7)

    test = pd.concat((test0,test1,test2),axis=1)
    test = np.array(test)
    
    predict_4 = []
    x=[]

    for row in range(len(trn)):
        if trn[row][1]==12 and trn[row][2]==1:
                x.append(trn[row][0])
    k_means = KMeans(init='k-means++', n_clusters=1, n_init=10)
    k_means.fit(np.array(x)[np.newaxis].T)
    dec_hol = k_means.cluster_centers_
    
    for row in range(len(test)):
        if test[row][1]==1:
            predict_4.append(dec_hol)
        else:
            predict_4.append(0)
    return predict_4,dec_hol

def DTL_weights():
    ''' This function takes in nothing, and executes a Decision Tree on the data from the
    ridership.
        Two decision trees are generated.
        This function outputs the weightings of the variables of these decision trees.'''
    
    count_trn = getCols(trn_data_df,12,13)
    count_trn = np.array(count_trn)
    
    trn0 = getCols(trn_data_df,1,6)
    trn1 = getCols(trn_data_df,9,11)
    trn2 = getCols(trn_data_df,11,12)
    print 'this is trn0 ', trn0
    print 'this is trn1 ', trn1
    print 'this is trn2 ', trn2
    sys.exit(1)
    
    trn = pd.concat((trn0,trn1,trn2),axis=1) 
    trn = np.array(trn) 
    
    #### Decision Tree Applied here #### 
    clf = DecisionTreeRegressor(min_samples_leaf=5)
    print 'this is trn ', trn
    print count_trn
    clf.fit(trn,count_trn)
    weight_1 = clf.feature_importances_
    ###  --- First Decision Tree Done ### 
    
    trn0 = getCols(trn_data_df,1,4)
    trn1 = getCols(trn_data_df,5,6)
    trn2 = getCols(trn_data_df,9,12)
    
    trnb = pd.concat((trn0,trn1,trn2),axis=1)
    trn = np.array(trnb)
    
    clf = DecisionTreeRegressor(min_samples_leaf=5)
    clf.fit(trn,count_trn)
    weight_2 = clf.feature_importances_

    return weight_1,weight_2

def Sum_predictions(predict_1,predict_2,predict_3,predict_4,weight_1,weight_2):
    '''Add the predictions''' 
    test = getCols(test_data_df,1,12)
    test = np.array(test)
    
    count_predict = []
    for row in range(len(test)):
        if test[row][3]==1:
            count_predict.append(predict_1[row]*(weight_1[5]+weight_1[6]+weight_1[7])+predict_2[row]*(weight_1[2]+weight_1[4])+predict_3[row]*(weight_1[0]+weight_1[1])+predict_4[row]*weight_1[3])
        else:
            count_predict.append(predict_1[row]*(weight_2[4]+weight_2[5]+weight_2[6])+predict_2[row]*(weight_2[2]+weight_2[3])+predict_3[row]*(weight_2[0]+weight_2[1]))
    return count_predict 


def Write_to_files(count_predict):
    '''this function writes the final counts to a csv file.''' 
    date = []
    with open('midterm_test.csv','rb') as f:
        next(f)
        csvreader = csv.reader(f,delimiter = ',')
        for row in csvreader:
            date.append(row[0])
    f.close()
    hour = np.genfromtxt('midterm_test.csv', delimiter=',', skiprows=1, usecols=(3))
    with open('prediction.csv','w') as f:
        f.write("date,hour,count\n")
        for i in range(len(count_predict)):
            f.write("%s,%f,%f\n" % (date[i],hour[i],round(count_predict[i])))
        f.close()

def Main():
    predict_1 = Prediction_Enviro_Variables()
    predict_2,cluster_centers = Prediction_Hour()
    predict_3,x,y = Prediction_Year_Month()
    predict_4,dec_hol = Prediction_Holiday()    
    weight_1,weight_2 = DTL_weights()
    count_predict = Sum_predictions(predict_1,predict_2,predict_3,predict_4,weight_1,weight_2)
    Write_to_files(count_predict)

Main()

