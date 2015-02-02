#Jordan Makansi
#CE 263N - Bike_share Analysis

import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn import linear_model
import csv

# Get training data
trn_data = np.genfromtxt('midterm_train.csv', delimiter=',', skiprows=1, usecols=(1,2,3,4,5,6,7,8,9,10,11,12))
year_trn = trn_data[:,0].T
month_trn = trn_data[:,1].T
hour_trn = trn_data[:,2].T
holiday_trn = trn_data[:,3].T
weekday_trn= trn_data[:,4].T
working_trn = trn_data[:,5].T
weather_type_trn = trn_data[:,6].T
temp_trn = trn_data[:,7].T
feels_like_trn = trn_data[:,8].T
humidity_trn = trn_data[:,9].T
windspeed_trn = trn_data[:,10].T
count_trn = trn_data[:,11]

# Get testing data
test_data = np.genfromtxt('midterm_test.csv', delimiter=',', skiprows=1, usecols=(1,2,3,4,5,6,7,8,9,10,11))
year_test = test_data[:,0].T
month_test = test_data[:,1].T
hour_test = test_data[:,2].T
holiday_test = test_data[:,3].T
weekday_test= test_data[:,4].T
working_test = test_data[:,5].T
weather_type_test = test_data[:,6].T
temp_test = test_data[:,7].T
feels_like_test = test_data[:,8].T
humidity_test = test_data[:,9].T
windspeed_test = test_data[:,10].T

#Prediction 1 - Effect of Temperature, Humidity, Windspeed (6,7,8,9,10)
trn = trn_data[:,[6,7,8,9,10]]
clf = DecisionTreeRegressor(max_depth=5, min_samples_leaf=5)
clf.fit(trn,count_trn)

neigh = KNeighborsClassifier(n_neighbors=10, weights='distance')
trn = trn_data[:,[8,9]]
test = test_data[:,[8,9]]
neigh.fit(trn, count_trn)
predict_1 = neigh.predict(test)

#Prediction 2 - Effect of hour, weekday/weekend (2,4)
trn = trn_data[:,[2,4,11]]
test = test_data[:,[2,4]]
predict_2 = []
cluster_centers = []

# For each hour, create a list of the counts 

for hour in range(24):
    x=[]
    for row in range(len(trn)):
            if trn[row][0]==hour:
                x.append(trn[row][2])
   
    # create a K-means cluster object, with 2 clusters (weekend and weekday)
    k_means = KMeans(init='k-means++', n_clusters=2, n_init=10)
    result = k_means.fit(np.array(x)[np.newaxis].T)
    
    cluster_centers.append(k_means.cluster_centers_)

for row in range(len(test)):
    
    for hour in range(24):
       if test[row][0]==hour:
           if test[row][1]==6 or test[row][1]==7:
               predict_2.append(min(cluster_centers[hour][0],cluster_centers[hour][1]))
           else:
               predict_2.append(max(cluster_centers[hour][0],cluster_centers[hour][1]))

#Prediction 3 - Effect of year, month (0,1)
trn = trn_data[:,[0,1,11]]
test = test_data[:,[0,1]]
x=[]
y=[]
predict_3 = []
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




#Prediction 4 - Effect of holiday in December (3)
trn = trn_data[:,[11,1,3,5]]
test = test_data[:,[1,3,5]]
predict_4 = []
x=[]
for row in range(len(trn)):
    if trn[row][1]==12 and trn[row][2]==1:
            x.append(trn[row][0])
k_means = KMeans(init='k-means++', n_clusters=1, n_init=10)
k_means.fit(np.array(x)[np.newaxis].T)
dec_hol = k_means.cluster_centers_
#print dec_hol
for row in range(len(test)):
    if test[row][1]==1:
        predict_4.append(dec_hol)
    else:
        predict_4.append(0)

#Weights from decision tree
trn = trn_data[:,[0,1,2,3,4,8,9,10]]

clf = DecisionTreeRegressor(min_samples_leaf=5)
clf.fit(trn,count_trn)
weight_1 = clf.feature_importances_
#print clf.feature_importances_

trn = trn_data[:,[0,1,2,4,8,9,10]]
clf = DecisionTreeRegressor(min_samples_leaf=5)
clf.fit(trn,count_trn)
weight_2 = clf.feature_importances_

#  ----------------- UP TO HERE ----------------------------

#Final prediction
test = test_data[:,[0,1,2,3,4,5,6,7,8,9,10]]
count_predict = []
for row in range(len(test)):
    if test[row][3]==1:
        count_predict.append(predict_1[row]*(weight_1[5]+weight_1[6]+weight_1[7])+predict_2[row]*(weight_1[2]+weight_1[4])+predict_3[row]*(weight_1[0]+weight_1[1])+predict_4[row]*weight_1[3])
    else:
        count_predict.append(predict_1[row]*(weight_2[4]+weight_2[5]+weight_2[6])+predict_2[row]*(weight_2[2]+weight_2[3])+predict_3[row]*(weight_2[0]+weight_2[1]))


#Writing to a file
date = []
with open('midterm_test.csv','rb') as f:
    next(f)
    csvreader = csv.reader(f,delimiter = ',')
    for row in csvreader:
        date.append(row[0])
f.close()
hour = np.genfromtxt('midterm_test.csv', delimiter=',', skiprows=1, usecols=(3))
with open('prediction1.csv','w') as f:
    f.write("date,hour,count\n")
    for i in range(len(count_predict)):
        f.write("%s,%f,%f\n" % (date[i],hour[i],round(count_predict[i])))
    f.close()

###### -------- DEBUGGING BREAK HERE ----
import sys
sys.exit()
