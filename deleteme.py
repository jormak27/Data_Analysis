import numpy as np 
import pandas as pd 

def getData():
    #Get training and testing data
    trn_data = pd.read_csv('midterm_train.csv')   
    test_data = pd.read_csv('midterm_test.csv')     
    return trn_data, test_data

trn_data_df, test_data_df = getData()

def getCols(dFrame, start_col, finish_col):
    #Get nearest neighbor attributes (numbers 6, 7, etc. correspond to above attributes)
    columns = dFrame.columns[start_col:finish_col]
    # return the dataframe as an array
    v = dFrame.loc[:,columns]
    return v

trn0 = getCols(trn_data_df,1,4)
trn1 = getCols(trn_data_df,5,6)
trn2 = getCols(trn_data_df,9,12)

trnb = pd.concat((trn0,trn1,trn2),axis=1)
#print trnb
print np.array(trnb)
#print trnb.columns

