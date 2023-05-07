import os
import sys
import pickle

import numpy as np
import pandas as pd

from datetime import datetime

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

from main_folder.exception import CustomException
from main_folder.logger import logging

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        
        os.makedirs(dir_path,exist_ok= True)
        
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            
            #Train model
            model.fit(X_train,y_train)
            
            # Predict Testing data
            y_test_pred = model.predict(X_test)
            
            #Get R2 scores for train and test data
            # train_data_score = r2_score(y_train,y_train_ped)
            
            test_model_score = r2_score(y_test,y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
            
        return report
    
    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    
def drop_null(df_copy,column):
    df_copy = df_copy.dropna(subset=[column])
    df_copy = df_copy.reset_index(drop=True)
    return df_copy

def drop_nonformat_time(df_copy,column):
    list1 = []

    for i in range(len(df_copy[column])):
        if '.' in df_copy[column][i] :
            list1.append(i)
        elif len(df_copy[column][i]) <= 2:
            list1.append(i)

    df_copy.drop(list1,axis=0,inplace=True)
    df_copy = df_copy.reset_index(drop=True)
    
    return df_copy
        
def convert_into_datetime(df_copy,column):
    arr = df_copy[column].values

    drop_values = []

    # Convert the objects to datetime objects
    arr_datetime = []
    for obj in arr:
            if len(obj) == 5:
                obj += ':00'
            try:
                arr_datetime.append(datetime.strptime(obj, '%H:%M:%S'))
            except:
                drop_values.append(obj)
                continue
    
    df_copy = df_copy.drop(df_copy[df_copy[column].isin(drop_values)].index)

    # Convert the datetime objects to the desired format
    arr_strftime = [obj.strftime('%H:%M:%S') for obj in arr_datetime]

    # Replace the original column with the new column
    df_copy[column] = arr_strftime
    
    return df_copy
                    
                    
def count_diff(df_copy,column1,column2,column3):
    df_copy[column1] = pd.to_datetime(df_copy[column1], format='%H:%M:%S')
    df_copy[column2] = pd.to_datetime(df_copy[column2], format='%H:%M:%S')
    df_copy[column3] = (df_copy[column2] - df_copy[column1]).dt.total_seconds() / 60
    df_copy[column3] = df_copy[column3].astype(np.int64)
    return df_copy

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)
