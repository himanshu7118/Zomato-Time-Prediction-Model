import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet

from sklearn.ensemble import RandomForestRegressor

from main_folder.exception import CustomException
from main_folder.logger import logging

from main_folder.utils import save_object,evaluate_model
import sys
import os

from dataclasses import dataclass

@dataclass
class ModalTrainConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModalTrainConfig()
        
    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('Aplitting Dependent and Independent variables from train and test')
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            models={
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'Elasticnet':ElasticNet(),
                "RandomForestRegressor":RandomForestRegressor(random_state=3)
            }
            
            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            
            print(model_report)
            print('\n====================================================')
            logging.info(f'Model Report : {model_report}')
            
            #TO get best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            print('\n==========================================================================')
            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')
            
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
            
            
        except Exception as e:
            logging.info('Exception occured at the Model training')
            raise CustomException(e,sys)
          