from main_folder.components.data_ingestion import DataIngestion
from main_folder.components.data_transformation import DataTransformtion
from main_folder.components.modal_trainer import ModelTrainer

import os
import sys
from main_folder.logger import logging
from main_folder.exception import CustomException



if __name__ == '__main__':
    obj = DataIngestion()
    train_data_path,train_data_path = obj.initiate_data_ingestion()
    data_transformation = DataTransformtion()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data_path,train_data_path)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr,test_arr)
    
    
            