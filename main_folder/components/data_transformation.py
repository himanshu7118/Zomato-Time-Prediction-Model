import os
import sys
from main_folder.logger import logging
from main_folder.exception import CustomException
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from dataclasses import dataclass
from main_folder.utils import save_object,drop_null,drop_nonformat_time,convert_into_datetime,count_diff

@dataclass
class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
    
class DataTransformtion:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()
        
    def get_data_transformation_object(self):
        try:
            logging.info('Data tranformation Initiated')
            categorical_columns = ['Weather_conditions', 'Road_traffic_density', 'Type_of_order',
                'Type_of_vehicle', 'Festival', 'City']
            numerical_columns = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude',
                'Restaurant_longitude', 'Delivery_location_latitude',
                'Delivery_location_longitude', 'Vehicle_condition',
                'multiple_deliveries', 'Time_diff']

            Weather_conditions_categories = ['Sunny', 'Cloudy', 'Windy','Sandstorms','Stormy','Fog']
            Road_traffic_density_categories = ['Low','Medium','High','Jam']
            Type_of_order_categories = ['Drinks', 'Snack', 'Meal','Buffet']
            Type_of_vehicle_categories = ['electric_scooter', 'scooter', 'motorcycle']
            Festival_categories = ['No', 'Yes']
            City_categories = ['Semi-Urban','Urban','Metropolitian']
            
            logging.info('Pipeline Intiated')
            
            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
                ]
            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[Weather_conditions_categories,Road_traffic_density_categories,Type_of_order_categories,Type_of_vehicle_categories,Festival_categories,City_categories])),
                ('scaler',StandardScaler())
                ]
            )

            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_columns),
            ('cat_pipeline',cat_pipeline,categorical_columns)
            ])
            
            logging.info('Pipeline Completed')
            
            return preprocessor
            
            
        except Exception as e:
            logging.info('Error in data transformation')
            raise CustomException(e,sys)
        
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info('read train and test data completed')
            logging.info(f'Train Dataframe Head: \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head: \n{test_df.head().to_string()}')
            
            logging.info('Obtaining preprocessing object')
            
            train_df = drop_null(train_df,'Time_Orderd')
            train_df = drop_null(train_df,'Time_Order_picked')
            
            test_df = drop_null(test_df,'Time_Orderd')
            test_df = drop_null(test_df,'Time_Order_picked')
            
            train_df = drop_nonformat_time(train_df,'Time_Orderd')
            train_df = drop_nonformat_time(train_df,'Time_Order_picked')
            
            test_df = drop_nonformat_time(test_df,'Time_Orderd')
            test_df = drop_nonformat_time(test_df,'Time_Order_picked')
            
            train_df = convert_into_datetime(train_df,'Time_Orderd')
            train_df = convert_into_datetime(train_df,'Time_Order_picked')
            
            test_df = convert_into_datetime(test_df,'Time_Orderd')
            test_df = convert_into_datetime(test_df,'Time_Order_picked')
            
            train_df = count_diff(train_df,'Time_Orderd','Time_Order_picked','Time_diff')
            test_df = count_diff(test_df,'Time_Orderd','Time_Order_picked','Time_diff')
            
            drop_columns = ['Time_Orderd','Time_Order_picked']
            train_df.drop(columns=drop_columns,axis=1)
            test_df.drop(columns=drop_columns,axis=1)
            
            preprocessor_obj = self.get_data_transformation_object()
            
            target_column_name = 'Time_taken (min)'
            drop_columns = [target_column_name]
            
            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            #Transforming using preprocessor object
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            logging.info('Applying preprocesscing object on training and testing datasets.')
            
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            logging.info('Preprocessor pickle file saved')
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            logging.info("Exception Occured in the initiate data transformation")
            raise CustomException(e,sys)
       
            
          
          
          
          



