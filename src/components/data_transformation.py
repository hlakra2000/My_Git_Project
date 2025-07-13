import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

from src.logger import logging
from src.exception import CustomException

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def perform_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info("Starting feature engineering")

            df.drop(columns=['precip_range'], inplace=True, errors='ignore')
            df['precipitation'] = df['precipitation'].fillna(0)

            df['pickup_date'] = pd.to_datetime(df['pickup_date'], dayfirst=True)
            df['dropoff_date'] = pd.to_datetime(df['dropoff_date'], dayfirst=True)
            df['pickup_time'] = pd.to_datetime(df['pickup_time'], format="%H:%M:%S", errors='coerce')
            df['dropoff_time'] = pd.to_datetime(df['dropoff_time'], format="%H:%M:%S", errors='coerce')

            df['pickup_day_of_week'] = df['pickup_date'].dt.day_name()
            df['pickup_day'] = df['pickup_date'].dt.day
            df['pickup_month'] = df['pickup_date'].dt.month
            df['dropoff_day'] = df['dropoff_date'].dt.day
            df['dropoff_month'] = df['dropoff_date'].dt.month

            day_mapping = {
                'Monday': 1, 'Tuesday': 2, 'Wednesday': 3,
                'Thursday': 4, 'Friday': 5, 'Saturday': 6, 'Sunday': 7
            }
            df['pickup_day_of_week_num'] = df['pickup_day_of_week'].map(day_mapping)

            df['pickup_hour'] = df['pickup_time'].dt.hour
            df['pickup_minute'] = df['pickup_time'].dt.minute
            df['dropoff_hour'] = df['dropoff_time'].dt.hour
            df['dropoff_minute'] = df['dropoff_time'].dt.minute

            if 'trip_duration(min)' in df.columns:
                df['trip_duration(min)'] = df['trip_duration(min)'].round(2)

            def categorize_congestion(speed):
                if speed > 20:
                    return 'Low'
                elif speed > 10:
                    return 'Medium'
                else:
                    return 'High'

            df['congestion_level'] = df['avg_speed_kmh'].apply(categorize_congestion)
            df['congestion_level'] = df['congestion_level'].str.strip().str.lower()
            mapping = {'low': 0, 'medium': 1, 'high': 2}
            df['congestion_level_numeric'] = df['congestion_level'].map(mapping)

            df['store_and_fwd_flag'] = df['store_and_fwd_flag'].map({'N': 0, 'Y': 1})

            for col in ['precipitation', 'snow fall', 'snow depth']:
                df[col] = df[col].replace('T', 0.001).astype(float)

            df['is_weekend'] = df['pickup_day_of_week_num'].isin([5, 6]).astype(int)
            df['is_peak_hour'] = df['pickup_hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)

            from sklearn.cluster import KMeans
            pickup_coords = df[['pickup_latitude', 'pickup_longitude']]
            kmeans_pickup = KMeans(n_clusters=10, random_state=42).fit(pickup_coords)
            df['pickup_cluster'] = kmeans_pickup.labels_

            dropoff_coords = df[['dropoff_latitude', 'dropoff_longitude']]
            kmeans_dropoff = KMeans(n_clusters=10, random_state=42).fit(dropoff_coords)
            df['dropoff_cluster'] = kmeans_dropoff.labels_

            df['rush_hour_intensity'] = df['pickup_hour'].apply(lambda x: 
                3 if x in [8, 9, 17, 18] else 
                2 if x in [7, 10, 16, 19] else 1)

            logging.info("Feature engineering completed")
            return df

        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self, df: pd.DataFrame) -> ColumnTransformer:
        try:
            logging.info("Identifying numerical and categorical columns")

            numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            if 'congestion_level_numeric' in numerical_cols:
                numerical_cols.remove('congestion_level_numeric')
            if 'congestion_level_numeric' in categorical_cols:
                categorical_cols.remove('congestion_level_numeric')

            logging.info(f"Numerical columns: {numerical_cols}")
            logging.info(f"Categorical columns: {categorical_cols}")

            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                ('scaler', StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer(transformers=[
                ('num', num_pipeline, numerical_cols),
                ('cat', cat_pipeline, categorical_cols)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logging.info("Reading training and test datasets")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Performing feature engineering")
            train_df = self.perform_feature_engineering(train_df)
            test_df = self.perform_feature_engineering(test_df)

            target_column = 'congestion_level_numeric'
            drop_columns = [
                'congestion_level','id','congestion_level_numeric','pickup_day_of_week','avg_speed_kmh','trip_duration(min)',
                'dropoff_day','dropoff_month', 'dropoff_hour','dropoff_minute','distance_km',
                'pickup_date','dropoff_date','pickup_time','dropoff_time'
            ]
            if target_column not in train_df.columns or target_column not in test_df.columns:
                raise CustomException(f"Target column '{target_column}' not found in input data", sys)

            # Extract target and drop columns
            y_train = train_df[target_column]
            y_test = test_df[target_column]

            X_train = train_df.drop(columns=drop_columns, errors='ignore')
            X_test = test_df.drop(columns=drop_columns, errors='ignore')

            preprocessing_obj = self.get_data_transformer_object(X_train)

            logging.info("Applying preprocessing to train/test sets")
            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)

            joblib.dump(preprocessing_obj, self.data_transformation_config.preprocessor_obj_file_path)
            logging.info(f"Saved preprocessor object to: {self.data_transformation_config.preprocessor_obj_file_path}")

            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
