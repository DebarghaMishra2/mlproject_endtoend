import os
import sys

import pandas as pd

from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.pipeline.exception import CustomException
from src.pipeline.logger import logging


@dataclass
class DataTransformationConfig:

    preprocessor_obj_file_path: str = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        """
        This function is responsible for creating the data transformation
        pipeline for numerical and categorical features.
        """

        try:

            # Numerical features from the dataset
            numeric_features = [
                "age",
                "Medu",
                "Fedu",
                "traveltime",
                "studytime",
                "failures",
                "schoolsup",
                "famsup",
                "paid",
                "activities",
                "nursery",
                "higher",
                "internet",
                "romantic",
                "famrel",
                "freetime",
                "goout",
                "Dalc",
                "Walc",
                "health",
                "absences",
                "G1",
                "G2",
                "grade_improvement",
                "alcohol_score",
                "avg_alcohol",
                "social_activity_score",
                "parent_education_avg",
                "parent_education_gap",
                "academic_risk"
            ]

            # Categorical features from the dataset
            categorical_features = [
                "school",
                "sex",
                "address",
                "famsize",
                "Pstatus",
                "Mjob",
                "Fjob",
                "reason",
                "guardian",
                "absence_level"
            ]

            logging.info(
                "Numerical features identified: %s",
                numeric_features
            )

            logging.info(
                "Categorical features identified: %s",
                categorical_features
            )

            # Numerical Pipeline
            numeric_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="median")
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            )

            # Categorical Pipeline
            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(handle_unknown="ignore")
                    )
                ]
            )

            logging.info(
                "Numerical pipeline created successfully"
            )

            logging.info(
                "Categorical pipeline created successfully"
            )

            # Combine both pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num",
                        numeric_pipeline,
                        numeric_features
                    ),
                    (
                        "cat",
                        categorical_pipeline,
                        categorical_features
                    )
                ]
            )

            logging.info(
                "Preprocessor object created successfully"
            )

            return preprocessor

        except Exception as e:

            logging.info(
                "Exception occurred while creating preprocessor object"
            )

            raise CustomException(e, sys)

    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):

        """
        This function reads the training and testing data,
        applies the preprocessing pipeline, and saves the
        preprocessor object.
        """

        try:

            logging.info(
                "Starting data transformation"
            )

            # Read training and testing datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(
                "Training and testing data loaded successfully"
            )

            # -------------------------------
            # Feature Engineering
            # -------------------------------

            # Grade improvement
            train_df["grade_improvement"] = (
                train_df["G2"] - train_df["G1"]
            )

            test_df["grade_improvement"] = (
                test_df["G2"] - test_df["G1"]
            )

            # Absence level
            train_df["absence_level"] = pd.cut(
                train_df["absences"],
                bins=[-1, 0, 5, 10, 20, 100],
                labels=[
                    "No Absence",
                    "Low",
                    "Moderate",
                    "High",
                    "Very High"
                ]
            )

            test_df["absence_level"] = pd.cut(
                test_df["absences"],
                bins=[-1, 0, 5, 10, 20, 100],
                labels=[
                    "No Absence",
                    "Low",
                    "Moderate",
                    "High",
                    "Very High"
                ]
            )

            # Alcohol score
            train_df["alcohol_score"] = (
                train_df["Dalc"] + train_df["Walc"]
            )

            test_df["alcohol_score"] = (
                test_df["Dalc"] + test_df["Walc"]
            )

            # Average alcohol consumption
            train_df["avg_alcohol"] = (
                train_df["Dalc"] + train_df["Walc"]
            ) / 2

            test_df["avg_alcohol"] = (
                test_df["Dalc"] + test_df["Walc"]
            ) / 2

            # Social activity score
            train_df["social_activity_score"] = (
                train_df["freetime"] + train_df["goout"]
            )

            test_df["social_activity_score"] = (
                test_df["freetime"] + test_df["goout"]
            )

            # Parent education average
            train_df["parent_education_avg"] = (
                train_df["Medu"] + train_df["Fedu"]
            ) / 2

            test_df["parent_education_avg"] = (
                test_df["Medu"] + test_df["Fedu"]
            ) / 2

            # Parent education gap
            train_df["parent_education_gap"] = (
                train_df["Medu"] - train_df["Fedu"]
            ).abs()

            test_df["parent_education_gap"] = (
                test_df["Medu"] - test_df["Fedu"]
            ).abs()

            # Academic risk
            train_df["academic_risk"] = (
                train_df["failures"] + train_df["absences"]
            )

            test_df["academic_risk"] = (
                test_df["failures"] + test_df["absences"]
            )

            logging.info(
                "Feature engineering completed"
            )

            # -------------------------------
            # Convert binary columns
            # -------------------------------

            binary_columns = [
                "schoolsup",
                "famsup",
                "paid",
                "activities",
                "nursery",
                "higher",
                "internet",
                "romantic"
            ]

            for col in binary_columns:

                train_df[col] = train_df[col].map(
                    {
                        "yes": 1,
                        "no": 0
                    }
                )

                test_df[col] = test_df[col].map(
                    {
                        "yes": 1,
                        "no": 0
                    }
                )

            logging.info(
                "Binary columns converted to numerical values"
            )

            # -------------------------------
            # Separate target variable
            # -------------------------------

            target_column_name = "G3"

            X_train = train_df.drop(
                target_column_name,
                axis=1
            )

            y_train = train_df[target_column_name]

            X_test = test_df.drop(
                target_column_name,
                axis=1
            )

            y_test = test_df[target_column_name]

            logging.info(
                "Input features and target variable separated"
            )

            # -------------------------------
            # Create preprocessor
            # -------------------------------

            preprocessing_obj = self.get_data_transformer_object()

            logging.info(
                "Applying preprocessing object to training data"
            )

            # Fit ONLY on training data
            X_train_transformed = preprocessing_obj.fit_transform(
                X_train
            )

            logging.info(
                "Training data transformation completed"
            )

            logging.info(
                "Applying preprocessing object to testing data"
            )

            # Only transform testing data
            X_test_transformed = preprocessing_obj.transform(
                X_test
            )

            logging.info(
                "Testing data transformation completed"
            )

            # -------------------------------
            # Save preprocessor
            # -------------------------------

            os.makedirs(
                os.path.dirname(
                    self.data_transformation_config
                    .preprocessor_obj_file_path
                ),
                exist_ok=True
            )

            import pickle

            with open(
                self.data_transformation_config
                .preprocessor_obj_file_path,
                "wb"
            ) as file_obj:

                pickle.dump(
                    preprocessing_obj,
                    file_obj
                )

            logging.info(
                "Preprocessor object saved successfully"
            )

            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test,
                self.data_transformation_config
                .preprocessor_obj_file_path
            )

        except Exception as e:

            logging.info(
                "Exception occurred during data transformation"
            )

            raise CustomException(e, sys)


if __name__ == "__main__":

    try:

        obj = DataTransformation()

        train_path = os.path.join(
            "artifacts",
            "train.csv"
        )

        test_path = os.path.join(
            "artifacts",
            "test.csv"
        )

        (
            X_train_transformed,
            X_test_transformed,
            y_train,
            y_test,
            preprocessor_path
        ) = obj.initiate_data_transformation(
            train_path,
            test_path
        )

        print(
            "Data transformation completed successfully!"
        )

        print(
            "Transformed training data shape:",
            X_train_transformed.shape
        )

        print(
            "Transformed testing data shape:",
            X_test_transformed.shape
        )

        print(
            "Preprocessor saved at:",
            preprocessor_path
        )

    except Exception as e:

        print(e)