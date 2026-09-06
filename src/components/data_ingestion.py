import os
import sys

import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.pipeline.exception import CustomException
from src.pipeline.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join(
        "artifacts",
        "train.csv"
    )

    test_data_path: str = os.path.join(
        "artifacts",
        "test.csv"
    )

    raw_data_path: str = os.path.join(
        "src",
        "pipeline",
        "data",
        "student-por.csv"
    )


class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        logging.info(
            "Entered the data ingestion method or component"
        )

        try:
            # Read the raw dataset
            df = pd.read_csv(
                self.ingestion_config.raw_data_path
            )

            logging.info(
                "Read the dataset as dataframe"
            )

            # Create artifacts directory
            os.makedirs(
                os.path.dirname(
                    self.ingestion_config.train_data_path
                ),
                exist_ok=True
            )

            logging.info(
                "Train test split initiated"
            )

            # Split the dataset
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            logging.info(
                "Train test split completed"
            )

            # Save training data
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False
            )

            # Save testing data
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False
            )

            logging.info(
                "Ingestion of data is completed"
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info(
                "Exception occurred during data ingestion"
            )

            raise CustomException(e, sys)


if __name__ == "__main__":

    obj = DataIngestion()

    train_data, test_data = obj.initiate_data_ingestion()

    print("Data ingestion completed successfully!")

    print("Training data:", train_data)
    print("Testing data:", test_data)