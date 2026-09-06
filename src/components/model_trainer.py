import os
import sys

import numpy as np

from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from src.pipeline.exception import CustomException
from src.pipeline.logger import logging
from src.pipeline.utils import save_object


@dataclass
class ModelTrainerConfig:

    trained_model_file_path: str = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTrainer:

    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):

        try:

            logging.info("Starting model training")


            # --------------------------------------------------
            # Separate features and target
            # --------------------------------------------------

            X_train = train_array[:, :-1]
            y_train = train_array[:, -1]

            X_test = test_array[:, :-1]
            y_test = test_array[:, -1]

            logging.info(
                "Train and test data separated"
            )


            # --------------------------------------------------
            # Define regression models
            # --------------------------------------------------

            models = {

                "Linear Regression":
                    LinearRegression(),

                "Decision Tree":
                    DecisionTreeRegressor(
                        random_state=42
                    ),

                "Random Forest":
                    RandomForestRegressor(
                        n_estimators=200,
                        random_state=42
                    ),

                "Gradient Boosting":
                    GradientBoostingRegressor(
                        random_state=42
                    )
            }


            model_report = {}


            # --------------------------------------------------
            # Train and evaluate models
            # --------------------------------------------------

            for model_name, model in models.items():

                logging.info(
                    f"Training {model_name}"
                )

                print(
                    f"\nTraining {model_name}..."
                )


                # Train model
                model.fit(
                    X_train,
                    y_train
                )


                # Make predictions
                y_pred = model.predict(
                    X_test
                )


                # Calculate MAE
                mae = mean_absolute_error(
                    y_test,
                    y_pred
                )


                # Calculate RMSE
                rmse = np.sqrt(
                    mean_squared_error(
                        y_test,
                        y_pred
                    )
                )


                # Calculate R2
                r2 = r2_score(
                    y_test,
                    y_pred
                )


                # Store results
                model_report[model_name] = {

                    "MAE": mae,

                    "RMSE": rmse,

                    "R2": r2
                }


                # Display results
                print(
                    f"{model_name}"
                )

                print(
                    f"MAE  : {mae:.4f}"
                )

                print(
                    f"RMSE : {rmse:.4f}"
                )

                print(
                    f"R2   : {r2:.4f}"
                )


                logging.info(
                    f"{model_name} - "
                    f"MAE: {mae:.4f}, "
                    f"RMSE: {rmse:.4f}, "
                    f"R2: {r2:.4f}"
                )


            # --------------------------------------------------
            # Select best model
            # --------------------------------------------------

            best_model_name = max(
                model_report,
                key=lambda model_name:
                model_report[model_name]["R2"]
            )


            best_model = models[
                best_model_name
            ]


            best_r2 = model_report[
                best_model_name
            ]["R2"]


            print(
                "\n======================================"
            )

            print(
                "Best Model:",
                best_model_name
            )

            print(
                "Best R2 Score:",
                round(best_r2, 4)
            )

            print(
                "======================================"
            )


            logging.info(
                f"Best model selected: "
                f"{best_model_name}"
            )


            # --------------------------------------------------
            # Save best model
            # --------------------------------------------------

            save_object(
                file_path=self.model_trainer_config
                .trained_model_file_path,

                obj=best_model
            )


            logging.info(
                "Best model saved successfully"
            )


            print(
                "\nBest model saved successfully!"
            )

            print(
                "Model saved at:",
                self.model_trainer_config
                .trained_model_file_path
            )


            return model_report


        except Exception as e:

            logging.info(
                "Exception occurred during model training"
            )

            raise CustomException(
                e,
                sys
            )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    try:

        # --------------------------------------------------
        # Step 1: Data Ingestion
        # --------------------------------------------------

        from src.components.data_ingestion import (
            DataIngestion
        )


        data_ingestion = DataIngestion()


        train_data, test_data = (
            data_ingestion.initiate_data_ingestion()
        )


        # --------------------------------------------------
        # Step 2: Data Transformation
        # --------------------------------------------------

        from src.components.data_transformation import (
            DataTransformation
        )


        data_transformation = DataTransformation()


        transformation_result = (
            data_transformation
            .initiate_data_transformation(
                train_data,
                test_data
            )
        )


        # --------------------------------------------------
        # IMPORTANT:
        # DataTransformation may return more than 2 values.
        # We only need the first two:
        #
        # [0] -> train_array
        # [1] -> test_array
        # --------------------------------------------------

        train_array = transformation_result[0]

        test_array = transformation_result[1]


        # --------------------------------------------------
        # Step 3: Model Training
        # --------------------------------------------------

        model_trainer = ModelTrainer()


        model_trainer.initiate_model_trainer(
            train_array,
            test_array
        )


    except Exception as e:

        raise CustomException(
            e,
            sys
        )