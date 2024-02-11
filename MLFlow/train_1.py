import argparse
import os
import pathlib
import warnings
from argparse import Namespace
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, LinearRegression
import mlflow
import mlflow.sklearn
import logging

warnings.filterwarnings("ignore")
np.random.seed(40)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = os.path.join(ROOT_DIR, "Data")
TARGET_VAR = "Outcome"
TRACKING_URI = "http://localhost:5000"


def parseargs() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", type=str, default="diabetes.csv", help="Alpha parameter")
    parser.add_argument("-a", "--alpha", type=float, default=1.0, help="Alpha parameter")
    parser.add_argument("-l", "--l1-ratio", dest="l1", type=float, default=0.5, help="L-1 regularization")
    parser.add_argument("--max-iter", dest='max_iter', type=int, default=1000, help="Max iteration")
    parser.add_argument("-i", "--intercept", type=bool, default=True, help="Fit Intercept")
    parser.add_argument("--experience-name", dest="experience_name", type=str, default="mlops-exp",
                        help="Experiment name")
    parser.add_argument("--model-name", dest="model_name", type=str, default="sklearn-lr-model",
                        help="Model name")
    arguments = parser.parse_args()
    return arguments


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def main(args):
    data_path = os.path.join(DATA_DIR, args.data)
    alpha = args.alpha
    l1_ratio = args.l1
    max_iter = args.max_iter
    intercept = args.intercept
    model_name = args.model_name
    experience_name = args.experience_name

    data = pd.read_csv(data_path)
    logger.info(f"Input data has: {len(data)} rows and {len(data.columns)} columns")
    train, test = train_test_split(data)

    train_x = train.drop([TARGET_VAR], axis=1)
    test_x = test.drop([TARGET_VAR], axis=1)
    train_y = train[[TARGET_VAR]]
    test_y = test[[TARGET_VAR]]

    mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment(experiment_name=f"{experience_name}")

    with mlflow.start_run():
        run = mlflow.active_run()
        logger.info(f"run_id: {run.info.run_id}; status: {run.info.status}")
        # mlflow.sklearn.autolog()
        # mlflow.autolog(log_models=False, log_model_signatures=False, log_input_examples=False, exclusive=True)

        lr = ElasticNet(alpha=alpha,
                        l1_ratio=l1_ratio,
                        fit_intercept=intercept,
                        max_iter=max_iter,
                        random_state=42
                        )

        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)
        rmse, mae, r2 = eval_metrics(test_y, predicted_qualities)

        logger.info(f"Elasticnet model (alpha={alpha}, l1_ratio={l1_ratio})")

        logger.info(f"RMSE = {rmse} | MAE = {mae} | R2 = {r2}")

        logger.info("Log parameters")
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("iter_max", max_iter)

        logger.info(f"Log metrics")
        mlflow.log_metrics({
            "rmse": rmse,
            "r2": r2,
            "mae": mae
        })

        logger.info("Log model")
        mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="sklearn-model",
            registered_model_name=model_name,
        )

        # model registry uri and tracking uri
        logger.info(f"Current model registry uri: {mlflow.get_registry_uri()}")
        logger.info(f"Current tracking uri: {mlflow.get_tracking_uri()}")

        # print(mlflow.get_run(run_id=run.info.run_id))
        mlflow.end_run()
        run = mlflow.get_run(run.info.run_id)
        logger.info(f"run_id: {run.info.run_id}; status: {run.info.status}")
        logger.info(f"Active run: {mlflow.active_run()}")

    # pprint(run.info)
    # pprint(run.data)
    # print(run.info.status)
    # print(type(run.info))


if __name__ == "__main__":
    params = parseargs()
    main(args=params)
