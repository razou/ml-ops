import os
import pathlib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet, LinearRegression
from metaflow import FlowSpec, Parameter, step

import logging
import warnings

warnings.filterwarnings("ignore")
np.random.seed(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = os.path.join(ROOT_DIR, "Data")
TARGET_VAR = "Outcome"


class MLUtils:

    @staticmethod
    def eval_metrics(actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2


class MainFlow(FlowSpec):
    data = Parameter(name="data", type=str, default="diabetes.csv", help="Alpha parameter")
    alpha = Parameter(name='alpha', type=float, default=0.1, help='Learning rate')
    l1_ratio = Parameter(name="l1-reg", type=float, default=0.5, help="L1 regularization")
    max_iter = Parameter(name="max-iter", type=int, default=1000, help="Max iteration")
    intercept = Parameter(name="intercept", type=bool, default=True, help="Fit Intercept")

    ml_utils = MLUtils()

    @step
    def start(self):
        print("data relative path: ", self.data)
        self.data_path = os.path.join(DATA_DIR, str(self.data))
        print("data absolute path: ", self.data_path)
        self.next(self.prep_data)

    @step
    def prep_data(self):
        data = pd.read_csv(self.data_path)
        logger.info(f"Input data has: {len(data)} rows and {len(data.columns)} columns")
        train, test = train_test_split(data)
        self.train_x = train.drop([TARGET_VAR], axis=1)
        self.test_x = test.drop([TARGET_VAR], axis=1)
        self.train_y = train[[TARGET_VAR]]
        self.test_y = test[[TARGET_VAR]]
        logger.info(f"Train data: {len(self.train_x)} samples")
        logger.info(f"Test data: {len(self.test_x)} samples")
        self.next(self.train_step)

    @step
    def train_step(self):
        self.lr_model = ElasticNet(alpha=self.alpha,
                                   l1_ratio=self.l1_ratio,
                                   fit_intercept=self.intercept,
                                   max_iter=self.max_iter,
                                   random_state=42)

        self.lr_model.fit(self.train_x, self.train_y)
        self.next(self.eval_step)

    @step
    def eval_step(self):
        predicted_qualities = self.lr_model.predict(self.test_x)
        self.rmse, self.mae, self.r2 = self.ml_utils.eval_metrics(self.test_y, predicted_qualities)
        self.next(self.end)

    @step
    def end(self):
        logger.info(f"Elasticnet model (alpha={self.alpha}, l1_ratio={self.l1_ratio})")
        logger.info(f"RMSE = {self.rmse} | MAE = {self.mae} | R2 = {self.r2}")


if __name__ == "__main__":
    MainFlow()
