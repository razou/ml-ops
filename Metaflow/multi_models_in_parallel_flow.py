import logging
import os
import pathlib
import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score

from metaflow import FlowSpec, step, Parameter

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = os.path.join(ROOT_DIR, "Data")
TARGET_VAR = "Outcome"


class ClassifiersFlow(FlowSpec):
    data = Parameter(name="data", type=str, default="diabetes.csv", help="Alpha parameter")
    max_depth = Parameter(name="max_depth", type=int, default=6, help="Max depth")
    random_state = Parameter(name="seed", type=int, default=21, help="Random seed, for reproducible results")
    n_estimators = Parameter(name="n_estimators", type=int, default=64,
                             help="Number of estimators (e.g, num trees in tree-based models")
    min_samples_split = Parameter(name="min_samples_split", type=int, default=2, help="Minimum samples for split")
    eval_metric = Parameter(name="eval_metric", type=str, default='f1', help="Evaluation metric")
    num_folds = Parameter(name="num_folds", type=int, default=10, help="Number of folds for cross validation")
    learning_rate = Parameter(name='learning_rate', type=float, default=0.1, help='Learning rate')
    max_iter = Parameter(name='max_iter', type=int, default=200, help='Number of iterations')

    def computes_scores(self, model):
        scores = cross_val_score(estimator=model,
                                 X=self.train_x,
                                 y=self.train_y,
                                 cv=self.num_folds,
                                 scoring=self.eval_metric
                                 )
        return scores

    @step
    def start(self):
        self.data_path = os.path.join(DATA_DIR, str(self.data))
        logger.info(f"Data absolute path: {self.data_path}")
        self.next(self.prep_data)

    @step
    def prep_data(self):
        data = pd.read_csv(self.data_path)
        logger.info(f"Input data has: {len(data)} rows and {len(data.columns)} columns")
        self.train_x = data.drop([TARGET_VAR], axis=1)
        self.train_y = data[[TARGET_VAR]]
        logger.info(f"Train data: {len(self.train_x)} samples")
        self.next(self.train_lr, self.train_rf, self.train_gbt)

    @step
    def train_lr(self):
        logger.info("Training Logistic Regression model ...")
        self.model_name = "Logistic Regression"
        self.lr = LogisticRegression(max_iter=self.max_iter, random_state=self.random_state)
        self.scores = self.computes_scores(model=self.lr)
        logger.info(f"LR scores: {np.round(np.mean(self.scores), 5)}")
        self.next(self.eval_perf)

    @step
    def train_rf(self):
        logger.info("Training Random Forest model ...")
        self.model_name = "Random Forest"
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state)
        self.scores = self.computes_scores(model=self.rf)
        logger.info(f"RF scores: {np.round(np.mean(self.scores), 5)}")
        self.next(self.eval_perf)

    @step
    def train_gbt(self):
        logger.info("Training Gradient Boosted Tree model ...")
        self.model_name = "Gradient Boosted Tree"

        self.gbt = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state)

        self.scores = self.computes_scores(model=self.gbt)
        logger.info(f"GBT scores: {np.round(np.mean(self.scores), 5)}")
        self.next(self.eval_perf)

    @step
    def eval_perf(self, modeling_tasks):
        self.scores = [
            (model.model_name,
             np.mean(model.scores),
             np.std(model.scores))
            for model in modeling_tasks
        ]
        self.next(self.end)

    @step
    def end(self):
        self.results = []
        plus_minus = "\u00B1"

        logger.info(f"Model performances ({self.eval_metric} score):")
        for name, avg_perf_score, std_perf_score in self.scores:
            self.results.append((name, avg_perf_score, std_perf_score))
            logger.info(f"  {name}: {np.round(avg_perf_score, 3)} {plus_minus} {np.round(std_perf_score, 3)}")


if __name__ == "__main__":
    ClassifiersFlow()
