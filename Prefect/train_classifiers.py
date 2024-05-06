import argparse
import os
import pathlib
import time
import warnings
from dataclasses import dataclass
from typing import Tuple, Union, List

import pandas as pd
import numpy as np
import sklearn
from prefect.artifacts import create_table_artifact
from prefect.task_runners import ConcurrentTaskRunner
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from prefect import task, flow, get_run_logger

warnings.filterwarnings("ignore")

ROOT_DIR = pathlib.Path(__file__).parent.parent.resolve()
DATA_DIR = os.path.join(ROOT_DIR, "Data")
TARGET_VAR = "Outcome"
RANDOM_SATE = 101
DECIMALS = 3


@dataclass
class ScoreItem:
    model_name: str
    avg_score: float
    std_scores: float
    algo: Union[sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin]
    training_time: float

    def __str__(self):
        plus_minus = "\u00B1"
        return (f"Model: {self.model_name} => CV score: {self.avg_score} {plus_minus} {self.std_scores}, "
                f"TrainingTime: {training_time} secs")


"""
class ScoreItem(BaseModel):
    model_name: str
    avg_score: float
    std_scores: float

    def __str__(self):
        plus_minus = "\u00B1"
        return f"Model: {self.model_name} => CV score: {self.avg_score} {plus_minus} {self.std_scores}"
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="diabetes.csv", help="Alpha parameter")
    parser.add_argument("--max_depth", type=int, default=6, help="Max depth")
    parser.add_argument("--n_estimators", type=int, default=64,
                        help="Number of estimators (e.g, num trees in tree-based models")
    parser.add_argument("--min_samples_split", type=int, default=2, help="Minimum samples for split")
    parser.add_argument("--eval_metric", type=str, default='f1', help="Evaluation metric")
    parser.add_argument("--num_folds", type=int, default=10, help="Number of folds for cross validation")
    parser.add_argument("--learning_rate", type=float, default=0.1, help='Learning rate')
    parser.add_argument("--max_iter", type=int, default=200, help='Number of iterations')
    parser.add_argument("--model_tag", type=str, default="test_candidate",
                        help="Adding ability to retrieve models by tag")
    parser.add_argument("--regularization", type=float, default=1.0,
                        help="Regularization: equivalent to C  parameter for LR or SVM")

    arguments = parser.parse_args()
    return arguments


def computes_scores(model, train_x: pd.DataFrame, train_y: pd.Series, eval_metric: str, num_fold: int = 10):
    scores = cross_val_score(
        estimator=model,
        X=train_x,
        y=train_y,
        cv=num_fold,
        scoring=eval_metric
    )
    return scores


@task(cache_result_in_memory=True)
def prep_data(data_name: str) -> Tuple[pd.DataFrame, Union[pd.Series, pd.DataFrame]]:
    logger = get_run_logger()
    data_path = os.path.join(DATA_DIR, data_name)
    logger.info(f"Data absolute path: {data_path}")
    data = pd.read_csv(data_path)
    logger.info(f"Input data has: {len(data)} rows and {len(data.columns)} columns")
    train_x = data.drop([TARGET_VAR], axis=1)
    train_y = data[[TARGET_VAR]]
    logger.info(f"Train data contains => {len(train_x)} samples")

    return train_x, train_y


@task
def train_clf(X: pd.DataFrame, y: pd.Series, algo: sklearn.base.ClassifierMixin, eval_metric: str) -> ScoreItem:
    logger = get_run_logger()
    model_name = algo.__class__.__name__

    start_time = time.time()
    cv_scores = computes_scores(model=algo, train_x=X, train_y=y, eval_metric=eval_metric)
    training_time = time.time() - start_time

    logger.info(f"{model_name} cv_scores: {np.round(np.mean(cv_scores), DECIMALS)}")

    score_item = ScoreItem(
        model_name,
        np.round(np.mean(cv_scores), DECIMALS),
        np.round(np.std(cv_scores), DECIMALS),
        algo,
        np.round(training_time, DECIMALS)
    )

    return score_item


@task
def create_score_table(scores: List[ScoreItem]):
    perf_per_model = []

    for score_item in scores:
        d = {
            "Model": score_item.model_name,
            "Average CV Score": score_item.avg_score,
            "Standard Deviation (Errors)": score_item.std_scores,
            "Time (sec)": score_item.training_time
        }
        perf_per_model.append(d)

    create_table_artifact(
        key="models-perf",
        table=perf_per_model,
        description="# Models' Performance Comparison"
    )


@task(
    persist_result=True,
    result_storage_key="{flow_run.flow_name}_{flow_run.name}_best_model.pickle",
    result_serializer="pickle"
)
def select_best_model(scores: List[ScoreItem], eval_metric: str, **kwargs):
    plus_minus = "\u00B1"

    logger = get_run_logger()
    logger.info(f"Model performances ({eval_metric} score):")
    for item_score in scores:
        model_name = item_score.model_name
        avg_scores = item_score.avg_score
        std_scores = item_score.std_scores
        logger.info(f"  {model_name}: {avg_scores} {plus_minus} {std_scores}")

    best_classifier_model = max(scores, key=lambda item: item.avg_score)

    train_x = kwargs.get("train_x")
    train_y = kwargs.get("train_y")

    if all(v is not None for v in [train_x, train_y]):
        logger.info("Fit and save best model")
        best_model = best_classifier_model.algo
        try:
            best_model.fit(X=train_x, y=train_y)
            return best_model
        except Exception as e:
            logger.error(f"Fitting best model failed due to: {e}")

    else:
        logger.info(f"'train_x' or 'train_y' parameter is None.")


@flow(
    task_runner=ConcurrentTaskRunner(),
    validate_parameters=True,
    version="v1",
    description="Train and compare multiple classifier models"
)
def train_flow(params: dict):
    logger = get_run_logger()
    scores = []

    n_estimators = params["n_estimators"]
    learning_rate = params["learning_rate"]
    eval_metric = params["eval_metric"]
    max_depth = params["max_depth"]
    max_iter = params["max_iter"]
    regularization = params["regularization"]
    random_state = params.get("random_state", RANDOM_SATE)
    data_name = params["data"]
    eval_metric = params["eval_metric"]

    lr = LogisticRegression(C=regularization, max_iter=max_iter, random_state=random_state)
    rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    gbt = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=random_state)
    svm = SVC(C=regularization, random_state=random_state)

    train_x, train_y = prep_data(data_name=data_name)

    for algo in [lr, rf, gbt, svm]:
        model_name = algo.__class__.__name__
        logger.info(f"Training {model_name} model ...")
        score_item = train_clf.submit(X=train_x, y=train_y, algo=algo, eval_metric=eval_metric)
        scores.append(score_item)

    logger.info("Create performance table artefact")
    create_score_table(scores=scores)
    select_best_model(scores=scores, eval_metric=eval_metric, train_x=train_x, train_y=train_y)
