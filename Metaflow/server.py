import asyncio
import logging
from fastapi import FastAPI
from metaflow import Flow
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_last_run(flow_name, tag):
    for r in Flow(flow_name).runs(tag):
        if r.successful:
            return r


def load_model(flow_name: str = 'ClassifiersFlow', tag_name: str = 'test_candidate', **kwargs):
    data_artefacts = (get_last_run(flow_name=flow_name, tag=tag_name))
    model = data_artefacts.data.best_classifier_model

    logger.debug(f"Testing model on sample data ...")
    cols_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI',
                  'DiabetesPedigreeFunction', 'Age']
    values = [10, 101, 76, 48, 180, 32.9, 0.171, 63]
    test_df = pd.DataFrame(dict(zip(cols_names, values)), index=[0])

    r = model.predict_proba(test_df)
    logger.debug(f"Prediction on test data: {r}")

    return model


best_classifier_model = load_model()
api = FastAPI()


@api.get("/")
def root():
    return {"message": "Welcome to prediction service"}


@api.post("/predict")
def diagnose(data: dict):
    test_df = pd.DataFrame(data, index=[0])
    model_classes = [str(x) for x in best_classifier_model.classes_]
    prediction = best_classifier_model.predict_proba(test_df)
    res = dict(zip(model_classes, prediction[0]))
    return res


@api.get("/shutdown")
def shutdown():
    loop = asyncio.get_event_loop()
    loop.stop()
    return {"message": "Server shutting down..."}
