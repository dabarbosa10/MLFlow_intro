import warnings
import argparse
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# get arguments From command

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.7)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.7)
args = parser.parse_args()


# evaluation function

def eval_metrics(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # read the CSV file
    csv_url = (
        "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/master/winequality-red.csv"
    )

    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception("Unable to download dataset. Error : %s", e)

    train, test = train_test_split(data, test_size=0.2, random_state=40)

    train_y = train[["quality"]]
    test_y = test[["quality"]]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)

    alpha = args.alpha
    l1_ratio = args.l1_ratio
    exp = mlflow.set_experiment(experiment_name="experiment_1")  # Experiment name

    # instantation of our model class

    with mlflow.start_run(experiment_id=exp.experiment_id):
        l1 = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        l1.fit(train_x, train_y)

        predicted_qualities = l1.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print(f"Elasticnet model (alpha={alpha}, l1_ratio={l1_ratio})")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R2: {r2}")

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(l1,"mymodel")
