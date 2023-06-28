import pandas as pd
import numpy as np
import os
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from urllib.parse import urlparse
import mlflow.sklearn
from mlflow.tracking import MlflowClient


#Read data
df = pd.read_csv("GercekZamanliTuketim-09062013-09062023.csv")

# Format revision
df['Tuketim Miktari (MWh)'] = df['Tuketim Miktari (MWh)'].str.replace(',','')
df['Tuketim Miktari (MWh)'] = pd.to_numeric(df['Tuketim Miktari (MWh)'])

# Create a datetime column by combining the "Tarih" and "Saat" columns
df['Datetime'] = pd.to_datetime(df['Tarih'] + ' ' + df['Saat'], format='%d.%m.%Y %H:%M')

# Remove unnecessary columns
df = df.drop(['Tarih', 'Saat'], axis=1)


# Set 'Datetime' as the index
df.set_index('Datetime', inplace=True)

# Split the data into training and testing
df_train, df_test = df[df.index < '2022-01-01'], df[df.index >= '2022-01-01']

print('Train:\t', len(df_train))
print('Test:\t', len(df_test))


# Define function of create_features
def create_features(df):
    df['Dayofyear'] = df.index.dayofyear
    df['Hour'] = df.index.hour
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    df['Year'] = df.index.year
    return df


df = create_features(df)

print(df)

#Create train and test with features
df_train = create_features(df_train)
df_test = create_features(df_test)

FEATURES = ['Dayofyear', 'Hour', 'Day', 'Quarter', 'Month', 'Year']
TARGET = 'Tuketim Miktari (MWh)'

X_train = df_train[FEATURES]
y_train = df_train[TARGET]

X_test = df_test[FEATURES]
y_test = df_test[TARGET]


os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5050/'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000/'

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

experiment_name = "Forecast of Electric"
mlflow.set_experiment(experiment_name)

registered_model_name = "ElectricModel"

number_of_trees=1000

with mlflow.start_run(run_name="with-reg-rf-sklearn") as run:
    estimator = xgb.XGBRegressor(n_estimators=number_of_trees)
    estimator.fit(X_train, y_train,
                        eval_set=[(X_train, y_train), (X_test, y_test)],
                        early_stopping_rounds=50,
                        verbose=True)


    y_pred = estimator.predict(X_test)

    (rmse, mae, r2) = eval_metrics(y_test, y_pred)

    print(f"Random Forest model number of trees: {number_of_trees}")
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    mlflow.log_param("n_estimators", number_of_trees)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

    # Model registry does not work with file store
    if tracking_url_type_store != "file":

        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(estimator, "model", registered_model_name=registered_model_name)
    else:
        mlflow.sklearn.log_model(estimator, "model")


# Optional part
#name = registered_model_name
#client = MlflowClient()
#
#model_uri = "runs:/{}/sklearn-model".format(run.info.run_id)
#print(model_uri)
#
#mv = client.create_model_version(name, model_uri, run.info.run_id)
#print("model version {} created".format(mv.version))
#last_mv = mv.version
#print(last_mv)
#
#def print_models_info(models):
#    for m in models:
#        print("name: {}".format(m.name))
#        print("latest version: {}".format(m.version))
#        print("run_id: {}".format(m.run_id))
#        print("current_stage: {}".format(m.current_stage))
#
#def get_latest_model_version(models):
#    for m in models:
#        print("name: {}".format(m.name))
#        print("latest version: {}".format(m.version))
#        print("run_id: {}".format(m.run_id))
#        print("current_stage: {}".format(m.current_stage))
#    return m.version
#
#models = client.get_latest_versions(name, stages=["None"])
#print_models_info(models)
#
#print(f"Latest version: { get_latest_model_version(models) }")