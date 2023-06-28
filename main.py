
from fastapi import FastAPI, Depends, Request
import joblib
from models import Electric, CreateUpdateElectric,ElectricDriftInput
from database import engine, get_db, create_db_and_tables
from sqlalchemy.orm import Session
import pandas as pd
from datetime import timedelta
from scipy.stats import ks_2samp


# Read models saved during train phase
estimator_loaded = joblib.load( "saved_models/xgb.pkl")

app = FastAPI()

# Creates all the tables defined in models module
create_db_and_tables()





def create_features(df):
    df['Dayofyear'] = df.index.dayofyear
    df['Hour'] = df.index.hour
    df['Day'] = df.index.day
    df['Month'] = df.index.month
    df['Quarter'] = df.index.quarter
    df['Year'] = df.index.year
    return df

def make_days_prediction(model, request):
    # parse input from request
    Date = request["Date"]

    # Make an input vector
    FEATURES = ['Dayofyear', 'Hour', 'Day', 'Quarter', 'Month', 'Year']

    start_date = pd.to_datetime(Date, format='%d.%m.%Y %H:%M')
    end_date = start_date + pd.DateOffset(days=4)
    future = pd.date_range(start=start_date, end=end_date, freq='D')
    future_df = pd.DataFrame(index=future)
    future_df_final = create_features(future_df)
    a = future_df_final[FEATURES]
#    prediction = model.predict(a)
    prediction_array = model.predict(a)
    prediction = prediction_array.tolist()
    print(prediction)
    print(type(prediction))
    print("--------------------------")
    return prediction

def make_hours_prediction(model, request):
    # parse input from request
    Date = request["Date"]

    # Make an input vector
    FEATURES = ['Dayofyear', 'Hour', 'Day', 'Quarter', 'Month', 'Year']

    start_date = pd.to_datetime(Date, format='%d.%m.%Y %H:%M')
    end_date = start_date + timedelta(hours=24)
    future = pd.date_range(start=start_date, end=end_date, freq='H')
    future_df = pd.DataFrame(index=future)
    future_df_final = create_features(future_df)
    a = future_df_final[FEATURES]
    prediction_array = model.predict(a)
    prediction = prediction_array.tolist()
    print(prediction)
    print(type(prediction))
    print("--------------------------")
    return prediction
def insert_energy(request, prediction, client_ip, db):
    new_energy = Electric(
        Date=request["Date"],
        prediction=prediction,
        client_ip=client_ip
    )

    with db as session:
        session.add(new_energy)
        session.commit()
        session.refresh(new_energy)

    return new_energy

# Object agnostic drift detection function
def detect_drift(data1, data2):
    ks_result = ks_2samp(data1, data2)
    if ks_result.pvalue < 0.05:
        return "Drift exits"
    else:
        return "No drift"

# Enegy Prediction endpoint
@app.post("/prediction/energy_days")
async def predict_energy(request: CreateUpdateElectric, fastapi_req: Request,  db: Session = Depends(get_db)):
    predictions = make_days_prediction(estimator_loaded, request.dict())
    db_insert_records = []
    for prediction in predictions:
        db_insert_record = insert_energy(
            request=request.dict(),
            prediction=prediction,
            client_ip=fastapi_req.client.host,
            db=db
        )
        db_insert_records.append(db_insert_record)
        print(predictions)
    return {"prediction": predictions, "db_record": db_insert_records}


@app.post("/prediction/energy_hours")
async def predict_energy(request: CreateUpdateElectric, fastapi_req: Request,  db: Session = Depends(get_db)):
    predictions = make_hours_prediction(estimator_loaded, request.dict())
    db_insert_records = []
    for prediction in predictions:
        db_insert_record = insert_energy(
            request=request.dict(),
            prediction=prediction,
            client_ip=fastapi_req.client.host,
            db=db,
        )
        db_insert_records.append(db_insert_record)
    return {"prediction": predictions, "db_record": db_insert_records}


# Energy drift detection endpoint
@app.post("/drift/energy")
async def detect(request: ElectricDriftInput):
    # Select training data
    train_df = pd.read_sql("select * from electrictrain", engine)

    # Select predicted data last n days
    prediction_df = pd.read_sql(f"""SELECT * FROM electric
                                    ORDER BY id DESC
                                    LIMIT {request.last_n_values}""",
                                engine)

    electric_drift = detect_drift(train_df.Datetime, prediction_df.Date)

    return {"electric_drift": electric_drift}


@app.get("/")
async def root():
    return {"data":"Wellcome to MLOps API"}
