import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import timedelta
import joblib

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


print(X_train)

reg = xgb.XGBRegressor(n_estimators=1000)
estimator = reg.fit(X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        early_stopping_rounds=50,
       verbose=True)


# Save model with joblib
joblib.dump(estimator, "saved_models/xgb.pkl")


# -------------------------------------------------------------------
# Funciton of predict days and hours and manual control
#def tahmin_et_gün(start_date):
#    start_date = pd.to_datetime(start_date, format='%d.%m.%Y %H:%M')
#    end_date = start_date + pd.DateOffset(days=5)
#    future = pd.date_range(start=start_date, end=end_date, freq='D')
#    future_df = pd.DataFrame(index=future)
#    future_df_final = create_features(future_df)
#    a = future_df_final[FEATURES]
#    tahmin = estimator.predict(a)
#    return tahmin
#
#
#gün_tahmin = tahmin_et_gün('23.07.2024 10:00')
#print(gün_tahmin)
#def tahmin_et_saat(tarih):
#    start_date = pd.to_datetime(tarih, format='%d.%m.%Y %H:%M')
#    end_date = start_date + timedelta(hours=24)
#    future = pd.date_range(start=start_date, end=end_date, freq='H')
#    future_df = pd.DataFrame(index=future)
#    future_df_final = create_features(future_df)
#    a = future_df_final[FEATURES]
#    tahmin = estimator.predict(a)
#    return tahmin
#
#saat_tahmin = tahmin_et_saat('23.07.2024 10:00')
#print(saat_tahmin)
#