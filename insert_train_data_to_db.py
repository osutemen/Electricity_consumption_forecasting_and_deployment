import pandas as pd
from sqlalchemy.sql import text as sa_text
from database import engine
from models import ElectricTrain
from sqlmodel import Session

# read data
#Read data
df = pd.read_csv("GercekZamanliTuketim-09062013-09062023.csv")



# Format revision
df['Tuketim Miktari (MWh)'] = df['Tuketim Miktari (MWh)'].str.replace(',','')
df['Tuketim Miktari (MWh)'] = pd.to_numeric(df['Tuketim Miktari (MWh)'])

# Create a datetime column by combining the "Tarih" and "Saat" columns
df['Datetime'] = pd.to_datetime(df['Tarih'] + ' ' + df['Saat'], format='%d.%m.%Y %H:%M')

# Remove unnecessary columns
df = df.drop(['Tarih', 'Saat'], axis=1)
df = df.reindex(columns=['Datetime', 'Tuketim Miktari (MWh)'])
df = df.rename(columns={'Tuketim Miktari (MWh)': 'Tuketim'})

#Convert datetime column to string
df['Datetime'] = df['Datetime'].astype(str)

print(df.info())

#Save the dataset as csv
#df.to_csv('tuketim.csv', index=False)

#Set 'Datetime' as the index
#df.set_index('Datetime', inplace=True)

#Truncate table with sqlalchemy
with Session(engine) as session:
    session.execute(sa_text(''' TRUNCATE TABLE electrictrain  '''))
    session.commit()

# Insert training data
records_to_insert = []

for df_idx, line in df.iterrows():
    records_to_insert.append(
                    ElectricTrain(Datetime=line[0],
                                  Tuketim=line[1]


                    )
    )

session.bulk_save_objects(records_to_insert)
session.commit()
# Ends database insertion
