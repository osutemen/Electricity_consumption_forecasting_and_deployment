from typing import Optional
from datetime import datetime
from sqlmodel import SQLModel, Field


class Electric(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    Date: str
    prediction: str


class CreateUpdateElectric(SQLModel):
    Date: str
    class Config:
        schema_extra = {
            "example": {
                "Date": '23.07.2024 10:00'
            }
        }


class ElectricTrain(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    Datetime: str
    Tuketim: float


class ElectricDriftInput(SQLModel):
    last_n_values: int

    class Config:
        schema_extra = {
            "example": {
                "last_n_values": 5,
            }
        }