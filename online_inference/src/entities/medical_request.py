import pandas as pd
from typing import List, Dict
from pydantic import BaseModel, validator


class MedicalRequest(BaseModel):
    """ Main request DTO class. """

    data: List[Dict[str, float]]
    real_features: List[str] = None

    @validator('data')
    def validate_model_features(cls, data: List[Dict[str, float]]):
        data_pd = pd.DataFrame(data)
        have_cols = list(data_pd.columns.values)
        if have_cols != MedicalRequest.real_features:
            raise ValueError('Either feature count or order isnt correct\n'
                             f'expected: {MedicalRequest.real_features}\n'
                             f'have:     {have_cols}')
        return data_pd
