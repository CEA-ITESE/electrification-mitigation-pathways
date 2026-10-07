import pandera.pandas as pa

class SourceDatasetSchema(pa.DataFrameModel):

    Case: pa.typing.Series[int] = pa.Field(coerce=True)
    Period: pa.typing.Series[int] = pa.Field(coerce=True)
    Region: pa.typing.Series[str] = pa.Field(coerce=True)
    
    class Config:
        strict = False
        coerce = True


class ReferenceDatasetSchema(pa.DataFrameModel):
    """Pandera schema model for validating core structural pillars of reference data."""

    Model: pa.typing.Series[str] = pa.Field(coerce=True)
    Scenario: pa.typing.Series[str] = pa.Field(coerce=True)
    Region: pa.typing.Series[str] = pa.Field(coerce=True)
    Year: pa.typing.Series[int] = pa.Field(coerce=True)

    class Config:
        strict = False
        coerce = True