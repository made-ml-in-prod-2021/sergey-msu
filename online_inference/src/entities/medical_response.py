from pydantic import BaseModel


class MedicalResponse(BaseModel):
    """ Main response DTO class. """

    result: int
