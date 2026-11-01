from pydantic import BaseModel, Field
from core import config

class PredictionRequest(BaseModel):
    """Request model for authorship verification."""
    text1: str = Field(..., description="First text to compare", min_length=config.MIN_TEXT_LENGTH, max_length=config.MAX_TEXT_LENGTH)
    text2: str = Field(..., description="Second text to compare", min_length=config.MIN_TEXT_LENGTH, max_length=config.MAX_TEXT_LENGTH)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text1": "This is a sample text written by an author.",
                    "text2": "Here is another text that might be by the same person."
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response model for authorship verification."""
    same_author_probability: float = Field(
        ...,
        description="Probability that both texts are written by the same author",
        ge=0.0,
        le=1.0
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "same_author_probability": 0.75
                }
            ]
        }
    }