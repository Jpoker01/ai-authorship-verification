from core import config
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Request model for authorship verification."""

    class PredictionRequest(BaseModel):
        """Request model for authorship verification."""

        text1: str = Field(
            description="First text to compare",
            min_length=config.MIN_TEXT_LENGTH,
            max_length=config.MAX_TEXT_LENGTH,
        )
        text2: str = Field(
            description="Second text to compare",
            min_length=config.MIN_TEXT_LENGTH,
            max_length=config.MAX_TEXT_LENGTH,
        )

        model_config = {
            "json_schema_extra": {
                "examples": [
                    {
                        "text1": "This is a sample text written by an author.",
                        "text2": "Here is another text that might be by the same person.",
                    },
                ],
            },
        }


class ModelDescription(BaseModel):
    """Name and test results of the model that made the prediction."""

    name: str = Field(description="Text representation, combination and classifier")
    test_f1: float = Field(description="F1 score on the test set", ge=0.0, le=1.0)
    test_accuracy: float = Field(description="Accuracy on the test set", ge=0.0, le=1.0)
    test_pairs: int = Field(description="Number of text pairs in the test set")


class PredictionResponse(BaseModel):
    """Response model for authorship verification."""

    same_author: bool = Field(
        description="Decision of the classifier: true if both texts have the same author",
    )
    score: float = Field(
        description="Model score - the classifier's probability of the same-author class",
        ge=0.0,
        le=1.0,
    )
    model: ModelDescription = Field(description="Model that made the prediction")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "same_author": True,
                    "score": 0.75,
                    "model": {
                        "name": "TF-IDF of words, absolute difference, logistic regression",
                        "test_f1": 0.9044,
                        "test_accuracy": 0.9068,
                        "test_pairs": 19999,
                    },
                },
            ],
        },
    }
