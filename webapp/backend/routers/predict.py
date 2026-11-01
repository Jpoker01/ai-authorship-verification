from fastapi import APIRouter, HTTPException
from schemas.prediction import PredictionRequest, PredictionResponse
from core import ml

router = APIRouter(
    prefix="/predict",
    tags=["prediction"]
)


@router.post("/", response_model=PredictionResponse)
async def predict_authorship(request: PredictionRequest):
    """
    Predict whether two texts are written by the same author.

    Args:
        request: PredictionRequest containing text1 and text2

    Returns:
        PredictionResponse with probabilities and prediction

    Raises:
        HTTPException: If prediction fails
    """
    try:
        result = ml.predict(request.text1, request.text2)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )