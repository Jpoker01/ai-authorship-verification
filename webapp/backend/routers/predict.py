from core import config, ml
from fastapi import APIRouter, HTTPException, status
from schemas.prediction import PredictionRequest, PredictionResponse

router = APIRouter(prefix="/predict", tags=["prediction"])

UNSUPPORTED_TEXT_DETAIL = (
    f'Each text can have at most {config.MAX_LESS_THAN_SIGNS} "<" and '
    f'{config.MAX_OPENING_BRACKETS} "[" characters and no broken HTML markup.'
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
    except ml.UnsupportedTextError as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=UNSUPPORTED_TEXT_DETAIL,
        ) from error
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e!s}")
