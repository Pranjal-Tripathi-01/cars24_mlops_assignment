import numpy as np
import cv2
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile,Response, status, HTTPException
from predict import prediction
from config.logger_config import logger

mlops_assignment = FastAPI()

origins = ["*"]
mlops_assignment.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],)

@mlops_assignment.get("/")
def read_root():
    return {"Pranjal Tripathi": "Cars24 MLOps Engineer Assignment"}

@mlops_assignment.post("/ai",status_code=status.HTTP_200_OK)
async def predict(response: Response, file: UploadFile = File(...)):
    contents = await file.read()
    logger.info("File Type: ", file.content_type)
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=f'Image file not found. Please provide a valid input')

    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result = prediction(img)

    return {'prediction':result}
