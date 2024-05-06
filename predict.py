import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from config.logger_config import logger

logger.info("*****Loading the model*****")
model= tf.keras.models.load_model('model/mri_model.h5')
logger.info("*****Model Loaded Successfully*****")

def prediction(img):
    img = cv2.resize(img, (256,256),interpolation= cv2.INTER_CUBIC)
    img = np.array(img)
    reshaped_img = img.reshape(-1, 256, 256, 3)
    flattening = model.predict(reshaped_img).flatten()
    result= np.argmax(flattening)
    print(result)
    if result== 0:
        return {'Prediction':'Normal'}
    if result== 1:
        return {'Prediction':'Parkinson Disease'}
    if result== 2:
        return {'Prediction':'Alzimers Disease'}
    if result== 3:
        return {'Prediction':'Glioma Tumor'}
    if result== 4:
        return {'Prediction':'Meningioma Tumor'}
    if result== 5:
        return {'Prediction':'Pituitary Tumor'}
