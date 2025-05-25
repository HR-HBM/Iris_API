from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
try:
    retinopathy_model = load_model("rsg_net_V3drModel.h5")
    edema_model = load_model("MobileNetV2_dme_model.h5")
except Exception as e:
    print(f"Error loading models: {e}")
    raise
# Preprocess image
def preprocess_dr_image(image: Image.Image):
    image = image.resize((512, 512))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def preprocess_me_image(image: Image.Image):
    image = image.resize((300, 300))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.get("/")
async def root():
    return {"message": "FastAPI app is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and process image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    dr_image_array = preprocess_dr_image(image)
    me_image_array = preprocess_me_image(image)


    # Get predictions
    retinopathy_pred = retinopathy_model.predict(dr_image_array)
    edema_pred = edema_model.predict(me_image_array)
    edema_result = "Positive" if edema_pred[0][0] > 0.5 else "Negative"


    # Convert predictions to labels
    retinopathy_labels = ["ICDR level 0", "ICDR level 1", "ICDR level 2", "ICDR level 3", "ICDR level 4"]
    retinopathy_result = retinopathy_labels[np.argmax(retinopathy_pred)]

    return {
        "retinopathy": retinopathy_result,
        "edema": edema_result,
        "retinopathy_probs": retinopathy_pred.tolist()[0],
        "edema_probs": edema_pred.tolist()[0]
    }
