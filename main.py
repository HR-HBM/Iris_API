from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io

app = FastAPI()

# Allow CORS for Node.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
retinopathy_model = load_model("retinopathy_model.h5")
edema_model = load_model("edema_model.h5")

# Preprocess image (adjust as per your models’ input requirements)
def preprocess_image(image: Image.Image):
    image = image.resize((224, 224))  # Example size
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and process image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image_array = preprocess_image(image)

    # Get predictions
    retinopathy_pred = retinopathy_model.predict(image_array)
    edema_pred = edema_model.predict(image_array)

    # Convert predictions to labels (adjust based on your model output)
    retinopathy_labels = ["ICDR level 0", "ICDR level 1", "ICDR level 2", "ICDR level 3", "ICDR level 4"]
    edema_labels = ["DME Positive", "DME Negative"]
    retinopathy_result = retinopathy_labels[np.argmax(retinopathy_pred)]
    edema_result = edema_labels[np.argmax(edema_pred)]

    return {
        "retinopathy": retinopathy_result,
        "edema": edema_result,
        "retinopathy_probs": retinopathy_pred.tolist()[0],
        "edema_probs": edema_pred.tolist()[0]
    }