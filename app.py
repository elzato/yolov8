from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import onnxruntime
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load the YOLOv8 model (pastikan jalur model benar)
session = onnxruntime.InferenceSession("yolov8.onnx")

def preprocess_image(image: Image):
    # Resize the image to 640x640 as required by YOLOv8
    image = image.resize((640, 640))
    image = np.array(image).astype(np.float32)
    image = image / 255.0  # Normalize the image
    image = np.transpose(image, (2, 0, 1))  # Channels first
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def detect_objects(image: Image):
    input_data = preprocess_image(image)
    inputs = {session.get_inputs()[0].name: input_data}
    outputs = session.run(None, inputs)
    return outputs

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Read the uploaded image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Run the object detection
    detections = detect_objects(image)

    # Return the results as JSON
    return JSONResponse(content={"detections": detections[0].tolist()})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
