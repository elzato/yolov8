import onnxruntime
import numpy as np
from PIL import Image
import io

# Muat model YOLOv8
session = onnxruntime.InferenceSession("model/yolov8.onnx")

def preprocess_image(image: Image):
    # Resize gambar dan normalisasi
    image = image.resize((640, 640))  # Ukuran standar untuk YOLOv8
    image = np.array(image).astype(np.float32)
    image = image / 255.0  # Normalisasi gambar
    image = np.transpose(image, (2, 0, 1))  # Channels first
    image = np.expand_dims(image, axis=0)  # Tambah dimensi batch
    return image

def detect_objects(image_data: bytes):
    # Mengubah data byte menjadi gambar
    image = Image.open(io.BytesIO(image_data))
    input_data = preprocess_image(image)

    # Menjalankan deteksi objek
    inputs = {session.get_inputs()[0].name: input_data}
    outputs = session.run(None, inputs)

    return outputs[0].tolist()  # Mengembalikan hasil deteksi objek
