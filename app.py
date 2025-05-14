import streamlit as st
import cv2
import numpy as np
import onnxruntime
import pyttsx3
from PIL import Image

# Setup TTS (Text to Speech)
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load YOLOv8 Model (ONNX)
def load_model():
    session = onnxruntime.InferenceSession("https://github.com/elzato/yolov8/raw/main/yolov8.onnx")  # Load from GitHub URL
    return session

# Fungsi untuk deteksi objek pada gambar
def detect_objects(image, model):
    input_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert PIL to OpenCV
    input_image = cv2.resize(input_image, (640, 640))
    input_image = input_image.transpose(2, 0, 1)  # HWC to CHW
    input_image = np.expand_dims(input_image, axis=0).astype(np.float32)  # Add batch dimension

    # Inference
    inputs = {model.get_inputs()[0].name: input_image}
    outputs = model.run(None, inputs)

    return outputs

# Fungsi untuk proses deteksi dan pelafalan
def process_detection(image, model):
    outputs = detect_objects(image, model)
    # Placeholder untuk proses deteksi
    # Misalnya mengasumsikan hasil deteksi adalah kategori objek dan koordinat
    objects = outputs[0]  # Asumsi output pertama adalah daftar objek
    detected_objects = []

    # Contoh deteksi (ganti dengan deteksi sesungguhnya)
    for obj in objects:
        detected_objects.append(obj)
    
    return detected_objects

# Streamlit App
st.title("Deteksi Objek Real-Time dengan YOLOv8")

# Load Model
model = load_model()

# Pilihan input: foto atau upload
input_option = st.radio("Pilih input gambar:", ["Foto Objek", "Upload Gambar"])

if input_option == "Foto Objek":
    st.write("Klik tombol untuk memfoto objek.")
    picture = st.camera_input("Ambil Foto")
    if picture:
        img = Image.open(picture)
        detected_objects = process_detection(img, model)
        st.image(img, caption="Gambar yang Diambil", use_column_width=True)
        st.write("Objek yang terdeteksi:", detected_objects)
        speak(f"Deteksi objek: {', '.join(detected_objects)}")

elif input_option == "Upload Gambar":
    uploaded_image = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        img = Image.open(uploaded_image)
        detected_objects = process_detection(img, model)
        st.image(img, caption="Gambar yang Diupload", use_column_width=True)
        st.write("Objek yang terdeteksi:", detected_objects)
        speak(f"Deteksi objek: {', '.join(detected_objects)}")
