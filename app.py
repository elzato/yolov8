import streamlit as st
import cv2
import numpy as np
import onnxruntime
from PIL import Image
import tempfile
import pyttsx3  # atau from gtts import gTTS

# Load model
session = onnxruntime.InferenceSession("yolov8.onnx")
input_name = session.get_inputs()[0].name

# Load TTS engine
engine = pyttsx3.init()

# Fungsi deteksi
def detect_objects(image):
    img = cv2.resize(image, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[np.newaxis, :, :, :]
    outputs = session.run(None, {input_name: img})[0]
    return outputs  # Sesuaikan dengan output model

# Fungsi TTS
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Streamlit UI
st.title("YOLOv8 Object Detection + TTS")

option = st.radio("Pilih input:", ["Upload Gambar", "Ambil Foto"])

if option == "Upload Gambar":
    uploaded = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = np.array(Image.open(uploaded).convert("RGB"))
        st.image(image, caption="Gambar Anda")
        if st.button("Deteksi"):
            results = detect_objects(image)
            st.write("Hasil deteksi:", results)  # Sesuaikan parsing
            speak("Detected object")

elif option == "Ambil Foto":
    img_data = st.camera_input("Ambil gambar")
    if img_data:
        image = np.array(Image.open(img_data).convert("RGB"))
        st.image(image, caption="Gambar Anda")
        if st.button("Deteksi"):
            results = detect_objects(image)
            st.write("Hasil deteksi:", results)
            speak("Detected object")
