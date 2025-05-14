from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from app.yolo_model import detect_objects  # Mengimpor fungsi deteksi objek
import io

app = FastAPI()

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Membaca file gambar yang di-upload
    image_data = await file.read()

    # Menjalankan deteksi objek
    detections = detect_objects(image_data)

    # Mengembalikan hasil deteksi sebagai JSON
    return JSONResponse(content={"detections": detections})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
