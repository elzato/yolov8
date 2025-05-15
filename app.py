from flask import Flask, render_template_string, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import io
from PIL import Image

app = Flask(__name__)

# Load YOLOv8 model (you can choose yolov8n.pt for faster inference)
model = YOLO("yolov8n.pt")

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>YOLOv8 Object Detection for Kids</title>
<style>
  body {
    font-family: 'Comic Sans MS', cursive, sans-serif;
    background: #ffefd5; /* Ceria dan ramah anak */
    margin: 0;
    padding: 20px;
    text-align: center;
  }
  h1 {
    color: #ff4500;
    font-size: 2em;
  }
  #videoWrapper, #uploadedWrapper {
    position: relative;
    display: inline-block;
  }
  video, canvas, img {
    border-radius: 15px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    max-width: 100%;
  }
  #canvas {
    position: absolute;
    top: 0;
    left: 0;
  }
  #captureBtn, #uploadInput, #clearUploadBtn {
    margin-top: 15px;
    font-size: 18px;
    padding: 12px 16px;
    border-radius: 8px;
    border: none;
    background-color: #32cd32;
    color: white;
    cursor: pointer;
  }
  #captureBtn:hover, #clearUploadBtn:hover {
    background-color: #3cb371;
  }
  #audioControls {
    margin-top: 20px;
    display: none;
    text-align: center;
  }
  .speak-btn {
    cursor: pointer;
    background: #32cd32;
    border: none;
    color: white;
    padding: 6px 10px;
    border-radius: 8px;
    font-size: 14px;
    margin: 5px;
  }
  .speak-btn:hover {
    background: #3cb371;
  }
  input[type="file"] {
    display: none;
  }
  label[for="uploadInput"] {
    background-color: #32cd32;
    color: white;
    padding: 12px 16px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 18px;
    display: inline-block;
    margin-top: 15px;
  }
</style>
</head>
<body>
<h1>🎨 Object Detection & Fun Learning! 🐾</h1>
<p>Mari belajar bahasa Inggris dengan deteksi objek . Ambil sebuah foto atau upload !</p>

<div id="videoWrapper">
  <video id="video" width="640" height="480" autoplay muted></video>
  <canvas id="canvas" width="640" height="480"></canvas>
</div>
<br />
<button id="captureBtn">📸 Capture Photo</button>
<br />

<label for="uploadInput">📂 Upload Photo</label>
<input type="file" id="uploadInput" accept="image/*" />
<button id="clearUploadBtn" style="display:none;">❌ Clear Upload</button>

<div id="uploadedWrapper" style="margin-top: 20px; display: none;">
  <h2>Uploaded Photo</h2>
  <img id="uploadedImage" />
  <canvas id="uploadedCanvas"></canvas>
</div>

<div id="audioControls">
  <h3>🔊 Pronunciation Controls</h3>
  <div id="pronounceButtons"></div>
</div>

<script>
// Setup webcam stream
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const captureBtn = document.getElementById('captureBtn');
const uploadInput = document.getElementById('uploadInput');
const clearUploadBtn = document.getElementById('clearUploadBtn');
const uploadedWrapper = document.getElementById('uploadedWrapper');
const uploadedImage = document.getElementById('uploadedImage');
const uploadedCanvas = document.getElementById('uploadedCanvas');
const uploadedCtx = uploadedCanvas.getContext('2d');
const audioControls = document.getElementById('audioControls');
const pronounceButtonsDiv = document.getElementById('pronounceButtons');

function speak(text) {
  if ('speechSynthesis' in window) {
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    window.speechSynthesis.speak(utterance);
  } else {
    alert('Speech Synthesis not supported in this browser.');
  }
}

function drawBoxes(image, detections, context, canvasElement) {
  context.clearRect(0, 0, canvasElement.width, canvasElement.height);
  context.drawImage(image, 0, 0, canvasElement.width, canvasElement.height);

  context.lineWidth = 3;
  context.font = '20px Comic Sans MS';
  context.textBaseline = 'top';

  detections.forEach(det => {
    const [x1, y1, x2, y2] = det.bbox;
    context.strokeStyle = '#ff4500';
    context.strokeRect(x1, y1, x2 - x1, y2 - y1);
    context.fillStyle = '#ff4500';
    context.fillText(det.label, x1 + 5, y1 + 5);
  });
}

function processDetections(data, image, context, canvasElement) {
  if (!data || !data.detections) return;

  drawBoxes(image, data.detections, context, canvasElement);

  pronounceButtonsDiv.innerHTML = '';
  audioControls.style.display = 'block';

  const uniqueLabels = new Set();
  data.detections.forEach(det => {
    if (!uniqueLabels.has(det.label)) {
      uniqueLabels.add(det.label);

      const pronounceBtn = document.createElement('button');
      pronounceBtn.className = 'speak-btn';
      pronounceBtn.textContent = `🔊 ${det.label}`;
      pronounceBtn.onclick = () => speak(det.label);
      pronounceButtonsDiv.appendChild(pronounceBtn);
    }
  });
}

// Send image to backend for detection
async function sendImage(imageBlob) {
  const formData = new FormData();
  formData.append('image', imageBlob, 'capture.jpg');

  const response = await fetch('/detect', {
    method: 'POST',
    body: formData
  });
  if (response.ok) {
    const data = await response.json();
    return data;
  } else {
    alert('Detection failed');
    return null;
  }
}

// Capture photo from webcam
captureBtn.addEventListener('click', async () => {
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  canvas.toBlob(async function(blob) {
    const data = await sendImage(blob);
    if(data){
      processDetections(data, canvas, ctx, canvas);
    }
  }, 'image/jpeg');
});

// Handle photo upload
uploadInput.addEventListener('change', async (event) => {
  if (!event.target.files.length) return;

  const file = event.target.files[0];
  const img = new Image();

  img.onload = async () => {
    uploadedWrapper.style.display = 'block';
    clearUploadBtn.style.display = 'inline-block';

    let scale = Math.min(640 / img.width, 480 / img.height, 1);
    uploadedCanvas.width = img.width * scale;
    uploadedCanvas.height = img.height * scale;

    uploadedCtx.clearRect(0, 0, uploadedCanvas.width, uploadedCanvas.height);
    uploadedCtx.drawImage(img, 0, 0, uploadedCanvas.width, uploadedCanvas.height);

    const data = await sendImage(file);
    if(data){
      processDetections(data, img, uploadedCtx, uploadedCanvas);
    }
  };
  img.src = URL.createObjectURL(file);
});

// Clear uploaded photo
clearUploadBtn.addEventListener('click', () => {
  uploadedWrapper.style.display = 'none';
  clearUploadBtn.style.display = 'none';
  uploadedCtx.clearRect(0, 0, uploadedCanvas.width, uploadedCanvas.height);
  uploadedImage.src = '';
  audioControls.style.display = 'none';
  
  // Reset the upload input to allow re-uploading the same file
  uploadInput.value = '';
});


// Access webcam
async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({video:true});
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
  } catch (e) {
    alert('Cannot access webcam. Please allow permission or use photo upload.');
  }
}

startWebcam();
</script>

</body>
</html>
"""

def read_image_from_request():
    if 'image' not in request.files:
        return None, "No image file"
    file = request.files['image']
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    in_memory_file.seek(0)
    image = Image.open(in_memory_file).convert('RGB')
    return np.array(image), None

@app.route('/')
def index():
    return render_template_string(HTML_PAGE)

@app.route('/detect', methods=['POST'])
def detect():
    img, err = read_image_from_request()
    if err or img is None:
        return jsonify({'error': 'Invalid image'}), 400

    # Run YOLOv8 detection
    results = model(img)

    # Parse results to send bounding box coords and labels
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            conf = box.conf[0].cpu().item()
            cls = int(box.cls[0].cpu())
            label = model.names[cls]
            if conf > 0.3:  # filter by confidence threshold
                detections.append({
                    'label': label,
                    'confidence': round(conf, 2),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })

    return jsonify({'detections': detections})

if __name__ == '__main__':
    print("Starting YOLOv8 detection web app on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)