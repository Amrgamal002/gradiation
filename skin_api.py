from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import torch
from torchvision import transforms

app = FastAPI()

# تحميل الموديل (غير المسار حسب ملفك الحقيقي)
model = YOLO("D:/grad_project/skin_yolo_project/yolov8_skin_cls42/weights/best.pt")

# أسماء الكلاسات حسب ترتيب مجلدات val
class_names = [
    "BA-cellulitis", "BA-impetigo", "FU-athlete-foot", "FU-nail-fungus",
    "FU-ringworm", "PA-cutaneous-larva-migrans", "VI-chickenpox", "VI-shingles"
]

# تحضير الصورة
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        result = model(img_tensor, verbose=False)
        probs = result[0].probs.data  # حولناها لتنسور
        pred_idx = torch.argmax(probs).item()
        confidence = float(probs[pred_idx])


    return {
        "class": class_names[pred_idx],
        "confidence": round(confidence * 100, 2)
    }
print("✅ File loaded successfully")
