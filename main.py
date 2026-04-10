import os
import io
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Klasifikasi Sayuran API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ⚠️ LABEL TETAP SAMA PERSIS SEPERTI ANDA
SPECIES_LABELS = [    
    'CabaiKeriting', 'CabaiRawit', 'KubisHijau', 'KubisMerah',
    'TerongGelatikHijau', 'TerongKopekUngu', 'TomatCherry', 'TomatRoma'
]
QUALITY_LABELS = [    
    'CabaiBusuk', 'CabaiLayu', 'CabaiMatang', 'CabaiMuda',
    'KubisBusuk', 'KubisLayu', 'KubisMatang', 'KubisMuda',
    'TerongBusuk', 'TerongLayu', 'TerongMatang', 'TerongMuda',
    'TomatBusuk', 'TomatLayu', 'TomatMatang', 'TomatMuda'
]

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
species_model = None
quality_model = None
explanations_data = None

@app.on_event("startup")
async def load_models_and_data():
    global species_model, quality_model, explanations_data
    try:
        species_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "species_model.h5"))
        quality_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, "quality_model.h5"))
        print("✅ Kedua model berhasil dimuat.")
        
        exp_path = os.path.join(DATA_DIR, "explanations.json")
        with open(exp_path, 'r', encoding='utf-8') as f:
            explanations_data = json.load(f)
        print("✅ Data penjelasan berhasil dimuat.")
    except Exception as e:
        print(f"❌ Gagal memuat: {e}")
        raise RuntimeError("Pastikan file .h5 dan explanations.json ada di path yang benar.")

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    resample = getattr(Image, "Resampling", Image).LANCZOS
    img = img.resize((128, 128), resample)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar (JPG/PNG).")

    try:
        img_bytes = await file.read()
        img_tensor = preprocess_image(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Gagal memproses gambar: {str(e)}")

    try:
        # 1. Prediksi Spesies
        sp_probs = species_model.predict(img_tensor, verbose=0)[0]
        sp_idx = np.argmax(sp_probs)
        sp_pred = SPECIES_LABELS[sp_idx]
        sp_conf = float(np.max(sp_probs))

        # 2. Prediksi Kualitas
        q_probs = quality_model.predict(img_tensor, verbose=0)[0]
        q_idx = np.argmax(q_probs)
        q_pred = QUALITY_LABELS[q_idx]
        q_conf = float(np.max(q_probs))

        # 3. Ambil Penjelasan dari JSON
        sp_exp = explanations_data.get("species", {}).get(sp_pred, {})
        q_exp = explanations_data.get("quality", {}).get(q_pred, {})

        return {
            "species": sp_pred,
            "species_confidence": sp_conf,
            "quality": q_pred,
            "quality_confidence": q_conf,
            "species_explanation": sp_exp,
            "quality_explanation": q_exp
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal melakukan prediksi: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)