# config.py — FreshSense AI Configuration

import os
import gdown


import os
import gdown

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR      = os.path.join(BASE_DIR, "models")
DETECTION_DIR   = os.path.join(MODELS_DIR, "detection")
PREDICTIONS_DIR = os.path.join(MODELS_DIR, "predictions")
MEMORY_PATH     = os.path.join(BASE_DIR, "data", "memory.json")

# Ensure directories exist
os.makedirs(DETECTION_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

# ---------------------------
# GOOGLE DRIVE FILE IDS
# ---------------------------

MODEL_URLS = {
    "detection": "1uOyoFdpGNAY6rK_eHR8N-NuyTA3JLwNp",

    "apple.pt": "1DY2iI9Mh2TCW_5tswXkG4rVq8KnAv1Ee",
    "grapes.pt": "1KLbhjtz0l6HIuiP7SrCp3eyYbuhMkAIo",
    "lemon.pt": "1M5Y8Mo87psR8aLtblqheQHf6ury-ReMq",
    "mango.pt": "1NQmrlbnZ1IbxrRhFqi_Re7SWlQRi3YsN",
    "papaya.pt": "1CKhXX2ZEHEAcTkPxWG8zFrYH-1QIP7vI",
    "paprika_pepper.pt": "18N_72-GMjTwdmkU7R4l_0WXj9FSVVWxQ",
    "strawberry.pt": "1LOf3s0U7jY_wmc5QQRChFI2NaIFk2Nmq",
    "tomato.pt": "1LOf3s0U7jY_wmc5QQRChFI2NaIFk2Nmq",
    "watermelon.pt": "1U5XCRbhdfkzoycYjhS3B8_tE5YFKQIQh",
}

# ---------------------------
# DOWNLOAD FUNCTION
# ---------------------------

def download_model(file_id, output_path):
    if not os.path.exists(output_path):
        print(f"Downloading {os.path.basename(output_path)}...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"{os.path.basename(output_path)} already exists.")

# ---------------------------
# DOWNLOAD ALL MODELS
# ---------------------------

# Detection model
DETECTION_MODEL = os.path.join(DETECTION_DIR, "yolo_fruits_and_vegetables.pt")
download_model(MODEL_URLS["detection"], DETECTION_MODEL)

# Prediction models
PREDICTION_MODELS = {}

for name, file_id in MODEL_URLS.items():
    if name == "detection":
        continue

    path = os.path.join(PREDICTIONS_DIR, name)
    download_model(file_id, path)
    PREDICTION_MODELS[name] = path


# ─── YOLO ─────────────────────────────────────────────────────────────────────
YOLO_CONF_THRESH = 0.15
YOLO_IOU_THRESH  = 0.45

# ─── Supported fruit types ────────────────────────────────────────────────────
SUPPORTED_FRUITS = [
    "apple", "grapes", "lemon", "mango", "papaya",
    "paprika_pepper", "strawberry", "tomato", "watermelon",
]

# ─── YOLO label → CNN label mapping ──────────────────────────────────────────
LABEL_MAP = {
    "apple":         "apple",
    "grape":         "grapes",
    "grapes":        "grapes",
    "strawberry":    "strawberry",
    "tomato":        "tomato",
    "lemon":         "lemon",
    "papaya":        "papaya",
    "watermelon":    "watermelon",
    "bell pepper":   "paprika_pepper",
    "capsicum":      "paprika_pepper",
    "pepper":        "paprika_pepper",
    "paprika":       "paprika_pepper",
    "paprika_pepper":"paprika_pepper",
    "mango":         "mango",
}

# ─── Shelf life table (initial values — agent will update these) ──────────────
DEFAULT_SHELF_LIFE = {
    "apple":          12,
    "grapes":         10,
    "lemon":          14,
    "mango":          10,
    "papaya":          8,
    "paprika_pepper":  9,
    "strawberry":      8,
    "tomato":         17,
    "watermelon":     15,
}

# ─── CNN prediction cutoff ────────────────────────────────────────────────────
CNN_CUTOFF = {
    "apple":          15,
    "grapes":          9,
    "lemon":          14,
    "mango":          10,
    "papaya":          8,
    "paprika_pepper":  9,
    "strawberry":      8,
    "tomato":         12,
    "watermelon":     15,
}

# ─── CNN day bias correction ──────────────────────────────────────────────────
CNN_DAY_BIAS = 1

# ─── Learning parameters ──────────────────────────────────────────────────────
ALPHA_INIT       = 0.7    # initial CNN reliability weight (per fruit ID)
ALPHA_SHELF      = 0.01    # shelf life learning rate
FRAME_LR         = 0.05   # frame-level learning rate

# ─── Tracking ─────────────────────────────────────────────────────────────────
BBOX_CENTRE_THRESH = 80   # px — max centre distance to match identities

# ─── Alerts ───────────────────────────────────────────────────────────────────
ALERT_ROTTEN       = 2    # remaining_days <= 2
ALERT_USE_NOW      = 3    # remaining_days <= 2
ALERT_EXPIRING     = 4    # remaining_days <= 3

# ─── Image preprocessing ──────────────────────────────────────────────────────
CNN_IMG_SIZE = 224
CNN_MEAN     = [0.485, 0.456, 0.406]
CNN_STD      = [0.229, 0.224, 0.225]
