from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
from werkzeug.utils import secure_filename
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from supabase import create_client, Client
from datetime import datetime, timezone
import logging
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
SCOPES = ['https://www.googleapis.com/auth/drive.file']  # Hardcode to avoid parsing issues

# Flask and logging
app = Flask(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# ⚡ Hardcoded class names from model training
CLASS_NAMES = {
    0: 'Basketball',
    1: 'Cricket',
    2: 'Rugby',
    3: 'badminton',
    4: 'boxing',
    5: 'football',
    6: 'swimming',
    7: 'wrestling'
}

# 🔧 Load model
MODEL_PATH = os.getenv("MODEL_PATH", "model/sports_classifier_efficientnetb3.h5")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("✅ Model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Model load failed: {e}")
    raise

target_size = model.input_shape[1:3]

# 🖼 Preprocessing to match training pipeline
def preprocess_image(file_path):
    img = Image.open(file_path).convert('RGB')
    img = img.resize(target_size)  # (300, 300)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# 🔗 Google Drive
def get_drive_service():
    # Load service account JSON from environment variable
    credentials_json = os.getenv("GOOGLE_CREDENTIALS")
    if not credentials_json:
        logging.error("GOOGLE_CREDENTIALS environment variable is not set")
        raise ValueError("GOOGLE_CREDENTIALS not set")

    # Ensure tmp directory exists locally; use /tmp for Vercel
    temp_dir = os.path.join(os.getcwd(), "tmp") if not os.getenv("VERCEL") else "/tmp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_cred_path = os.path.join(temp_dir, "service_account.json")

    try:
        with open(temp_cred_path, "w") as f:
            f.write(credentials_json)
        creds = service_account.Credentials.from_service_account_file(temp_cred_path, scopes=SCOPES)
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        logging.error(f"Failed to initialize Drive service: {e}")
        raise

def upload_to_drive(filepath):
    service = get_drive_service()
    file_metadata = {'name': os.path.basename(filepath), 'parents': [os.getenv("DRIVE_FOLDER_ID")]}
    media = MediaFileUpload(filepath, mimetype='image/jpeg')
    uploaded = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
    return uploaded.get('id')

# 🔗 Supabase
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'status': 'fail', 'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '' or not file.content_type.startswith('image/'):
        return jsonify({'status': 'fail', 'error': 'Invalid file'}), 400

    file.seek(0, os.SEEK_END)
    if file.tell() > MAX_FILE_SIZE:
        return jsonify({'status': 'fail', 'error': 'File too large'}), 400
    file.seek(0)

    filename = secure_filename(file.filename)
    temp_dir = os.path.join(os.getcwd(), "tmp") if not os.getenv("VERCEL") else "/tmp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"temp_{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}")

    try:
        file.save(temp_path)
        drive_file_id = upload_to_drive(temp_path)

        img = preprocess_image(temp_path)
        preds = model.predict(img)
        pred_index = int(np.argmax(preds[0]))

        confidence = float(preds[0][pred_index])
        result_class = CLASS_NAMES.get(pred_index, "Unknown")

        logging.info(f"Prediction: {result_class} ({confidence:.2f})")

        supabase.table('predictions').insert({ 
            'class': result_class, 
            'confidence': confidence, 
            'drive_file_id': drive_file_id, 
            'timestamp': datetime.now(timezone.utc).isoformat()
        }).execute()

        return jsonify({ 'status': 'success', 'class': result_class, 'confidence': confidence })
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'status': 'fail', 'error': 'Prediction failed'}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Vercel serverless function entry point
if __name__ == '__main__':
    app.run()