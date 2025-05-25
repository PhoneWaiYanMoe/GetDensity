import requests
import cv2
import numpy as np
import tensorflow as tf
import time
import os
import json
import shutil
import traceback
import logging
import threading
import sys
from flask import Flask, jsonify, request
from flask_cors import CORS
import psutil
import gc
from datetime import datetime

# Disable GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Set up logging with immediate flushing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ],
    force=True
)
logging.getLogger().setLevel(logging.INFO)
logging.info(f"Starting application, TensorFlow version: {tf.__version__}")
logging.info(f"Initial memory usage: {psutil.virtual_memory().percent}%")

# Flask app setup
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global variables for in-memory storage
current_densities = {}
today_densities = {}
critical_densities = {}
historical_data = {}
last_update_time = None
is_processing = False

# Parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128

# Dice Loss
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth))

# Load Models
def load_trained_model(model_path, custom_objects=None):
    try:
        logging.info(f"Loading model: {model_path}")
        logging.info(f"Memory usage before loading: {psutil.virtual_memory().percent}%")
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        logging.info(f"Memory usage after loading: {psutil.virtual_memory().percent}%")
        return model
    except Exception as e:
        logging.error(f"Failed to load model {model_path}: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise

# Check system resources
def check_system_resources():
    try:
        total, used, free = shutil.disk_usage('/')
        total_mb = total // 1024 // 1024
        used_mb = used // 1024 // 1024
        free_mb = free // 1024 // 1024
        logging.info(f"Disk space - Total: {total_mb}MB, Used: {used_mb}MB, Free: {free_mb}MB")
        if free < 300 * 1024 * 1024:  # 300MB
            logging.warning("Low disk space detected!")
            return False
        return True
    except Exception as e:
        logging.error(f"Error checking system resources: {e}")
        return True

# Initialize models (skip downloads for testing, assume models are in repo)
road_model = None
vehicle_model = None

def initialize_models():
    global road_model, vehicle_model
    try:
        logging.info("=== Model Initialization Starting ===")
        logging.info(f"Current directory: {os.getcwd()}")
        logging.info(f"Directory contents: {os.listdir('.')}")
        logging.info(f"Memory usage: {psutil.virtual_memory().percent}%")
        if not check_system_resources():
            logging.warning("System resources check failed, but continuing...")
        logging.info("=== Loading TensorFlow Models ===")
        logging.info("Loading road segmentation model...")
        road_model = load_trained_model("unet_road_segmentation.Better.keras", custom_objects={"dice_loss": dice_loss})
        gc.collect()
        logging.info("✓ Road model loaded")
        logging.info("Loading vehicle classification model...")
        vehicle_model = load_trained_model("unet_multi_classV1.keras", custom_objects={"dice_loss": dice_loss})
        gc.collect()
        logging.info("✓ Vehicle model loaded")
        logging.info("=== ✓ ALL MODELS LOADED SUCCESSFULLY ===")
    except Exception as e:
        logging.error("=== ✗ MODEL INITIALIZATION FAILED ===")
        logging.error(f"Error: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        raise

# Function to manage historical density in memory
def manage_historical_densities():
    global today_densities, critical_densities
    today = datetime.now().date()
    if not today_densities or today_densities.get('date') != today.strftime('%Y-%m-%d'):
        if today_densities and 'date' in today_densities:
            new_critical = {}
            for cam_id in today_densities:
                if cam_id != 'date':
                    timestamps = today_densities[cam_id]
                    max_density = max(timestamps.values()) if timestamps else 0.0
                    new_critical[cam_id] = max_density
            if new_critical:
                critical_densities = new_critical
                logging.info(f"Updated critical density from yesterday's max: {critical_densities}")
        today_densities = {'date': today.strftime('%Y-%m-%d')}
    if not critical_densities:
        sample_critical_densities = {
            'A': 80.0, 'B': 70.0, 'C': 75.0, 'D': 85.0, 'E': 80.0, 'F': 60.0,
            'G': 70.0, 'H': 90.0, 'I': 85.0, 'J': 75.0, 'K': 80.0, 'L': 80.0
        }
        critical_densities = {}
        for cam in cameras:
            cam_location = cam[1]
            camera_id = camera_mapping.get(cam_location)
            if camera_id:
                critical_densities[camera_id] = sample_critical_densities.get(camera_id, 100.0)
        logging.info(f"Initialized sample critical density: {critical_densities}")

# Main processing function
def fetch_and_process_densities():
    global current_densities, today_densities, last_update_time, is_processing
    if is_processing:
        logging.info("Previous processing still running, skipping this cycle")
        return
    is_processing = True
    try:
        try:
            response = session.get(main_url, timeout=10)
            logging.info("Visited main webpage to get cookies.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to visit main webpage: {e}")
            return
        manage_historical_densities()
        new_densities = {}
        for cam_id, cam_location in cameras:
            camera_id = camera_mapping.get(cam_location)
            if not camera_id:
                logging.warning(f"No camera mapping for {cam_location}, skipping.")
                continue
            logging.info(f"Processing camera {camera_id} ({cam_location})")
            params = default_params.copy()
            params["id"] = cam_id
            img = None
            for attempt in range(3):
                try:
                    response = session.get(base_url, params=params, timeout=10)
                    if response.status_code == 200:
                        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if img is not None:
                            break
                        else:
                            logging.warning(f"Camera {cam_id} ({cam_location}), Attempt {attempt + 1}: Failed to decode image.")
                    else:
                        logging.warning(f"Camera {cam_id} ({cam_location}), Attempt {attempt + 1}: Failed to fetch image: {response.status_code}")
                    time.sleep(2)
                except requests.exceptions.RequestException as e:
                    logging.warning(f"Camera {cam_id} ({cam_location}), Attempt {attempt + 1}: Network error: {e}")
                    time.sleep(2)
            if img is None:
                logging.error(f"Camera {cam_id} ({cam_location}): Failed to fetch or decode image after 3 attempts. Skipping.")
                continue
            try:
                img_processed = preprocess_image(img)
                road_pred = road_model.predict(img_processed, verbose=0)
                road_mask = postprocess_road_mask(road_pred)
                segmented_road, mask_resized = extract_segmented_road(img, road_mask)
                segmented_road_resized = cv2.resize(segmented_road, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
                segmented_road_resized = np.expand_dims(segmented_road_resized, axis=0)
                vehicle_pred = vehicle_model.predict(segmented_road_resized, verbose=0)
                vehicle_mask = postprocess_vehicle_mask(vehicle_pred)
                vehicle_mask_resized = cv2.resize(vehicle_mask.astype(np.uint8),
                                                (img.shape[1], img.shape[0]),
                                                interpolation=cv2.INTER_NEAREST)
                vehicle_pixels = np.count_nonzero(vehicle_mask_resized)
                road_pixels = np.count_nonzero(mask_resized)
                vehicle_density = (vehicle_pixels / road_pixels) * 100 if road_pixels > 0 else 0
                vehicle_density = min(vehicle_density, 100.0)
                kc = critical_densities.get(camera_id, 100.0)
                adjusted_density = (vehicle_density / kc) * 100
                adjusted_density = min(adjusted_density, 100.0)
                new_densities[camera_id] = {
                    'density': round(adjusted_density, 2),
                    'raw_density': round(vehicle_density, 2),
                    'critical_density': kc,
                    'location': cam_location,
                    'timestamp': datetime.now().isoformat()
                }
                logging.info(f"Camera {camera_id} ({cam_location}): Density = {adjusted_density:.2f}% (Raw: {vehicle_density:.2f}%, kc: {kc})")
                if camera_id not in today_densities:
                    today_densities[camera_id] = {}
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                today_densities[camera_id][timestamp] = vehicle_density
            except Exception as e:
                logging.error(f"Error processing image for {cam_location}: {e}")
                continue
        current_densities = new_densities
        last_update_time = datetime.now().isoformat()
        logging.info(f"Density update completed at {last_update_time}")
    except Exception as e:
        logging.error(f"Error in fetch_and_process_densities: {e}")
    finally:
        is_processing = False

# Background thread function
def background_processor():
    while True:
        try:
            logging.info(f"Starting new density fetch cycle at {datetime.now()}")
            fetch_and_process_densities()
            logging.info("Waiting 20 seconds before next cycle...")
            time.sleep(20)
        except Exception as e:
            logging.error(f"Error in background processor: {e}")
            time.sleep(20)

# API Routes
@app.route('/')
def home():
    return jsonify({
        "message": "Ho Chi Minh City Traffic Density API",
        "status": "running",
        "last_update": last_update_time,
        "total_cameras": len(cameras),
        "models_loaded": road_model is not None and vehicle_model is not None,
        "endpoints": {
            "/densities": "Get current traffic density",
            "/densities/<camera_id>": "Get specific camera density",
            "/status": "Get API status",
            "/cameras": "Get all camera locations"
        }
    })

@app.route('/densities')
def get_densities():
    return jsonify({
        "densities": current_densities,
        "last_update": last_update_time,
        "total_cameras": len(current_densities),
        "status": "success"
    })

@app.route('/densities/<camera_id>')
def get_camera_density(camera_id):
    camera_id = camera_id.upper()
    if camera_id in current_densities:
        return jsonify({
            "camera_id": camera_id,
            "data": current_densities[camera_id],
            "status": "success"
        })
    else:
        return jsonify({
            "error": f"Camera {camera_id} not found",
            "available_cameras": list(current_densities.keys()),
            "status": "error"
        }), 404

@app.route('/status')
def get_status():
    return jsonify({
        "status": "healthy",
        "last_update": last_update_time,
        "processing": is_processing,
        "cameras_online": len(current_densities),
        "total_cameras": len(cameras),
        "uptime": datetime.now().isoformat(),
        "models_loaded": {
            "road_model": road_model is not None,
            "vehicle_model": vehicle_model is not None
        }
    })

@app.route('/cameras')
def get_cameras():
    camera_info = []
    for cam_id, cam_location in cameras:
        camera_id = camera_mapping.get(cam_location)
        camera_info.append({
            "id": camera_id,
            "internal_id": cam_id,
            "location": cam_location,
            "online": camera_id in current_densities if current_densities else False
        })
    return jsonify({
        "cameras": camera_info,
        "total": len(camera_info),
        "status": "success"
    })

@app.route('/historical/<camera_id>')
def get_historical_data(camera_id):
    camera_id = camera_id.upper()
    if camera_id in today_densities and camera_id != 'date':
        return jsonify({
            "camera_id": camera_id,
            "date": today_densities.get('date'),
            "data": today_densities[camera_id],
            "status": "success"
        })
    else:
        return jsonify({
            "error": f"No historical data for camera {camera_id}",
            "status": "error"
        }), 404

# Camera mapping
camera_mapping = {
    'Lý Thái Tổ - Sư Vạn Hạnh': 'A',
    'Ba Tháng Hai - Cao Thắng': 'B',
    'Điện Biên Phủ – Cao Thắng': 'C',
    'Nút giao Ngã sáu Nguyễn Tri Phương_1': 'D',
    'Nút giao Ngã sáu Nguyễn Tri Phương': 'E',
    'Nút giao Lê Đại Hành 2 (Lê Đại Hành)': 'F',
    'Lý Thái Tổ - Nguyễn Đình Chiểu': 'G',
    'Nút giao Ngã sáu Cộng Hòa_1': 'H',
    'Nút giao Ngã sáu Cộng Hòa': 'I',
    'Điện Biên Phủ - Cách Mạng Tháng Tám': 'J',
    'Nút giao Công Trường Dân Chủ': 'K',
    'Nút giao Công Trường Dân Chủ_1': 'L'
}

# List of cameras with their IDs and locations
cameras = [
    ("6623e7076f998a001b2523ea", "Lý Thái Tổ - Sư Vạn Hạnh"),
    ("5deb576d1dc17d7c5515acf8", "Ba Tháng Hai - Cao Thắng"),
    ("63ae7a9cbfd3d90017e8f303", "Điện Biên Phủ – Cao Thắng"),
    ("5deb576d1dc17d7c5515ad21", "Nút giao Ngã sáu Nguyễn Tri Phương"),
    ("5deb576d1dc17d7c5515ad22", "Nút giao Ngã sáu Nguyễn Tri Phương_1"),
    ("5d8cdd26766c880017188974", "Nút giao Lê Đại Hành 2 (Lê Đại Hành)"),
    ("63ae763bbfd3d90017e8f0c4", "Lý Thái Tổ - Nguyễn Đình Chiểu"),
    ("5deb576d1dc17d7c5515acf6", "Nút giao Ngã sáu Cộng Hòa"),
    ("5deb576d1dc17d7c5515acf7", "Nút giao Ngã sáu Cộng Hòa_1"),
    ("5deb576d1dc17d7c5515acf2", "Điện Biên Phủ - Cách Mạng Tháng Tám"),
    ("5deb576d1dc17d7c5515acf9", "Nút giao Công Trường Dân Chủ"),
    ("5deb576d1dc17d7c5515acfa", "Nút giao Công Trường Dân Chủ_1")
]

# Create a session to persist cookies
session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
})

# Base URL and default parameters for the camera feed
main_url = "https://giaothong.hochiminhcity.gov.vn"
base_url = "https://giaothong.hochiminhcity.gov.vn:8007/Render/CameraHandler.ashx"
default_params = {
    "bg": "black",
    "w": 300,
    "h": 230
}