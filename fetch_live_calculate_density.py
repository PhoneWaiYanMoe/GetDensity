import requests
import cv2
import numpy as np
import tensorflow as tf
import time
import os
import json
from datetime import datetime, timedelta
import unicodedata
import logging
import threading
from flask import Flask, jsonify, request
from flask_cors import CORS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
        return tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    except Exception as e:
        logging.error(f"Failed to load model {model_path}: {e}")
        raise

# Download model from URL if not exists
def download_model_if_needed(model_name, download_url):
    if not os.path.exists(model_name):
        logging.info(f"Downloading {model_name}...")
        try:
            response = requests.get(download_url, timeout=60)
            response.raise_for_status()
            with open(model_name, 'wb') as f:
                f.write(response.content)
            logging.info(f"{model_name} downloaded successfully")
        except Exception as e:
            logging.error(f"Failed to download {model_name}: {e}")
            raise

# Preprocess Image
def preprocess_image(img):
    try:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        y = clahe.apply(y)
        enhanced_img = cv2.merge((y, cr, cb))
        img = cv2.cvtColor(enhanced_img, cv2.COLOR_YCrCb2BGR)
        img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        logging.error(f"Error preprocessing image: {e}")
        raise

# Post-process Road Segmentation
def postprocess_road_mask(prediction):
    try:
        prediction = prediction.squeeze()
        return (prediction > 0.5).astype(np.uint8)
    except Exception as e:
        logging.error(f"Error postprocessing road mask: {e}")
        raise

# Post-process Vehicle Segmentation
def postprocess_vehicle_mask(prediction):
    try:
        prediction = prediction.squeeze()
        return np.argmax(prediction, axis=-1)
    except Exception as e:
        logging.error(f"Error postprocessing vehicle mask: {e}")
        raise

# Extract Segmented Road
def extract_segmented_road(original_image, road_mask):
    try:
        mask_resized = cv2.resize(road_mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        segmented_road = cv2.bitwise_and(original_image, original_image, mask=mask_resized.astype(np.uint8) * 255)
        return segmented_road, mask_resized
    except Exception as e:
        logging.error(f"Error extracting segmented road: {e}")
        raise

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

# Initialize models (will be loaded when app starts)
road_model = None
vehicle_model = None

def initialize_models():
    global road_model, vehicle_model
    
    try:
        # Use the EXACT filenames from your GitHub release
        road_model_url = "https://github.com/PhoneWaiYanMoe/GetDensity/releases/download/v1.0/unet_road_segmentation.Better.keras"
        vehicle_model_url = "https://github.com/PhoneWaiYanMoe/GetDensity/releases/download/v1.0/unet_multi_classV1.keras"
        
        # Download models with MATCHING local filenames
        download_model_if_needed("unet_road_segmentation.Better.keras", road_model_url)
        download_model_if_needed("unet_multi_classV1.keras", vehicle_model_url)
        
        # Load models with the SAME filenames
        road_model = load_trained_model("unet_road_segmentation.Better.keras")
        vehicle_model = load_trained_model("unet_multi_classV1.keras")
        logging.info("Models loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load models: {e}")
        raise
    
    try:
        # Download models if they don't exist
        # Correct GitHub release URLs
        road_model_url = "https://github.com/PhoneWaiYanMoe/GetDensity/releases/download/v1.0/unet_road_segmentation.Better.keras"
        vehicle_model_url = "https://github.com/PhoneWaiYanMoe/GetDensity/releases/download/v1.0/unet_multi_classV1.keras"
        
        # Download models
        download_model_if_needed("unet_road_segmentation.keras", road_model_url)
        download_model_if_needed("unet_multi_classV1.keras", vehicle_model_url)
        
        # Load models
        road_model = load_trained_model("unet_road_segmentation.keras")
        vehicle_model = load_trained_model("unet_multi_classV1.keras")
        logging.info("Models loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load models: {e}")
        raise

# Function to manage historical densities in memory
def manage_historical_densities():
    global today_densities, critical_densities
    
    today = datetime.now().date()
    
    # Initialize today's densities if not exists or if it's a new day
    if not today_densities or today_densities.get('date') != today.strftime('%Y-%m-%d'):
        # If we have data from previous day, calculate max densities as critical
        if today_densities and 'date' in today_densities:
            new_critical = {}
            for cam_id in today_densities:
                if cam_id != 'date':
                    timestamps = today_densities[cam_id]
                    max_density = max(timestamps.values()) if timestamps else 0.0
                    new_critical[cam_id] = max_density
            if new_critical:
                critical_densities = new_critical
                logging.info(f"Updated critical densities from yesterday's max: {critical_densities}")
        
        # Reset today's densities for new day
        today_densities = {'date': today.strftime('%Y-%m-%d')}

    # Initialize critical densities if empty (first run)
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
        logging.info(f"Initialized sample critical densities: {critical_densities}")

# Main processing function
def fetch_and_process_densities():
    global current_densities, today_densities, last_update_time, is_processing
    
    if is_processing:
        logging.info("Previous processing still running, skipping this cycle")
        return
    
    is_processing = True
    
    try:
        # Visit the main webpage to get cookies
        try:
            response = session.get(main_url, timeout=10)
            logging.info("Visited main webpage to get cookies.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to visit main webpage: {e}")
            return

        # Manage historical densities
        manage_historical_densities()

        new_densities = {}

        for cam_id, cam_location in cameras:
            # Map camera location to ID
            camera_id = camera_mapping.get(cam_location)
            if not camera_id:
                logging.warning(f"No camera mapping for {cam_location}, skipping.")
                continue

            logging.info(f"Processing camera {camera_id} ({cam_location})")

            # Update the params with the current camera ID
            params = default_params.copy()
            params["id"] = cam_id

            # Fetch the live image
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

            # Process the image to calculate density
            try:
                img_processed = preprocess_image(img)
                # Step 1: Road Segmentation
                road_pred = road_model.predict(img_processed, verbose=0)
                road_mask = postprocess_road_mask(road_pred)
                # Step 2: Extract Segmented Road
                segmented_road, mask_resized = extract_segmented_road(img, road_mask)
                # Step 3: Vehicle Segmentation on Road
                segmented_road_resized = cv2.resize(segmented_road, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
                segmented_road_resized = np.expand_dims(segmented_road_resized, axis=0)
                vehicle_pred = vehicle_model.predict(segmented_road_resized, verbose=0)
                vehicle_mask = postprocess_vehicle_mask(vehicle_pred)
                vehicle_mask_resized = cv2.resize(vehicle_mask.astype(np.uint8), 
                                                (img.shape[1], img.shape[0]), 
                                                interpolation=cv2.INTER_NEAREST)
                # Step 4: Calculate Vehicle Density
                vehicle_pixels = np.count_nonzero(vehicle_mask_resized)
                road_pixels = np.count_nonzero(mask_resized)
                vehicle_density = (vehicle_pixels / road_pixels) * 100 if road_pixels > 0 else 0
                vehicle_density = min(vehicle_density, 100.0)

                # Use critical density for this camera
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

                # Update today's densities
                if camera_id not in today_densities:
                    today_densities[camera_id] = {}
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                today_densities[camera_id][timestamp] = vehicle_density

            except Exception as e:
                logging.error(f"Error processing image for {cam_location}: {e}")
                continue

        # Update global current densities
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
        "message": "Traffic Density API",
        "status": "running",
        "last_update": last_update_time,
        "total_cameras": len(cameras),
        "endpoints": {
            "/densities": "Get current traffic densities",
            "/densities/<camera_id>": "Get specific camera density",
            "/status": "Get API status",
            "/cameras": "Get all camera locations"
        }
    })

@app.route('/densities')
def get_densities():
    """Get all current traffic densities"""
    return jsonify({
        "densities": current_densities,
        "last_update": last_update_time,
        "total_cameras": len(current_densities),
        "status": "success"
    })

@app.route('/densities/<camera_id>')
def get_camera_density(camera_id):
    """Get density for a specific camera"""
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
    """Get API status and health check"""
    return jsonify({
        "status": "healthy",
        "last_update": last_update_time,
        "processing": is_processing,
        "cameras_online": len(current_densities),
        "total_cameras": len(cameras),
        "uptime": datetime.now().isoformat(),
        "models_loaded": road_model is not None and vehicle_model is not None
    })

@app.route('/cameras')
def get_cameras():
    """Get all camera locations and mappings"""
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
    """Get today's historical data for a specific camera"""
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

# Initialize and start the application
if __name__ == '__main__':
    try:
        # Initialize models
        initialize_models()
        
        # Start background processing thread
        processor_thread = threading.Thread(target=background_processor, daemon=True)
        processor_thread.start()
        logging.info("Background processor started")
        
        # Get port from environment (Railway sets this)
        port = int(os.environ.get('PORT', 5000))
        
        # Start Flask app
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
        
    except Exception as e:
        logging.error(f"Failed to start application: {e}")
        exit(1)