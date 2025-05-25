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
from datetime import datetime, timedelta
from collections import defaultdict

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
current_densities = {}  # Current real-time densities (updated every 15 seconds)
today_densities = {}    # All density readings for today
critical_densities = {} # Critical density thresholds (max from previous day)
last_update_time = None
is_processing = False

# Parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
UPDATE_INTERVAL = 15  # Update every 15 seconds

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

# Initialize models
road_model = None
vehicle_model = None

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
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        logging.info(f"✓ Model {model_path} loaded successfully")
        return model
    except Exception as e:
        logging.error(f"Failed to load model {model_path}: {e}")
        raise

# Download model from GitHub
def download_model_from_github(model_name, download_url):
    if not os.path.exists(model_name):
        logging.info(f"Downloading {model_name} from GitHub...")
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Railway-Server) AppleWebKit/537.36',
                'Accept': 'application/octet-stream, */*'
            }
            
            response = requests.get(download_url, headers=headers, timeout=180, stream=True)
            response.raise_for_status()
            
            with open(model_name, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size = os.path.getsize(model_name)
            logging.info(f"✓ {model_name} downloaded - {file_size/1024/1024:.1f}MB")
            
        except Exception as e:
            logging.error(f"Failed to download {model_name}: {e}")
            raise
    else:
        logging.info(f"✓ {model_name} already exists")

def initialize_models():
    global road_model, vehicle_model
    
    try:
        logging.info("=== Model Initialization Starting ===")
        
        # Download models if needed
        model_urls = {
            "unet_road_segmentation.Better.keras": "https://github.com/PhoneWaiYanMoe/GetDensity/releases/download/v1.0/unet_road_segmentation.Better.keras",
            "unet_multi_classV1.keras": "https://github.com/PhoneWaiYanMoe/GetDensity/releases/download/v1.0/unet_multi_classV1.keras"
        }
        
        for model_name, url in model_urls.items():
            try:
                download_model_from_github(model_name, url)
            except Exception as e:
                logging.error(f"Failed to download {model_name}: {e}")
                continue
        
        # Load models
        if os.path.exists("unet_road_segmentation.Better.keras"):
            road_model = load_trained_model("unet_road_segmentation.Better.keras", custom_objects={"dice_loss": dice_loss})
        
        if os.path.exists("unet_multi_classV1.keras"):
            vehicle_model = load_trained_model("unet_multi_classV1.keras", custom_objects={"dice_loss": dice_loss})
        
        if road_model and vehicle_model:
            logging.info("=== ✓ ALL MODELS LOADED SUCCESSFULLY ===")
        else:
            logging.warning("=== ⚠ SOME MODELS FAILED TO LOAD ===")
        
    except Exception as e:
        logging.error(f"Model initialization failed: {e}")

# Function to manage historical densities and critical densities
def manage_historical_densities():
    global today_densities, critical_densities
    
    today = datetime.now().date()
    today_str = today.strftime('%Y-%m-%d')
    yesterday = (today - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Check if it's a new day
    if not today_densities or today_densities.get('date') != today_str:
        logging.info(f"New day detected: {today_str}")
        
        # If we have data from previous day, calculate critical densities
        if today_densities and 'date' in today_densities and today_densities['date'] == yesterday:
            logging.info("Calculating critical densities from yesterday's data...")
            new_critical = {}
            
            for camera_id in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']:
                if camera_id in today_densities:
                    # Find the maximum density for this camera yesterday
                    max_density = 0.0
                    for timestamp, density in today_densities[camera_id].items():
                        if isinstance(density, (int, float)) and density > max_density:
                            max_density = density
                    new_critical[camera_id] = max_density
                    logging.info(f"Camera {camera_id}: Critical density = {max_density:.2f}%")
            
            if new_critical:
                critical_densities = new_critical
                logging.info(f"✓ Updated critical densities: {critical_densities}")
        
        # Reset today's densities for new day
        today_densities = {'date': today_str}
        logging.info("Reset today's densities for new day")

    # Initialize critical densities if empty (first run)
    if not critical_densities:
        # Hardcoded initial values based on typical HCMC traffic patterns
        critical_densities = {
            'A': 75.0,  # Lý Thái Tổ - Sư Vạn Hạnh (busy business district)
            'B': 65.0,  # Ba Tháng Hai - Cao Thắng (moderate traffic)
            'C': 70.0,  # Điện Biên Phủ – Cao Thắng (busy road)
            'D': 85.0,  # Ngã sáu Nguyễn Tri Phương_1 (very busy intersection)
            'E': 80.0,  # Ngã sáu Nguyễn Tri Phương (busy intersection)
            'F': 55.0,  # Lê Đại Hành 2 (less congested area)
            'G': 68.0,  # Lý Thái Tổ - Nguyễn Đình Chiểu (moderate)
            'H': 90.0,  # Ngã sáu Cộng Hòa_1 (very busy, near airport)
            'I': 85.0,  # Ngã sáu Cộng Hòa (busy, near airport)
            'J': 72.0,  # Điện Biên Phủ - Cách Mạng Tháng Tám (busy)
            'K': 78.0,  # Công Trường Dân Chủ (city center)
            'L': 76.0   # Công Trường Dân Chủ_1 (city center)
        }
        logging.info(f"✓ Initialized critical densities (hardcoded): {critical_densities}")

# Image processing functions
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

def postprocess_road_mask(prediction):
    try:
        prediction = prediction.squeeze()
        return (prediction > 0.5).astype(np.uint8)
    except Exception as e:
        logging.error(f"Error postprocessing road mask: {e}")
        raise

def postprocess_vehicle_mask(prediction):
    try:
        prediction = prediction.squeeze()
        return np.argmax(prediction, axis=-1)
    except Exception as e:
        logging.error(f"Error postprocessing vehicle mask: {e}")
        raise

def extract_segmented_road(original_image, road_mask):
    try:
        mask_resized = cv2.resize(road_mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        segmented_road = cv2.bitwise_and(original_image, original_image, mask=mask_resized.astype(np.uint8) * 255)
        return segmented_road, mask_resized
    except Exception as e:
        logging.error(f"Error extracting segmented road: {e}")
        raise

# Main processing function
def fetch_and_process_densities():
    global current_densities, today_densities, last_update_time, is_processing
    
    if is_processing:
        logging.info("Previous processing still running, skipping this cycle")
        return
    
    if road_model is None or vehicle_model is None:
        logging.warning("Models not loaded, skipping density processing")
        return
    
    is_processing = True
    
    try:
        # Visit main webpage to get cookies
        try:
            response = session.get(main_url, timeout=10)
            logging.info("Visited main webpage to get cookies")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to visit main webpage: {e}")
            return

        # Manage historical densities
        manage_historical_densities()

        new_densities = {}
        current_time = datetime.now()

        for cam_id, cam_location in cameras:
            camera_id = camera_mapping.get(cam_location)
            if not camera_id:
                logging.warning(f"No camera mapping for {cam_location}, skipping")
                continue

            logging.info(f"Processing camera {camera_id} ({cam_location})")

            # Fetch the live image
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
                except requests.exceptions.RequestException as e:
                    logging.warning(f"Camera {cam_id}, Attempt {attempt + 1}: {e}")
                    time.sleep(2)
            
            if img is None:
                logging.error(f"Camera {cam_id}: Failed to fetch image after 3 attempts")
                continue

            # Process the image
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
                
                # Calculate density
                vehicle_pixels = np.count_nonzero(vehicle_mask_resized)
                road_pixels = np.count_nonzero(mask_resized)
                raw_density = (vehicle_pixels / road_pixels) * 100 if road_pixels > 0 else 0
                raw_density = min(raw_density, 100.0)

                # Get critical density for normalization
                kc = critical_densities.get(camera_id, 100.0)
                normalized_density = (raw_density / kc) * 100
                normalized_density = min(normalized_density, 100.0)

                # Store current density
                new_densities[camera_id] = {
                    'density': round(normalized_density, 2),
                    'raw_density': round(raw_density, 2),
                    'critical_density': kc,
                    'location': cam_location,
                    'timestamp': current_time.isoformat(),
                    'status': 'online'
                }

                # Store in today's densities
                if camera_id not in today_densities:
                    today_densities[camera_id] = {}
                timestamp_key = current_time.strftime('%H:%M:%S')
                today_densities[camera_id][timestamp_key] = raw_density

                logging.info(f"Camera {camera_id}: Raw={raw_density:.2f}%, Normalized={normalized_density:.2f}%, Critical={kc:.2f}%")

            except Exception as e:
                logging.error(f"Error processing image for {cam_location}: {e}")
                continue

        # Update global current densities
        current_densities = new_densities
        last_update_time = current_time.isoformat()
        logging.info(f"✓ Density update completed - {len(new_densities)} cameras processed")

    except Exception as e:
        logging.error(f"Error in fetch_and_process_densities: {e}")
    finally:
        is_processing = False

# Background thread function
def background_processor():
    while True:
        try:
            logging.info(f"Starting density fetch cycle at {datetime.now()}")
            fetch_and_process_densities()
            logging.info(f"Waiting {UPDATE_INTERVAL} seconds before next cycle...")
            time.sleep(UPDATE_INTERVAL)
        except Exception as e:
            logging.error(f"Error in background processor: {e}")
            time.sleep(UPDATE_INTERVAL)

# =============================================================================
# API ROUTES
# =============================================================================

@app.route('/')
def home():
    """API Information"""
    return jsonify({
        "message": "Ho Chi Minh City Traffic Density API",
        "version": "1.0.0",
        "status": "running",
        "last_update": last_update_time,
        "update_interval": f"{UPDATE_INTERVAL} seconds",
        "total_cameras": len(cameras),
        "models_loaded": road_model is not None and vehicle_model is not None,
        "endpoints": {
            "/densities": "Get current densities of all cameras (updated every 15s)",
            "/densities/<camera_id>": "Get specific camera density (A-L)",
            "/cameras": "List all cameras with details",
            "/today-densities": "Get all density readings for today",
            "/today-densities/<camera_id>": "Get today's readings for specific camera",
            "/critical-densities": "Get critical density thresholds",
            "/status": "Get API status and health check"
        }
    })

@app.route('/densities')
def get_current_densities():
    """Get current densities of all cameras (updated every 15 seconds)"""
    return jsonify({
        "message": "Current traffic densities for all cameras",
        "data": current_densities,
        "last_update": last_update_time,
        "update_interval": f"{UPDATE_INTERVAL} seconds",
        "cameras_online": len(current_densities),
        "total_cameras": len(cameras),
        "timestamp": datetime.now().isoformat(),
        "status": "success"
    })

@app.route('/densities/<camera_id>')
def get_camera_density(camera_id):
    """Get current density for a specific camera"""
    camera_id = camera_id.upper()
    
    if camera_id in current_densities:
        return jsonify({
            "camera_id": camera_id,
            "data": current_densities[camera_id],
            "last_update": last_update_time,
            "status": "success"
        })
    else:
        return jsonify({
            "error": f"Camera {camera_id} not found or offline",
            "available_cameras": sorted(list(current_densities.keys())),
            "status": "error"
        }), 404

@app.route('/cameras')
def get_all_cameras():
    """List all cameras with their current status"""
    camera_info = []
    
    for cam_id, cam_location in cameras:
        camera_id = camera_mapping.get(cam_location)
        is_online = camera_id in current_densities
        current_data = current_densities.get(camera_id, {})
        
        camera_info.append({
            "id": camera_id,
            "internal_id": cam_id,
            "location": cam_location,
            "online": is_online,
            "current_density": current_data.get('density', None),
            "raw_density": current_data.get('raw_density', None),
            "critical_density": critical_densities.get(camera_id, None),
            "last_update": current_data.get('timestamp', None)
        })
    
    return jsonify({
        "message": "All traffic cameras in Ho Chi Minh City",
        "cameras": camera_info,
        "total_cameras": len(camera_info),
        "online_cameras": len(current_densities),
        "offline_cameras": len(camera_info) - len(current_densities),
        "last_update": last_update_time,
        "status": "success"
    })

@app.route('/today-densities')
def get_today_densities():
    """Get all density readings for today"""
    return jsonify({
        "message": "All density readings for today",
        "date": today_densities.get('date', datetime.now().strftime('%Y-%m-%d')),
        "data": {k: v for k, v in today_densities.items() if k != 'date'},
        "total_cameras": len([k for k in today_densities.keys() if k != 'date']),
        "status": "success"
    })

@app.route('/today-densities/<camera_id>')
def get_today_camera_densities(camera_id):
    """Get today's density readings for a specific camera"""
    camera_id = camera_id.upper()
    
    if camera_id in today_densities and camera_id != 'date':
        camera_data = today_densities[camera_id]
        
        # Calculate statistics
        if camera_data:
            densities = list(camera_data.values())
            stats = {
                "min": round(min(densities), 2),
                "max": round(max(densities), 2),
                "avg": round(sum(densities) / len(densities), 2),
                "readings_count": len(densities)
            }
        else:
            stats = {"min": 0, "max": 0, "avg": 0, "readings_count": 0}
        
        return jsonify({
            "camera_id": camera_id,
            "date": today_densities.get('date'),
            "location": next((cam[1] for cam in cameras if camera_mapping.get(cam[1]) == camera_id), "Unknown"),
            "readings": camera_data,
            "statistics": stats,
            "status": "success"
        })
    else:
        return jsonify({
            "error": f"No data found for camera {camera_id} today",
            "available_cameras": [k for k in today_densities.keys() if k != 'date'],
            "status": "error"
        }), 404

@app.route('/critical-densities')
def get_critical_densities():
    """Get critical density thresholds for all cameras"""
    camera_details = []
    
    for camera_id, critical_value in critical_densities.items():
        location = next((cam[1] for cam in cameras if camera_mapping.get(cam[1]) == camera_id), "Unknown")
        camera_details.append({
            "camera_id": camera_id,
            "location": location,
            "critical_density": critical_value,
            "description": f"Traffic density threshold for {location}"
        })
    
    return jsonify({
        "message": "Critical density thresholds (updated daily at midnight)",
        "description": "These values represent the maximum density from the previous day and are used for normalization",
        "critical_densities": critical_densities,
        "camera_details": camera_details,
        "last_updated": "Daily at 00:00 (midnight)",
        "total_cameras": len(critical_densities),
        "status": "success"
    })

@app.route('/status')
def get_api_status():
    """Get API status and health check"""
    return jsonify({
        "status": "healthy",
        "service": "HCMC Traffic Density API",
        "version": "1.0.0",
        "last_update": last_update_time,
        "processing": is_processing,
        "update_interval": f"{UPDATE_INTERVAL} seconds",
        "cameras": {
            "total": len(cameras),
            "online": len(current_densities),
            "offline": len(cameras) - len(current_densities)
        },
        "models": {
            "road_model_loaded": road_model is not None,
            "vehicle_model_loaded": vehicle_model is not None,
            "both_loaded": road_model is not None and vehicle_model is not None
        },
        "data_status": {
            "current_densities_count": len(current_densities),
            "today_cameras_tracked": len([k for k in today_densities.keys() if k != 'date']),
            "critical_densities_set": len(critical_densities)
        },
        "uptime": datetime.now().isoformat(),
        "memory_usage": f"{psutil.virtual_memory().percent}%"
    })

@app.route('/health')
def health_check():
    """Simple health check for load balancer"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": road_model is not None and vehicle_model is not None
    })

# Initialize the application
if __name__ == '__main__':
    try:
        logging.info("=== HO CHI MINH CITY TRAFFIC DENSITY API ===")
        logging.info("=== STARTING INITIALIZATION ===")
        
        # Initialize models and data
        initialize_models()
        manage_historical_densities()  # Initialize data structures
        
        # Start background processing thread
        processor_thread = threading.Thread(target=background_processor, daemon=True)
        processor_thread.start()
        logging.info(f"✓ Background processor started (update interval: {UPDATE_INTERVAL}s)")
        
        # Get port from environment
        port = int(os.environ.get('PORT', 5000))
        
        logging.info("=== ✓ APPLICATION READY ===")
        logging.info(f"✓ API available at port {port}")
        logging.info(f"✓ Current densities update every {UPDATE_INTERVAL} seconds")
        logging.info(f"✓ Critical densities update daily at midnight")
        
        # Start Flask app
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
        
    except Exception as e:
        logging.error("=== ✗ APPLICATION STARTUP FAILED ===")
        logging.error(f"Error: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        exit(1)