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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

# List of cameras with their IDs and locations (updated to match camera_mapping)
cameras = [
    ("6623e7076f998a001b2523ea", "Lý Thái Tổ - Sư Vạn Hạnh"),
    ("5deb576d1dc17d7c5515acf8", "Ba Tháng Hai - Cao Thắng"),
    ("63ae7a9cbfd3d90017e8f303", "Điện Biên Phủ – Cao Thắng"),  # Fixed to use en dash
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

# Paths for models and output
road_model_path = "unet_road_segmentation (Better).keras"
vehicle_model_path = "unet_multi_classV1.keras"
base_directory = r"E:\playground\flutter_playground_vsc\Flutter_ggmap_project_Without_Sequential"
densities_dir = os.path.join(base_directory, "densities")
today_densities_path = os.path.join(densities_dir, "today_densities.json")
yesterday_max_densities_path = os.path.join(densities_dir, "yesterday_max_densities.json")
critical_densities_path = os.path.join(densities_dir, "critical_densities.json")
output_json_path = os.path.join(base_directory, "densities.json")

# Create directories if they don't exist
os.makedirs(densities_dir, exist_ok=True)

# Create a session to persist cookies n 
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

# Load models
try:
    road_model = load_trained_model(road_model_path)
    vehicle_model = load_trained_model(vehicle_model_path)
    logging.info("Models loaded successfully")
except Exception as e:
    logging.error(f"Failed to load models: {e}")
    exit(1)

# Function to manage historical densities
def manage_historical_densities():
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    day_before_yesterday = today - timedelta(days=2)

    # Initialize today’s densities
    today_densities = {}
    if os.path.exists(today_densities_path):
        try:
            with open(today_densities_path, 'r', encoding='utf-8') as f:
                today_densities = json.load(f)
            # Check if today_densities is from today
            if 'date' in today_densities:
                file_date = datetime.strptime(today_densities['date'], '%Y-%m-%d').date()
                if file_date != today:
                    # Move today’s densities to yesterday if it’s from a previous day
                    max_densities = {}
                    for cam_id in today_densities:
                        if cam_id != 'date':
                            timestamps = today_densities[cam_id]
                            max_density = max(timestamps.values()) if timestamps else 0.0
                            max_densities[cam_id] = max_density
                    with open(yesterday_max_densities_path, 'w', encoding='utf-8') as f:
                        json.dump({'date': yesterday.strftime('%Y-%m-%d'), **max_densities}, f, ensure_ascii=False)
                    today_densities = {'date': today.strftime('%Y-%m-%d')}
                    logging.info(f"Updated yesterday_max_densities.json with max densities from {file_date}")
            else:
                today_densities = {'date': today.strftime('%Y-%m-%d')}
        except Exception as e:
            logging.error(f"Error reading today_densities.json: {e}")
            today_densities = {'date': today.strftime('%Y-%m-%d')}
    else:
        today_densities = {'date': today.strftime('%Y-%m-%d')}

    # Load yesterday’s max densities as critical densities
    critical_densities = {}
    if os.path.exists(yesterday_max_densities_path):
        try:
            with open(yesterday_max_densities_path, 'r', encoding='utf-8') as f:
                yesterday_data = json.load(f)
                file_date = datetime.strptime(yesterday_data['date'], '%Y-%m-%d').date()
                if file_date == yesterday:
                    critical_densities = {k: v for k, v in yesterday_data.items() if k != 'date'}
                else:
                    # If yesterday’s data is older, delete it
                    os.remove(yesterday_max_densities_path)
                    logging.info(f"Deleted outdated yesterday_max_densities.json from {file_date}")
        except Exception as e:
            logging.error(f"Error reading yesterday_max_densities.json: {e}")

    # Save critical densities
    if critical_densities:
        with open(critical_densities_path, 'w', encoding='utf-8') as f:
            json.dump(critical_densities, f, ensure_ascii=False)
        logging.info(f"Saved critical densities to {critical_densities_path}")
    else:
        # Sample critical densities for the first run (based on typical traffic patterns)
        sample_critical_densities = {
            'A': 80.0,  # Lý Thái Tổ - Sư Vạn Hạnh (busy intersection)
            'B': 70.0,  # Ba Tháng Hai - Cao Thắng (moderate traffic)
            'C': 75.0,  # Điện Biên Phủ – Cao Thắng (busy road)
            'D': 85.0,  # Ngã sáu Nguyễn Tri Phương_1 (very busy)
            'E': 80.0,  # Ngã sáu Nguyễn Tri Phương (busy)
            'F': 60.0,  # Lê Đại Hành 2 (less busy)
            'G': 70.0,  # Lý Thái Tổ - Nguyễn Đình Chiểu (moderate)
            'H': 90.0,  # Ngã sáu Cộng Hòa_1 (very busy)
            'I': 85.0,  # Ngã sáu Cộng Hòa (busy)
            'J': 75.0,  # Điện Biên Phủ - Cách Mạng Tháng Tám (busy)
            'K': 80.0,  # Công Trường Dân Chủ (busy)
            'L': 80.0   # Công Trường Dân Chủ_1 (busy)
        }
        critical_densities = {}
        for cam in cameras:
            cam_location = cam[1]
            camera_id = camera_mapping.get(cam_location)
            if camera_id:
                critical_densities[camera_id] = sample_critical_densities.get(camera_id, 100.0)
            else:
                logging.warning(f"No mapping found for {cam_location}, skipping in critical densities")
        with open(critical_densities_path, 'w', encoding='utf-8') as f:
            json.dump(critical_densities, f, ensure_ascii=False)
        logging.info(f"Initialized sample critical densities to {critical_densities_path}: {critical_densities}")

    # Delete day before yesterday’s data if it exists
    if os.path.exists(yesterday_max_densities_path):
        try:
            with open(yesterday_max_densities_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                file_date = datetime.strptime(data['date'], '%Y-%m-%d').date()
                if file_date <= day_before_yesterday:
                    os.remove(yesterday_max_densities_path)
                    logging.info(f"Deleted data from {file_date} (day before yesterday or older)")
        except Exception as e:
            logging.error(f"Error checking/deleting old data: {e}")

    return today_densities, critical_densities

# Main processing function
def fetch_and_process_densities():
    # Visit the main webpage to get cookies
    try:
        response = session.get(main_url, timeout=10)
        logging.info("Visited main webpage to get cookies.")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to visit main webpage: {e}")
        return

    # Manage historical densities
    try:
        today_densities, critical_densities = manage_historical_densities()
        logging.info(f"Critical densities (from yesterday's max): {critical_densities}")
    except Exception as e:
        logging.error(f"Error managing historical densities: {e}")
        return

    current_densities = {}

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

            current_densities[camera_id] = adjusted_density
            logging.info(f"Camera {camera_id} ({cam_location}): Density = {adjusted_density:.2f}% (Raw: {vehicle_density:.2f}%, kc: {kc})")

            # Update today’s densities
            if camera_id not in today_densities:
                today_densities[camera_id] = {}
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            today_densities[camera_id][timestamp] = vehicle_density

        except Exception as e:
            logging.error(f"Error processing image for {cam_location}: {e}")
            continue

    # Save today’s densities
    try:
        with open(today_densities_path, 'w', encoding='utf-8') as f:
            json.dump(today_densities, f, ensure_ascii=False)
        logging.info(f"Today’s densities updated at {today_densities_path}")
    except Exception as e:
        logging.error(f"Error saving today_densities.json: {e}")

    # Save current densities for the app
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(current_densities, f, ensure_ascii=False)
        logging.info(f"Current densities saved to {output_json_path}")
    except Exception as e:
        logging.error(f"Error saving densities.json: {e}")

# Run the script in a loop
try:
    while True:
        logging.info(f"Starting new density fetch cycle at {datetime.now()}")
        fetch_and_process_densities()
        logging.info("Waiting 20 seconds before next cycle...")
        time.sleep(20)
except KeyboardInterrupt:
    logging.info("Program interrupted by user.")
except Exception as e:
    logging.error(f"An unexpected error occurred: {e}")
finally:
    logging.info("Script finished.")