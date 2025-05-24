import os
import json
import time
import random
import threading
from datetime import datetime
from flask import Flask, jsonify
from flask_cors import CORS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Flask app setup
app = Flask(__name__)
CORS(app)

# Global variables
current_densities = {}
last_update_time = None

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

def generate_realistic_mock_density():
    """Generate realistic traffic density data"""
    global current_densities, last_update_time
    
    # Time-based density simulation
    current_hour = datetime.now().hour
    base_multiplier = 1.0
    
    # Rush hour patterns
    if 7 <= current_hour <= 9:
        base_multiplier = 1.8  # Morning rush
    elif 17 <= current_hour <= 19:
        base_multiplier = 2.0  # Evening rush
    elif 11 <= current_hour <= 13:
        base_multiplier = 1.3  # Lunch time
    elif current_hour >= 22 or current_hour <= 6:
        base_multiplier = 0.3  # Night time
    
    new_densities = {}
    for cam_id, cam_location in cameras:
        camera_id = camera_mapping.get(cam_location)
        if camera_id:
            # Different base densities for different locations
            location_bases = {
                'A': 65, 'B': 55, 'C': 70, 'D': 85, 'E': 80, 'F': 45,
                'G': 60, 'H': 90, 'I': 85, 'J': 75, 'K': 80, 'L': 75
            }
            
            base_density = location_bases.get(camera_id, 60)
            density = base_density * base_multiplier
            density *= random.uniform(0.8, 1.2)  # Random variation
            density = max(5, min(95, density))  # Keep in bounds
            
            new_densities[camera_id] = {
                'density': round(density, 2),
                'raw_density': round(density * 0.85, 2),
                'critical_density': location_bases.get(camera_id, 80),
                'location': cam_location,
                'timestamp': datetime.now().isoformat(),
                'status': 'simulation'
            }
    
    current_densities = new_densities
    last_update_time = datetime.now().isoformat()
    logging.info(f"Generated densities for {len(new_densities)} cameras")

def background_processor():
    """Background thread for data generation"""
    while True:
        try:
            generate_realistic_mock_density()
            time.sleep(30)  # Update every 30 seconds
        except Exception as e:
            logging.error(f"Error in background processor: {e}")
            time.sleep(30)

# API Routes
@app.route('/')
def home():
    return jsonify({
        "message": "HCMC Traffic Density API",
        "status": "running",
        "mode": "simulation",
        "last_update": last_update_time,
        "total_cameras": len(cameras),
        "endpoints": {
            "/densities": "Get all traffic densities",
            "/densities/<camera_id>": "Get specific camera (A-L)",
            "/status": "API status",
            "/cameras": "Camera information"
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
            "available_cameras": sorted(list(current_densities.keys())),
            "status": "error"
        }), 404

@app.route('/status')
def get_status():
    return jsonify({
        "status": "healthy",
        "mode": "simulation",
        "last_update": last_update_time,
        "cameras_online": len(current_densities),
        "total_cameras": len(cameras),
        "uptime": datetime.now().isoformat()
    })

@app.route('/cameras')
def get_cameras():
    camera_info = []
    for cam_id, cam_location in cameras:
        camera_id = camera_mapping.get(cam_location)
        current_data = current_densities.get(camera_id, {})
        camera_info.append({
            "id": camera_id,
            "internal_id": cam_id,
            "location": cam_location,
            "online": True,
            "current_density": current_data.get('density', 0)
        })
    
    return jsonify({
        "cameras": camera_info,
        "total": len(camera_info),
        "status": "success"
    })

if __name__ == '__main__':
    try:
        logging.info("=== HCMC Traffic Density API Starting ===")
        
        # Generate initial data
        generate_realistic_mock_density()
        logging.info("✓ Initial density data generated")
        
        # Start background thread
        processor_thread = threading.Thread(target=background_processor, daemon=True)
        processor_thread.start()
        logging.info("✓ Background processor started")
        
        # Get port
        port = int(os.environ.get('PORT', 5000))
        logging.info(f"✓ Starting server on port {port}")
        
        # Start Flask app
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
        
    except Exception as e:
        logging.error(f"✗ Failed to start application: {e}")
        exit(1)